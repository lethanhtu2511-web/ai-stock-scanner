
import os
import pandas as pd
import streamlit as st

from portfolio import (
    init_db,
    get_settings,
    save_settings,
    get_positions,
    get_transactions,
    get_portfolio_summary,
    calculate_trade_preview,
    add_transaction,
)
from scan_store import init_scan_db, get_latest_scan, get_scan_history
from data_provider import fetch_stock_quote
from worker import run_scan


# =====================================================
# CONFIG
# =====================================================

DB_PATH = os.getenv("DATABASE_PATH", "data/stock_scanner.sqlite3")
APP_VERSION = "2.4"

st.set_page_config(
    page_title=f"AI Stock Scanner {APP_VERSION}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db(DB_PATH)
init_scan_db(DB_PATH)

SIGNAL_NAMES = {
    "BUY": "MUA XÁC NHẬN",
    "WAIT": "CHỜ XÁC NHẬN",
    "WATCH": "THEO DÕI",
    "SKIP": "BỎ QUA",
    "FILTERED": "BỊ LOẠI",
}

STRATEGY_NAMES = {
    "SWING": "Swing 2–5 phiên",
    "REVERSAL": "Đảo chiều 5–10 phiên",
    "PULLBACK": "Pullback 3–7 phiên",
}

STRATEGY_ORDER = ["SWING", "REVERSAL", "PULLBACK"]


# =====================================================
# FORMAT
# =====================================================

def money(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.0f} đ"
    except (TypeError, ValueError):
        return "N/A"


def number(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def safe_float(value, default=None):
    try:
        result = float(value)
        if pd.isna(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def current_strategy_rows(rows):
    return [
        row for row in rows
        if row.get("strategy") in STRATEGY_ORDER
    ]


def price_status_name(status):
    return {
        "REALTIME": "Giá đã xác minh trong phiên",
        "LAST_SESSION": "Giá phiên gần nhất",
        "D1_ONLY": "Chỉ có giá D1",
        "UNVERIFIED": "Giá chưa xác minh",
        "D1_STALE": "Dữ liệu D1 cũ",
    }.get(status, "Không rõ trạng thái")


# =====================================================
# PRICE DATA
# =====================================================

def extract_price(raw):
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        value = safe_float(raw)
        return value if value and value > 0 else None

    if isinstance(raw, dict):
        for key in ("price", "close", "last_price", "lastPrice"):
            value = safe_float(raw.get(key))
            if value and value > 0:
                return value

    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_daily_quote(symbol):
    try:
        raw = fetch_stock_quote(symbol)
        return {
            "price": extract_price(raw),
            "quote_date": (
                raw.get("quote_date")
                if isinstance(raw, dict) else None
            ),
            "source": (
                raw.get("source")
                if isinstance(raw, dict) else "D1"
            ),
        }
    except Exception:
        return {
            "price": None,
            "quote_date": None,
            "source": None,
        }


# =====================================================
# PORTFOLIO DATA
# =====================================================

def load_positions_with_prices():
    positions = get_positions(DB_PATH)
    rows = []

    settings = get_settings(DB_PATH)
    sell_cost_rate = (
        float(settings["fee_sell"])
        + float(settings["sell_tax"])
    )

    for position in positions:
        symbol = position["symbol"]
        quantity = int(position["quantity"])
        avg_cost = float(position["avg_cost"])
        total_cost = float(position["total_cost"])

        quote = get_daily_quote(symbol)
        current_price = quote["price"]

        market_value = (
            quantity * current_price
            if current_price is not None else None
        )

        estimated_sell_proceeds = (
            market_value * (1 - sell_cost_rate)
            if market_value is not None else None
        )

        unrealized_pnl = (
            estimated_sell_proceeds - total_cost
            if estimated_sell_proceeds is not None else None
        )

        pnl_percent = (
            unrealized_pnl / total_cost * 100
            if unrealized_pnl is not None and total_cost > 0
            else None
        )

        rows.append({
            "symbol": symbol,
            "quantity": quantity,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "current_price": current_price,
            "quote_date": quote["quote_date"],
            "source": quote["source"],
            "market_value": market_value,
            "estimated_sell_proceeds": estimated_sell_proceeds,
            "unrealized_pnl": unrealized_pnl,
            "pnl_percent": pnl_percent,
        })

    return rows


def show_portfolio_summary():
    summary = get_portfolio_summary(DB_PATH)
    positions = load_positions_with_prices()

    total_market_value = sum(
        row["market_value"] or 0 for row in positions
    )
    total_unrealized = sum(
        row["unrealized_pnl"] or 0 for row in positions
    )

    st.subheader("Tổng quan tài sản")

    col1, col2 = st.columns(2)
    col1.metric(
        "Tổng giá vốn còn nắm giữ",
        money(summary["total_cost"]),
    )
    col2.metric(
        "Giá trị thị trường",
        money(total_market_value),
    )

    col3, col4 = st.columns(2)
    col3.metric(
        "Lãi/lỗ tạm tính sau phí bán",
        money(total_unrealized),
    )
    col4.metric(
        "Lãi/lỗ đã thực hiện",
        money(summary["realized_pnl"]),
    )

    st.metric(
        "Tổng số cổ phiếu đang nắm giữ",
        f"{summary['open_quantity']:,}",
    )

    return positions


def portfolio_display_dataframe(positions):
    rows = []

    for item in positions:
        rows.append({
            "Mã": item["symbol"],
            "Khối lượng": item["quantity"],
            "Giá vốn bình quân": item["avg_cost"],
            "Giá đóng cửa D1": item["current_price"],
            "Ngày dữ liệu": item["quote_date"],
            "Giá trị thị trường": item["market_value"],
            "Lãi/lỗ sau phí bán": item["unrealized_pnl"],
            "% lãi/lỗ": item["pnl_percent"],
        })

    return pd.DataFrame(rows)


# =====================================================
# PORTFOLIO ADVISOR
# =====================================================

def get_latest_signal_map():
    """
    Lấy kết quả của lượt quét gần nhất.
    Mỗi mã có thể có ba chiến lược.
    Ưu tiên tín hiệu tốt nhất theo thứ tự:
    BUY > WAIT > WATCH > SKIP > FILTERED.
    """
    rows = current_strategy_rows(get_latest_scan(DB_PATH))

    priority = {
        "BUY": 5,
        "WAIT": 4,
        "WATCH": 3,
        "SKIP": 2,
        "FILTERED": 1,
    }

    result = {}

    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue

        current = result.get(symbol)

        rank = (
            priority.get(row.get("signal"), 0),
            safe_float(row.get("score"), 0),
            safe_float(row.get("reward_risk"), 0),
        )

        if current is None:
            result[symbol] = row
            continue

        current_rank = (
            priority.get(current.get("signal"), 0),
            safe_float(current.get("score"), 0),
            safe_float(current.get("reward_risk"), 0),
        )

        if rank > current_rank:
            result[symbol] = row

    return result


def build_advice(position, signal):
    symbol = position["symbol"]
    quantity = position["quantity"]
    avg_cost = position["avg_cost"]
    price = position["current_price"]
    pnl_pct = position["pnl_percent"]

    if price is None:
        return {
            "symbol": symbol,
            "action": "CHƯA ĐỦ DỮ LIỆU",
            "color": "warning",
            "reason": "Không lấy được giá D1 để đánh giá.",
            "signal": None,
            "score": None,
            "stop_loss": None,
            "target": None,
            "pnl_pct": pnl_pct,
        }

    if signal is None:
        return {
            "symbol": symbol,
            "action": "GIỮ / THEO DÕI",
            "color": "info",
            "reason": (
                "Chưa có kết quả quét mới cho mã này. "
                "Không nên quyết định mua thêm chỉ dựa vào giá vốn."
            ),
            "signal": None,
            "score": None,
            "stop_loss": None,
            "target": None,
            "pnl_pct": pnl_pct,
        }

    signal_name = signal.get("signal")
    score = safe_float(signal.get("score"))
    stop = safe_float(signal.get("stop_loss"))
    target = safe_float(signal.get("target_price"))
    rr = safe_float(signal.get("reward_risk"))
    strategy = STRATEGY_NAMES.get(
        signal.get("strategy"),
        signal.get("strategy", "Không rõ"),
    )

    status = signal.get("price_data_status")
    reasons = signal.get("reasons") or []
    blockers = signal.get("blockers") or []
    penalties = signal.get("penalties") or []

    # D1 portfolio quote is not a verified intraday quote.
    # Never advise adding solely because the scan row says BUY
    # when the scan price itself is not verified.
    verified = status == "REALTIME"

    if stop is not None and price <= stop:
        action = "ƯU TIÊN BÁN / GIẢM TỶ TRỌNG"
        color = "error"
        reason = (
            f"Giá D1 hiện tại đã chạm hoặc thấp hơn vùng cắt lỗ "
            f"phân tích ({money(stop)}). Cần ưu tiên bảo toàn vốn."
        )

    elif target is not None and price >= target:
        action = "CÂN NHẮC CHỐT LỜI"
        color = "success"
        reason = (
            f"Giá D1 đã đạt hoặc vượt mục tiêu kỹ thuật "
            f"({money(target)}). Có thể cân nhắc chốt một phần "
            f"và đánh giá lại phần còn lại."
        )

    elif signal_name == "BUY" and verified:
        action = "TIẾP TỤC NẮM GIỮ"
        color = "success"
        reason = (
            "Tín hiệu quét đang là MUA xác nhận và dữ liệu giá "
            "trong phiên đã được xác minh. Vì mã đã nằm trong danh mục, "
            "không tự động khuyến nghị mua thêm."
        )

    elif signal_name == "BUY" and not verified:
        action = "GIỮ / CHỜ XÁC NHẬN"
        color = "warning"
        reason = (
            "Kết quả kỹ thuật có tín hiệu MUA nhưng dữ liệu giá "
            "không ở trạng thái realtime đã xác minh. "
            "Chưa đủ căn cứ để khuyến nghị mua thêm."
        )

    elif signal_name == "WAIT":
        action = "GIỮ / CHỜ"
        color = "warning"
        reason = (
            "Mã đang ở trạng thái chờ xác nhận. "
            "Không nên gia tăng vị thế khi điều kiện mua chưa hoàn tất."
        )

    elif signal_name == "WATCH":
        action = "THEO DÕI, KHÔNG MUA THÊM"
        color = "warning"
        reason = (
            "Điểm kỹ thuật mới ở mức theo dõi. "
            "Chưa có tín hiệu đủ mạnh để gia tăng vị thế."
        )

    else:
        action = "CÂN NHẮC GIẢM TỶ TRỌNG"
        color = "error"
        reason = (
            "Tín hiệu quét hiện tại không thuận lợi. "
            "Cần xem xét giảm rủi ro, nhất là khi giá tiến gần vùng cắt lỗ."
        )

    detail = []

    if score is not None:
        detail.append(f"Điểm kỹ thuật: {score:.0f}/100")

    detail.append(f"Chiến lược: {strategy}")
    detail.append(f"Tín hiệu: {SIGNAL_NAMES.get(signal_name, signal_name)}")

    if rr is not None:
        detail.append(f"RR sau phí theo kết quả quét: {rr:.2f}")

    if stop is not None:
        detail.append(f"Cắt lỗ tham chiếu: {money(stop)}")

    if target is not None:
        detail.append(f"Mục tiêu tham chiếu: {money(target)}")

    detail.append(
        "Giá quét đã xác minh realtime"
        if verified
        else "Giá quét chưa được xác minh realtime"
    )

    return {
        "symbol": symbol,
        "action": action,
        "color": color,
        "reason": reason,
        "signal": signal_name,
        "score": score,
        "stop_loss": stop,
        "target": target,
        "pnl_pct": pnl_pct,
        "detail": detail,
        "reasons": reasons,
        "blockers": blockers,
        "penalties": penalties,
        "quantity": quantity,
        "avg_cost": avg_cost,
        "current_price": price,
    }


def show_portfolio_advisor(positions):
    st.divider()
    st.subheader("🧠 CỐ VẤN DANH MỤC")

    st.caption(
        "Đánh giá từng mã đang nắm giữ dựa trên giá D1, giá vốn, "
        "lãi/lỗ sau phí bán và kết quả quét mới nhất. "
        "Đây là công cụ hỗ trợ quyết định, không phải lệnh giao dịch."
    )

    if not positions:
        st.info("Danh mục chưa có cổ phiếu để cố vấn.")
        return

    signals = get_latest_signal_map()

    if not signals:
        st.warning(
            "Chưa có kết quả quét của ba chiến lược. "
            "Hãy chạy quét thị trường để cập nhật đánh giá kỹ thuật."
        )

    advice_rows = []

    for position in positions:
        advice = build_advice(
            position,
            signals.get(position["symbol"]),
        )
        advice_rows.append(advice)

    for advice in advice_rows:
        st.markdown("---")

        st.markdown(f"### {advice['symbol']}")

        if advice["color"] == "error":
            st.error(f"**Khuyến nghị: {advice['action']}**")
        elif advice["color"] == "success":
            st.success(f"**Khuyến nghị: {advice['action']}**")
        elif advice["color"] == "warning":
            st.warning(f"**Khuyến nghị: {advice['action']}**")
        else:
            st.info(f"**Khuyến nghị: {advice['action']}**")

        st.write(advice["reason"])

        if advice.get("quantity") is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Số lượng", f"{advice['quantity']:,}")
            c2.metric("Giá vốn", money(advice["avg_cost"]))
            c3.metric("Giá D1", money(advice["current_price"]))

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Lãi/lỗ sau phí bán",
            (
                f"{advice['pnl_pct']:+.2f}%"
                if advice.get("pnl_pct") is not None
                else "N/A"
            ),
        )
        c2.metric(
            "Cắt lỗ tham chiếu",
            money(advice.get("stop_loss")),
        )
        c3.metric(
            "Mục tiêu tham chiếu",
            money(advice.get("target")),
        )

        if advice.get("detail"):
            with st.expander("Chi tiết phân tích"):
                for item in advice["detail"]:
                    st.write(f"• {item}")

                if advice.get("reasons"):
                    st.markdown("**Điều kiện đã đạt**")
                    for item in advice["reasons"]:
                        st.write(f"✓ {item}")

                if advice.get("blockers"):
                    st.markdown("**Điều kiện chưa đạt**")
                    for item in advice["blockers"]:
                        st.write(f"• {item}")

                if advice.get("penalties"):
                    st.markdown("**Điểm cần thận trọng**")
                    for item in advice["penalties"]:
                        st.write(f"• {item}")

    st.caption(
        "Lưu ý: Giá D1 dùng cho danh mục là giá đóng cửa gần nhất, "
        "không phải giá realtime. Cắt lỗ và mục tiêu là mức tham chiếu "
        "từ lần quét gần nhất, không phải lệnh bán tự động."
    )


# =====================================================
# SCAN DISPLAY
# =====================================================

def display_scan_table(rows):
    if not rows:
        st.info("Không có kết quả.")
        return

    df = pd.DataFrame(rows)

    if "signal" in df.columns:
        df["Tín hiệu"] = df["signal"].map(SIGNAL_NAMES).fillna(df["signal"])

    if "strategy" in df.columns:
        df["Chiến lược"] = (
            df["strategy"].map(STRATEGY_NAMES).fillna(df["strategy"])
        )

    if "price_data_status" in df.columns:
        df["Trạng thái giá"] = df["price_data_status"].map(
            price_status_name
        )

    rename = {
        "symbol": "Mã",
        "score": "Điểm kỹ thuật",
        "price": "Giá dùng phân tích",
        "d1_close": "Giá đóng cửa D1",
        "realtime_price": "Giá realtime đã xác minh",
        "kbs_quote_price": "Giá KBS nhận được",
        "stop_loss": "Cắt lỗ",
        "target_price": "Mục tiêu",
        "reward_risk": "RR sau phí",
        "reason": "Lý do",
        "data_date": "Ngày dữ liệu D1",
        "avg_turnover_20d": "GTGD TB 20 phiên",
        "median_turnover_20d": "GTGD trung vị 20 phiên",
    }

    df = df.rename(columns=rename)

    preferred = [
        "Mã",
        "Chiến lược",
        "Tín hiệu",
        "Điểm kỹ thuật",
        "Trạng thái giá",
        "Giá realtime đã xác minh",
        "Giá KBS nhận được",
        "Giá đóng cửa D1",
        "Giá dùng phân tích",
        "Cắt lỗ",
        "Mục tiêu",
        "RR sau phí",
        "Ngày dữ liệu D1",
        "GTGD TB 20 phiên",
        "Lý do",
    ]

    columns = [col for col in preferred if col in df.columns]

    if columns:
        df = df[columns]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def show_strategy_metrics(df):
    if df.empty:
        st.info("Chưa có kết quả.")
        return

    st.subheader("Thống kê theo chiến lược")
    cols = st.columns(3)

    for index, strategy in enumerate(STRATEGY_ORDER):
        strategy_df = df[df["strategy"] == strategy]

        buy_count = int((strategy_df["signal"] == "BUY").sum())
        wait_count = int((strategy_df["signal"] == "WAIT").sum())
        watch_count = int((strategy_df["signal"] == "WATCH").sum())

        cols[index].metric(
            STRATEGY_NAMES[strategy],
            f"{buy_count} mã MUA",
        )
        cols[index].caption(
            f"Chờ: {wait_count} | Theo dõi: {watch_count}"
        )


def show_signal_explanation(row):
    st.subheader(f"Phân tích chi tiết: {row.get('symbol', '')}")

    st.write(
        f"**Chiến lược:** "
        f"{STRATEGY_NAMES.get(row.get('strategy'), row.get('strategy'))}"
    )
    st.write(
        f"**Tín hiệu:** "
        f"{SIGNAL_NAMES.get(row.get('signal'), row.get('signal'))}"
    )
    st.write(
        f"**Điểm kỹ thuật:** {number(row.get('score'), 0)}/100"
    )

    status = row.get("price_data_status")
    message = (
        row.get("price_data_message")
        or price_status_name(status)
    )

    if status == "REALTIME":
        st.success(message)
    elif status in ("LAST_SESSION", "D1_ONLY", "UNVERIFIED"):
        st.warning(message)
    elif status == "D1_STALE":
        st.error(message)
    else:
        st.info(message)

    c1, c2 = st.columns(2)
    c1.metric("Giá realtime đã xác minh", money(row.get("realtime_price")))
    c2.metric("Giá KBS nhận được", money(row.get("kbs_quote_price")))

    c1, c2 = st.columns(2)
    c1.metric("Giá đóng cửa D1", money(row.get("d1_close")))
    c2.metric("Giá dùng lập kế hoạch", money(row.get("price")))

    c1, c2, c3 = st.columns(3)
    c1.metric("Cắt lỗ", money(row.get("stop_loss")))
    c2.metric("Mục tiêu", money(row.get("target_price")))
    c3.metric("RR sau phí", number(row.get("reward_risk")))

    for key, title in [
        ("reasons", "Điều kiện đã đạt"),
        ("blockers", "Điều kiện chưa đạt"),
        ("penalties", "Điểm cần thận trọng"),
    ]:
        items = row.get(key) or []
        if items:
            st.markdown(f"**{title}**")
            for item in items:
                st.write(f"• {item}")

    st.caption(
        "Giá KBS nhận được không đồng nghĩa với giá realtime đã xác minh. "
        "Điểm kỹ thuật không phải xác suất thắng và tín hiệu không bảo đảm lợi nhuận."
    )


# =====================================================
# DASHBOARD
# =====================================================

def show_dashboard():
    st.title(f"📈 AI STOCK SCANNER {APP_VERSION}")

    st.caption(
        "Phân tích HOSE/HNX bằng chỉ báo D1, "
        "kiểm tra giá trong phiên và quản lý danh mục."
    )

    positions = show_portfolio_summary()

    st.divider()
    st.subheader("Danh mục hiện tại")

    if positions:
        st.dataframe(
            portfolio_display_dataframe(positions),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Giá danh mục sử dụng giá đóng cửa D1.")
    else:
        st.info("Bạn chưa có cổ phiếu trong danh mục.")

    st.divider()

    latest = current_strategy_rows(get_latest_scan(DB_PATH))

    st.subheader("Kết quả quét gần nhất")

    if latest:
        df = pd.DataFrame(latest)
        show_strategy_metrics(df)
        display_scan_table(latest[:30])
    else:
        st.info("Chưa có kết quả quét của ba chiến lược mới.")

    show_portfolio_advisor(positions)


# =====================================================
# SCANNER PAGE
# =====================================================

def show_scanner():
    st.title("🔎 QUÉT CỔ PHIẾU")

    st.info(
        "Ba chiến lược độc lập: Swing 2–5 phiên, "
        "Đảo chiều 5–10 phiên và Pullback 3–7 phiên."
    )

    st.caption(
        "Yêu cầu tối thiểu 200 phiên lịch sử và thanh khoản đạt ngưỡng. "
        "Chỉ báo kỹ thuật tính trên D1."
    )

    st.warning(
        "Nếu giá trong phiên không xác minh được, bot không xác nhận BUY."
    )

    if st.button(
        "🚀 QUÉT THỊ TRƯỜNG NGAY",
        use_container_width=True,
        type="primary",
    ):
        progress_bar = st.progress(
            0,
            text="Chuẩn bị quét thị trường...",
        )
        progress_text = st.empty()

        try:
            def update_progress(percent, message):
                percent = max(0, min(100, int(percent)))
                progress_bar.progress(percent, text=f"{percent}%")
                progress_text.caption(message)

            saved = run_scan(progress_callback=update_progress)
            get_daily_quote.clear()

            if saved > 0:
                st.success(
                    f"Hoàn thành lượt quét. Đã lưu {saved:,} dòng kết quả."
                )
            else:
                st.warning(
                    "Không lưu được kết quả. Hãy kiểm tra Cloud logs."
                )

        except Exception as exc:
            st.error(f"Quét thất bại: {exc}")

        finally:
            progress_bar.empty()

    st.divider()

    latest = current_strategy_rows(get_latest_scan(DB_PATH))

    if not latest:
        st.warning("Chưa có kết quả. Hãy chạy quét thị trường trước.")
        return

    df = pd.DataFrame(latest)

    st.subheader("Tình trạng dữ liệu giá")

    if "price_data_status" in df.columns:
        status_counts = df["price_data_status"].value_counts()

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Realtime xác minh",
            int(status_counts.get("REALTIME", 0)),
        )
        c2.metric(
            "Giá phiên gần nhất",
            int(status_counts.get("LAST_SESSION", 0)),
        )
        c3.metric(
            "Chỉ có D1",
            int(status_counts.get("D1_ONLY", 0)),
        )

        c1, c2 = st.columns(2)
        c1.metric(
            "Chưa xác minh",
            int(status_counts.get("UNVERIFIED", 0)),
        )
        c2.metric(
            "D1 cũ",
            int(status_counts.get("D1_STALE", 0)),
        )

    show_strategy_metrics(df)

    st.divider()
    st.subheader("Thống kê tín hiệu")

    signal_counts = df["signal"].value_counts()
    cols = st.columns(5)

    signal_order = [
        "BUY", "WAIT", "WATCH", "SKIP", "FILTERED"
    ]

    for index, signal in enumerate(signal_order):
        cols[index].metric(
            SIGNAL_NAMES[signal],
            int(signal_counts.get(signal, 0)),
        )

    st.divider()
    st.subheader("⭐ Danh sách ưu tiên theo dõi")

    candidates = df[
        df["signal"].isin(["BUY", "WAIT"])
    ].copy()

    if not candidates.empty:
        candidates["_signal_rank"] = candidates["signal"].map({
            "BUY": 0,
            "WAIT": 1,
        })

        candidates["_rr"] = pd.to_numeric(
            candidates["reward_risk"],
            errors="coerce",
        )

        candidates = candidates.sort_values(
            ["_signal_rank", "score", "_rr"],
            ascending=[True, False, False],
            na_position="last",
        )

        display_scan_table(
            candidates.head(15)
            .drop(columns=["_signal_rank", "_rr"])
            .to_dict("records")
        )

        options = [
            (
                f"{row['symbol']} - "
                f"{STRATEGY_NAMES.get(row['strategy'], row['strategy'])} - "
                f"{SIGNAL_NAMES.get(row['signal'], row['signal'])} - "
                f"{row.get('score', 0)} điểm"
            )
            for _, row in candidates.head(30).iterrows()
        ]

        selected = st.selectbox(
            "Chọn mã cần phân tích chi tiết",
            options,
            key="candidate_detail",
        )

        selected_index = options.index(selected)
        selected_row = candidates.head(30).iloc[selected_index].to_dict()

        show_signal_explanation(selected_row)

    else:
        st.info(
            "Hiện chưa có mã MUA hoặc CHỜ. "
            "Không có tín hiệu cũng là một kết quả hợp lệ."
        )

    st.divider()
    st.subheader("Lọc kết quả")

    strategy_options = {
        "Tất cả chiến lược": "ALL",
        "Swing 2–5 phiên": "SWING",
        "Đảo chiều 5–10 phiên": "REVERSAL",
        "Pullback 3–7 phiên": "PULLBACK",
    }

    strategy_label = st.selectbox(
        "Chọn chiến lược",
        list(strategy_options.keys()),
    )

    signal_options = {
        "Tất cả tín hiệu": "ALL",
        "MUA xác nhận": "BUY",
        "Chờ xác nhận": "WAIT",
        "Theo dõi": "WATCH",
        "Bỏ qua": "SKIP",
        "Bị loại": "FILTERED",
    }

    signal_label = st.selectbox(
        "Chọn tín hiệu",
        list(signal_options.keys()),
    )

    filtered = df.copy()
    selected_strategy = strategy_options[strategy_label]
    selected_signal = signal_options[signal_label]

    if selected_strategy != "ALL":
        filtered = filtered[
            filtered["strategy"] == selected_strategy
        ]

    if selected_signal != "ALL":
        filtered = filtered[
            filtered["signal"] == selected_signal
        ]

    min_score = st.slider(
        "Điểm kỹ thuật tối thiểu",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

    filtered = filtered[
        pd.to_numeric(
            filtered["score"],
            errors="coerce",
        ).fillna(0) >= min_score
    ]

    if filtered.empty:
        st.info("Không có kết quả phù hợp bộ lọc.")
    else:
        st.subheader(
            f"Có {len(filtered)} kết quả chiến lược"
        )
        display_scan_table(filtered.to_dict("records"))

    st.divider()
    st.subheader("Lịch sử quét")

    history = get_scan_history(
        limit=150,
        db_path=DB_PATH,
    )

    history = current_strategy_rows(history)

    if history:
        display_scan_table(history)
    else:
        st.info("Chưa có lịch sử của ba chiến lược mới.")


# =====================================================
# PORTFOLIO PAGE
# =====================================================

def show_portfolio():
    st.title("💼 QUẢN LÝ DANH MỤC")

    st.warning(
        "Bot chỉ ghi nhận giao dịch do bạn nhập. "
        "Không tự động đặt lệnh mua hoặc bán."
    )

    positions = show_portfolio_summary()

    st.divider()
    st.subheader("🧠 Cố vấn danh mục")
    show_portfolio_advisor(positions)

    st.divider()
    st.subheader("Ghi nhận giao dịch")

    with st.form("transaction_form"):
        side = st.radio(
            "Loại giao dịch",
            ["MUA", "BÁN"],
            horizontal=True,
        )

        symbol = st.text_input(
            "Mã cổ phiếu",
            placeholder="Ví dụ: HDB",
        ).upper().strip()

        quantity = st.number_input(
            "Khối lượng",
            min_value=1,
            value=100,
            step=100,
        )

        price = st.number_input(
            "Giá khớp (đồng/cổ phiếu)",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )

        note = st.text_input("Ghi chú")

        submitted = st.form_submit_button(
            "XEM CHI PHÍ GIAO DỊCH",
            use_container_width=True,
        )

    if submitted:
        try:
            preview = calculate_trade_preview(
                side,
                symbol,
                int(quantity),
                price,
                db_path=DB_PATH,
            )

            st.session_state["pending_trade"] = {
                "side": side,
                "symbol": symbol,
                "quantity": int(quantity),
                "price": float(price),
                "note": note,
                "preview": preview,
            }

        except Exception as exc:
            st.error(str(exc))

    pending = st.session_state.get("pending_trade")

    if pending:
        st.divider()
        st.subheader("Xác nhận giao dịch")

        preview = pending["preview"]

        st.write(f"Loại: **{pending['side']}**")
        st.write(f"Mã: **{pending['symbol']}**")
        st.write(f"Khối lượng: **{pending['quantity']:,}**")
        st.write(f"Giá khớp: **{money(pending['price'])}**")
        st.write(f"Giá trị giao dịch: **{money(preview['gross'])}**")
        st.write(f"Phí: **{money(preview['fee'])}**")
        st.write(f"Thuế: **{money(preview['tax'])}**")

        st.metric(
            "Tiền cần chi / tiền thực nhận",
            money(preview["cash_flow"]),
        )

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            if st.button(
                "XÁC NHẬN LƯU",
                type="primary",
                use_container_width=True,
            ):
                try:
                    add_transaction(
                        pending["side"],
                        pending["symbol"],
                        pending["quantity"],
                        pending["price"],
                        pending["note"],
                        DB_PATH,
                    )

                    del st.session_state["pending_trade"]
                    st.success("Đã lưu giao dịch.")
                    st.rerun()

                except Exception as exc:
                    st.error(str(exc))

        with cancel_col:
            if st.button(
                "HỦY",
                use_container_width=True,
            ):
                del st.session_state["pending_trade"]
                st.rerun()

    st.divider()
    st.subheader("Danh mục đang nắm giữ")

    positions = load_positions_with_prices()

    if positions:
        st.dataframe(
            portfolio_display_dataframe(positions),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Giá D1 là giá đóng cửa gần nhất. "
            "Lãi/lỗ ước tính đã trừ phí bán và thuế bán."
        )
    else:
        st.info("Chưa có cổ phiếu.")

    st.divider()
    st.subheader("Lịch sử giao dịch")

    transactions = get_transactions(DB_PATH)

    if transactions:
        st.dataframe(
            pd.DataFrame(transactions),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Chưa có giao dịch nào được ghi nhận.")


# =====================================================
# SETTINGS
# =====================================================

def show_settings():
    st.title("⚙️ CÀI ĐẶT HỆ THỐNG")

    settings = get_settings(DB_PATH)

    st.subheader("Quản lý rủi ro")

    with st.form("settings_form"):
        risk_pct = st.number_input(
            "Rủi ro tối đa mỗi giao dịch (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(settings["risk_pct"]),
            step=0.1,
        )

        st.subheader("Phí giao dịch")

        fee_buy = st.number_input(
            "Phí mua (%)",
            min_value=0.0,
            max_value=2.0,
            value=float(settings["fee_buy"]) * 100,
            step=0.01,
        )

        fee_sell = st.number_input(
            "Phí bán (%)",
            min_value=0.0,
            max_value=2.0,
            value=float(settings["fee_sell"]) * 100,
            step=0.01,
        )

        sell_tax = st.number_input(
            "Thuế bán (%)",
            min_value=0.0,
            max_value=2.0,
            value=float(settings["sell_tax"]) * 100,
            step=0.01,
        )

        saved = st.form_submit_button(
            "LƯU CÀI ĐẶT",
            use_container_width=True,
        )

    if saved:
        try:
            save_settings({
                "risk_pct": risk_pct,
                "fee_buy": fee_buy / 100,
                "fee_sell": fee_sell / 100,
                "sell_tax": sell_tax / 100,
            }, DB_PATH)

            st.success("Đã lưu cài đặt.")
            st.rerun()

        except Exception as exc:
            st.error(str(exc))

    st.caption(
        "Đối chiếu phí mua, phí bán và thuế với xác nhận "
        "giao dịch thực tế của công ty chứng khoán."
    )

    st.divider()
    st.subheader("Thông tin hệ thống")

    st.write(f"Phiên bản: AI Stock Scanner {APP_VERSION}")
    st.write("Thị trường: HOSE / HNX")
    st.write("Khung chỉ báo kỹ thuật: D1")
    st.write("Giá xác nhận: Giá trong phiên đã xác minh")
    st.write("Chiến lược: Swing / Reversal / Pullback")
    st.write("Tự động đặt lệnh: Không")


# =====================================================
# MAIN
# =====================================================

def main():
    with st.sidebar:
        st.title("AI STOCK SCANNER")

        page = st.radio(
            "Điều hướng",
            [
                "Tổng quan",
                "Quét cổ phiếu",
                "Danh mục đầu tư",
                "Cài đặt",
            ],
        )

        st.divider()
        st.caption(f"AI Stock Scanner {APP_VERSION}")
        st.caption("Chỉ báo kỹ thuật D1")

    if page == "Tổng quan":
        show_dashboard()

    elif page == "Quét cổ phiếu":
        show_scanner()

    elif page == "Danh mục đầu tư":
        show_portfolio()

    elif page == "Cài đặt":
        show_settings()


if __name__ == "__main__":
    main()
