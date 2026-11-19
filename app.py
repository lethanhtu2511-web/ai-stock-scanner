
import os
import threading
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from scan_store import (
    init_scan_db,
    get_latest_scan,
    get_scan_history,
    get_scan_summary,
)
from worker import run_scan
from portfolio import (
    add_transaction,
    get_positions,
    get_transactions,
    get_portfolio_summary,
    get_cash_summary,
    get_settings,
    save_settings,
)


# =========================================================
# CONFIGURATION
# =========================================================

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DB_PATH = os.getenv("DATABASE_PATH", "data/stock_scanner.sqlite3")
APP_VERSION = "3.0"

AUTO_REFRESH_SECONDS = 300

AUTO_SCAN_ENABLED = (
    os.getenv("AUTO_SCAN_ENABLED", "0").strip().lower()
    not in {"0", "false", "no", "off"}
)

st.set_page_config(
    page_title=f"AI Stock Scanner {APP_VERSION}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_scan_db(DB_PATH)

SIGNAL_NAMES = {
    "BUY": "MUA XÁC NHẬN",
    "WAIT": "CHỜ XÁC NHẬN",
    "WATCH": "THEO DÕI",
    "SKIP": "BỎ QUA",
    "FILTERED": "BỊ LỌC",
}

STRATEGY_NAMES = {
    "SWING": "Swing 2–5 phiên",
    "REVERSAL": "Reversal 5–10 phiên",
    "PULLBACK": "Pullback 3–7 phiên",
}

STRATEGY_ORDER = [
    "SWING",
    "REVERSAL",
    "PULLBACK",
]


# =========================================================
# UTILITIES
# =========================================================

def money(value):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.0f} đ"
    except (TypeError, ValueError):
        return "—"


def number(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def is_market_session(now=None):
    now = now or datetime.now(VN_TZ)

    if now.weekday() >= 5:
        return False

    current = now.time()

    return (
        dt_time(9, 0) <= current <= dt_time(11, 30)
        or dt_time(13, 0) <= current <= dt_time(15, 0)
    )


def parse_scan_time(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=VN_TZ)

        return parsed.astimezone(VN_TZ)

    except (TypeError, ValueError):
        return None


def auto_scan_due(now, latest_scan_at):
    if now.weekday() >= 5:
        return False

    latest = parse_scan_time(latest_scan_at)

    if is_market_session(now):
        if latest is None:
            return True

        age = (now - latest).total_seconds()

        return age >= AUTO_REFRESH_SECONDS - 10

    # Quét D1 sau khi thị trường đóng cửa.
    if now.time() >= dt_time(15, 35):
        if latest is None:
            return True

        return not (
            latest.date() == now.date()
            and latest.time() >= dt_time(15, 35)
        )

    return False


def current_strategy_rows(rows):
    return [
        row for row in rows
        if row.get("strategy") in STRATEGY_ORDER
    ]


# =========================================================
# TABLE
# =========================================================

def display_scan_table(rows):
    if not rows:
        st.info("Chưa có kết quả phù hợp.")
        return

    df = pd.DataFrame(rows)

    if "signal" in df.columns:
        df["Tín hiệu"] = (
            df["signal"]
            .map(SIGNAL_NAMES)
            .fillna(df["signal"])
        )

    if "strategy" in df.columns:
        df["Chiến lược"] = (
            df["strategy"]
            .map(STRATEGY_NAMES)
            .fillna(df["strategy"])
        )

    rename = {
        "symbol": "Mã",
        "score": "Điểm",
        "price": "Giá phân tích",
        "realtime_price": "Giá realtime",
        "d1_close": "Đóng cửa D1",
        "live_change_pct": "Biến động phiên (%)",
        "stop_loss": "Cắt lỗ",
        "target_price": "Mục tiêu",
        "reward_risk": "RR sau phí",
        "relative_strength": "Sức mạnh tương đối (%)",
        "market_context": "Thị trường",
        "price_data_status": "Trạng thái giá",
        "reason": "Lý do",
        "data_date": "Ngày D1",
    }

    df = df.rename(columns=rename)

    preferred = [
        "Mã",
        "Chiến lược",
        "Tín hiệu",
        "Điểm",
        "Giá realtime",
        "Đóng cửa D1",
        "Biến động phiên (%)",
        "Cắt lỗ",
        "Mục tiêu",
        "RR sau phí",
        "Sức mạnh tương đối (%)",
        "Thị trường",
        "Ngày D1",
        "Lý do",
    ]

    columns = [
        col for col in preferred
        if col in df.columns
    ]

    if columns:
        df = df[columns]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# STRATEGY METRICS
# =========================================================

def show_strategy_metrics(df):
    if df.empty:
        return

    cols = st.columns(3)

    for index, strategy in enumerate(STRATEGY_ORDER):
        subset = df[df["strategy"] == strategy]

        buys = int((subset["signal"] == "BUY").sum())
        waits = int((subset["signal"] == "WAIT").sum())
        watches = int((subset["signal"] == "WATCH").sum())

        cols[index].metric(
            STRATEGY_NAMES[strategy],
            f"{buys} MUA",
        )

        cols[index].caption(
            f"Chờ: {waits} | Theo dõi: {watches}"
        )


# =========================================================
# SIGNAL DETAIL
# =========================================================

def show_signal_detail(row):
    st.subheader(f"Phân tích {row.get('symbol', '')}")

    st.write(
        f"**Chiến lược:** "
        f"{STRATEGY_NAMES.get(row.get('strategy'), row.get('strategy'))}"
    )

    st.write(
        f"**Tín hiệu:** "
        f"{SIGNAL_NAMES.get(row.get('signal'), row.get('signal'))}"
    )

    st.write(
        f"**Điểm kỹ thuật:** "
        f"{number(row.get('score'), 0)}/100"
    )

    status = row.get("price_data_status")
    status_text = (
        row.get("price_data_message")
        or status
        or "Không rõ"
    )

    if status == "REALTIME":
        st.success(status_text)

    elif status in (
        "LAST_SESSION",
        "D1_ONLY",
        "UNVERIFIED",
    ):
        st.warning(status_text)

    else:
        st.info(status_text)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Giá realtime",
        money(row.get("realtime_price")),
    )

    c2.metric(
        "Đóng cửa D1",
        money(row.get("d1_close")),
    )

    c3.metric(
        "Biến động phiên",
        f"{number(row.get('live_change_pct'))}%",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Cắt lỗ",
        money(row.get("stop_loss")),
    )

    c2.metric(
        "Mục tiêu",
        money(row.get("target_price")),
    )

    c3.metric(
        "RR sau phí",
        number(row.get("reward_risk")),
    )

    for key, heading in [
        ("reasons", "Điều kiện đạt"),
        ("blockers", "Điều kiện chưa đạt"),
        ("penalties", "Điểm cần thận trọng"),
    ]:
        items = row.get(key) or []

        if items:
            st.markdown(f"**{heading}**")

            for item in items:
                st.write(f"- {item}")

    st.caption(
        "Điểm kỹ thuật không phải xác suất thắng. "
        "Giá realtime chỉ được xác nhận khi timestamp từ nguồn còn mới. "
        "Tín hiệu không bảo đảm lợi nhuận."
    )


# =========================================================
# SCAN LOCK
# =========================================================

@st.cache_resource
def get_scan_lock():
    return threading.Lock()


# =========================================================
# AUTO REFRESH AND SCAN
# =========================================================

@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def auto_refresh_panel(show_details=True):

    now = datetime.now(VN_TZ)

    summary = get_scan_summary(DB_PATH)

    latest_time = summary.get("latest_scan_at")

    lock = get_scan_lock()

    if (
        AUTO_SCAN_ENABLED
        and auto_scan_due(now, latest_time)
    ):

        acquired = lock.acquire(blocking=False)

        if acquired:
            try:
                summary = get_scan_summary(DB_PATH)

                latest_time = summary.get("latest_scan_at")

                now = datetime.now(VN_TZ)

                if auto_scan_due(now, latest_time):

                    with st.spinner(
                        "Đang cập nhật giá và quét HOSE/HNX..."
                    ):
                        saved = run_scan()

                    if saved:
                        st.success(
                            f"Đã cập nhật {saved:,} dòng kết quả."
                        )
                    else:
                        st.warning(
                            "Lượt quét chưa lưu được kết quả. "
                            "Kiểm tra nhật ký Streamlit Cloud."
                        )

            except Exception as exc:
                st.error(
                    f"Không thể tự động quét: {exc}"
                )

            finally:
                lock.release()

        else:
            st.info(
                "Một phiên khác đang thực hiện lượt quét."
            )

    summary = get_scan_summary(DB_PATH)

    latest_time = summary.get("latest_scan_at")

    latest_dt = parse_scan_time(latest_time)

    if latest_dt:
        age_seconds = max(
            0,
            (
                datetime.now(VN_TZ) - latest_dt
            ).total_seconds(),
        )

        st.caption(
            f"Lượt quét gần nhất: "
            f"{latest_dt.strftime('%d/%m/%Y %H:%M:%S')} "
            f"· Cách đây {int(age_seconds)} giây "
            f"· Tự làm mới mỗi 5 phút khi phiên còn mở."
        )

    else:
        st.caption("Chưa có lượt quét nào được lưu.")

    if is_market_session() and AUTO_SCAN_ENABLED:
        st.info(
            "Đang trong giờ giao dịch. Bot tự quét lại "
            "sau mỗi 300 giây khi trang đang được mở."
        )

    elif is_market_session() and not AUTO_SCAN_ENABLED:
        st.info(
            "Tự động quét do worker nền đảm nhiệm; "
            "giao diện làm mới kết quả mỗi 300 giây."
        )

    else:
        st.warning(
            "Ngoài giờ giao dịch: không thể xác nhận giá realtime. "
            "Sau 15:35 bot chỉ thực hiện lượt quét D1 cuối ngày một lần."
        )

    latest = current_strategy_rows(
        get_latest_scan(DB_PATH)
    )

    if not latest:
        st.warning(
            "Chưa có kết quả. Hãy nhấn QUÉT NGAY để chạy lượt đầu tiên."
        )
        return

    df = pd.DataFrame(latest)

    st.subheader("Trạng thái dữ liệu")

    status_source = (
        df.drop_duplicates("symbol")
        if "symbol" in df.columns
        else df
    )

    status_counts = status_source.get(
        "price_data_status",
        pd.Series(dtype=str),
    ).value_counts()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Realtime xác minh",
        int(status_counts.get("REALTIME", 0)),
    )

    c2.metric(
        "Giá phiên gần nhất",
        int(status_counts.get("LAST_SESSION", 0)),
    )

    c3.metric(
        "Chưa xác minh",
        int(status_counts.get("UNVERIFIED", 0)),
    )

    c4.metric(
        "D1 cũ",
        int(status_counts.get("D1_STALE", 0)),
    )

    verified_rows = (
        status_source[
            status_source.get("price_data_status") == "REALTIME"
        ]
        if "price_data_status" in status_source.columns
        else pd.DataFrame()
    )

    if (
        not verified_rows.empty
        and "quote_source_time" in verified_rows.columns
    ):

        source_times = pd.to_datetime(
            verified_rows["quote_source_time"],
            errors="coerce",
        ).dropna()

        if not source_times.empty:
            st.caption(
                "Timestamp mới nhất do nguồn báo giá cung cấp: "
                f"{source_times.max().strftime('%d/%m/%Y %H:%M:%S')}"
            )

    if "quote_age_seconds" in df.columns:

        verified = df[
            df.get("price_data_status") == "REALTIME"
        ]

        ages = pd.to_numeric(
            verified.get("quote_age_seconds"),
            errors="coerce",
        ).dropna()

        if not ages.empty:
            st.caption(
                "Độ trễ lớn nhất của báo giá đã xác minh: "
                f"{int(ages.max())} giây."
            )

    show_strategy_metrics(df)

    signal_counts = df["signal"].value_counts()

    cols = st.columns(5)

    for idx, signal in enumerate(
        ["BUY", "WAIT", "WATCH", "SKIP", "FILTERED"]
    ):

        cols[idx].metric(
            SIGNAL_NAMES[signal],
            int(signal_counts.get(signal, 0)),
        )

    st.divider()

    st.subheader("Cơ hội ưu tiên")

    candidates = df[
        df["signal"].isin(["BUY", "WAIT"])
    ].copy()

    if not candidates.empty:

        candidates["_rank"] = candidates["signal"].map(
            {"BUY": 0, "WAIT": 1}
        )

        candidates["_rr"] = pd.to_numeric(
            candidates.get("reward_risk"),
            errors="coerce",
        )

        candidates = candidates.sort_values(
            ["_rank", "score", "_rr"],
            ascending=[True, False, False],
            na_position="last",
        )

        display_scan_table(
            candidates.head(20).to_dict("records")
        )

        if show_details:

            top_candidates = candidates.head(30)

            options = [
                (
                    f"{row['symbol']} · "
                    f"{STRATEGY_NAMES.get(row['strategy'], row['strategy'])} · "
                    f"{SIGNAL_NAMES.get(row['signal'], row['signal'])} · "
                    f"{row.get('score', 0)} điểm"
                )
                for _, row in top_candidates.iterrows()
            ]

            selected = st.selectbox(
                "Chọn mã xem phân tích",
                options,
                key="auto_scan_candidate_detail",
            )

            selected_index = options.index(selected)

            show_signal_detail(
                top_candidates.iloc[selected_index].to_dict()
            )

    else:
        st.info(
            "Hiện chưa có mã đủ điều kiện MUA hoặc CHỜ. "
            "Bot không cố tạo tín hiệu khi thị trường chưa phù hợp."
        )

    if show_details:
        with st.expander("Toàn bộ kết quả quét"):
            display_scan_table(latest)


# =========================================================
# PAGES
# =========================================================

def show_dashboard():

    st.title("📈 AI STOCK SCANNER")

    st.caption(
        "HOSE/HNX · Swing 2–5 phiên · "
        "Reversal 5–10 phiên · Pullback 3–7 phiên"
    )

    auto_refresh_panel(show_details=False)


def show_scanner():

    st.title("🔎 QUÉT CỔ PHIẾU")

    st.info(
        "Bot cập nhật mỗi 300 giây trong giờ giao dịch "
        "khi trang đang mở. Phân tích kỹ thuật sử dụng D1; "
        "giá trong phiên dùng để xác nhận điểm mua "
        "và lập kế hoạch rủi ro."
    )

    if st.button(
        "🚀 QUÉT THỊ TRƯỜNG NGAY",
        type="primary",
        use_container_width=True,
    ):

        progress = st.progress(
            0,
            text="Đang chuẩn bị...",
        )

        message = st.empty()

        def update_progress(percent, text):

            progress.progress(
                max(0, min(100, int(percent))),
                text=f"{int(percent)}%",
            )

            message.caption(text)

        try:
            saved = run_scan(
                progress_callback=update_progress
            )

            if saved:
                st.success(
                    f"Đã quét và lưu {saved:,} dòng."
                )

            else:
                st.warning(
                    "Không lưu được kết quả. "
                    "Kiểm tra Cloud logs."
                )

        except Exception as exc:
            st.error(
                f"Quét thất bại: {exc}"
            )

        finally:
            progress.empty()
            message.empty()

    auto_refresh_panel(show_details=True)


def show_history():

    st.title("🕘 LỊCH SỬ QUÉT")

    history = current_strategy_rows(
        get_scan_history(
            limit=500,
            db_path=DB_PATH,
        )
    )

    if not history:
        st.info("Chưa có lịch sử quét.")
        return

    df = pd.DataFrame(history)

    if "scanned_at" in df.columns:
        st.caption(
            "Lịch sử lưu trên SQLite của môi trường chạy ứng dụng. "
            "Streamlit Cloud có thể làm mất dữ liệu cục bộ "
            "khi khởi động lại."
        )

    signal_filter = st.selectbox(
        "Lọc tín hiệu",
        [
            "Tất cả",
            "MUA",
            "CHỜ",
            "THEO DÕI",
            "BỎ QUA",
            "BỊ LỌC",
        ],
    )

    mapping = {
        "MUA": "BUY",
        "CHỜ": "WAIT",
        "THEO DÕI": "WATCH",
        "BỎ QUA": "SKIP",
        "BỊ LỌC": "FILTERED",
    }

    if signal_filter != "Tất cả":
        df = df[
            df["signal"] == mapping[signal_filter]
        ]

    display_scan_table(
        df.to_dict("records")
    )



# =========================================================
# PORTFOLIO
# =========================================================

def show_portfolio():
    st.title("💼 QUẢN LÝ DANH MỤC")

    st.caption(
        "Theo dõi vị thế, giá vốn bình quân đã gồm phí mua, "
        "lãi/lỗ tạm tính sau phí và thuế bán. Bot chỉ tư vấn, "
        "không tự đặt lệnh."
    )

    settings = get_settings(DB_PATH)
    positions = get_positions(DB_PATH)
    latest_rows = current_strategy_rows(get_latest_scan(DB_PATH))

    # Lấy một dòng có dữ liệu giá tốt nhất cho mỗi mã.
    market_by_symbol = {}
    signal_rank = {"BUY": 0, "WAIT": 1, "WATCH": 2, "SKIP": 3, "FILTERED": 4}
    for row in sorted(
        latest_rows,
        key=lambda x: (
            signal_rank.get(x.get("signal"), 9),
            -(float(x.get("score") or 0)),
        ),
    ):
        symbol = row.get("symbol")
        if not symbol or symbol in market_by_symbol:
            continue
        quote_is_verified = row.get("price_data_status") == "REALTIME"
        market_price = (
            row.get("realtime_price")
            if quote_is_verified
            else row.get("d1_close") or row.get("price")
        )
        market_by_symbol[symbol] = {
            "price": market_price,
            "source": "Realtime đã xác minh" if quote_is_verified else "D1/giá phiên gần nhất",
            "signal": row.get("signal"),
            "strategy": STRATEGY_NAMES.get(row.get("strategy"), row.get("strategy")),
            "score": row.get("score"),
            "reason": row.get("reason"),
        }

    st.subheader("Tổng quan")
    summary = get_portfolio_summary(DB_PATH)
    cash = get_cash_summary(DB_PATH)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mã đang nắm giữ", len(positions))
    c2.metric("Giá trị vốn còn lại", money(summary.get("total_cost", 0)))
    c3.metric("Lãi/lỗ đã thực hiện", money(summary.get("realized_pnl", 0)))
    c4.metric("Tiền mua ròng", money(cash.get("net_buys", 0) - cash.get("net_sells", 0)))

    if positions:
        display_rows = []
        total_market_value = 0.0
        total_unrealized = 0.0
        for position in positions:
            symbol = position["symbol"]
            market = market_by_symbol.get(symbol, {})
            current = market.get("price")
            avg_cost = float(position["avg_cost"])
            quantity = int(position["quantity"])
            if current is not None:
                current = float(current)
                net_value = quantity * current * (
                    1 - float(settings["fee_sell"]) - float(settings["sell_tax"])
                )
                cost_value = quantity * avg_cost
                unrealized = net_value - cost_value
                total_market_value += net_value
                total_unrealized += unrealized
                pnl_pct = unrealized / cost_value * 100 if cost_value else 0
            else:
                unrealized = None
                pnl_pct = None
            advice = market.get("signal", "CHƯA CÓ TÍN HIỆU")
            display_rows.append({
                "Mã": symbol,
                "Số lượng": quantity,
                "Giá vốn TB (gồm phí)": round(avg_cost, 2),
                "Giá tham chiếu": current,
                "Nguồn giá": market.get("source", "Chưa có dữ liệu"),
                "Lãi/lỗ tạm tính": round(unrealized, 0) if unrealized is not None else None,
                "Lãi/lỗ (%)": round(pnl_pct, 2) if pnl_pct is not None else None,
                "Tín hiệu mới nhất": SIGNAL_NAMES.get(advice, advice),
                "Chiến lược": market.get("strategy", "—"),
            })
        st.subheader("Vị thế đang nắm giữ")
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
        st.metric("Tổng lãi/lỗ tạm tính sau phí bán và thuế", money(total_unrealized))
        st.caption(
            "Giá tham chiếu ưu tiên realtime chỉ khi timestamp nguồn được xác minh; "
            "nếu không sẽ dùng giá D1/phiên gần nhất. Đây là số liệu ước tính, "
            "không phải giá khớp lệnh."
        )
        st.metric("Giá trị thị trường ròng ước tính", money(total_market_value))
    else:
        st.info("Chưa có vị thế. Ghi nhận giao dịch mua đầu tiên ở phần bên dưới.")

    st.divider()
    st.subheader("Ghi nhận giao dịch")
    with st.form("portfolio_trade_form", clear_on_submit=True):
        left, right = st.columns(2)
        side = left.selectbox("Loại giao dịch", ["MUA", "BÁN"])
        symbol = right.text_input("Mã cổ phiếu", max_chars=10).strip().upper()
        qty_col, price_col = st.columns(2)
        quantity = qty_col.number_input("Số lượng", min_value=100, step=100, value=100)
        price = price_col.number_input("Giá khớp (đồng/cổ phiếu)", min_value=0.01, step=0.1, value=10.0)
        note = st.text_input("Ghi chú (không bắt buộc)")
        submitted = st.form_submit_button("LƯU GIAO DỊCH", type="primary", use_container_width=True)
        if submitted:
            try:
                result = add_transaction(
                    side, symbol, int(quantity), float(price), note, db_path=DB_PATH
                )
                st.success(
                    f"Đã ghi nhận {side} {symbol}: {int(quantity):,} cp. "
                    f"Phí: {money(result['fee'])}; thuế: {money(result['tax'])}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Không thể lưu giao dịch: {exc}")

    with st.expander("Cài đặt vốn và phí giao dịch"):
        with st.form("portfolio_settings_form"):
            capital = st.number_input("Vốn dự kiến (đồng)", min_value=0.0, value=float(settings["capital"]), step=1000000.0)
            risk_pct = st.number_input("Rủi ro tối đa mỗi lệnh (%)", min_value=0.1, max_value=10.0, value=float(settings["risk_pct"]), step=0.1)
            fee_buy = st.number_input("Phí mua (%)", min_value=0.0, max_value=10.0, value=float(settings["fee_buy"]) * 100, step=0.01) / 100
            fee_sell = st.number_input("Phí bán (%)", min_value=0.0, max_value=10.0, value=float(settings["fee_sell"]) * 100, step=0.01) / 100
            sell_tax = st.number_input("Thuế bán (%)", min_value=0.0, max_value=10.0, value=float(settings["sell_tax"]) * 100, step=0.01) / 100
            save = st.form_submit_button("LƯU CÀI ĐẶT")
            if save:
                try:
                    save_settings({
                        "capital": capital,
                        "risk_pct": risk_pct,
                        "fee_buy": fee_buy,
                        "fee_sell": fee_sell,
                        "sell_tax": sell_tax,
                    }, DB_PATH)
                    st.success("Đã lưu cài đặt.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể lưu cài đặt: {exc}")

    with st.expander("Lịch sử giao dịch"):
        transactions = get_transactions(DB_PATH)
        if transactions:
            st.dataframe(pd.DataFrame(transactions), use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có giao dịch được ghi nhận.")

def show_system_info():

    st.title("⚙️ THÔNG TIN HỆ THỐNG")

    st.write(f"Phiên bản: AI Stock Scanner {APP_VERSION}")
    st.write("Thị trường: HOSE / HNX")
    st.write("Chu kỳ làm mới: 300 giây")
    st.write("Phân tích kỹ thuật: D1")
    st.write("Chiến lược: Swing / Reversal / Pullback")
    st.write("Tự động đặt lệnh: Không")
    st.write("Quản lý danh mục: Đã bật")

    st.warning(
        "Chu kỳ 300 giây là chu kỳ ứng dụng yêu cầu làm mới. "
        "Dữ liệu chỉ được gọi là realtime nếu nguồn KBS cung cấp "
        "timestamp hợp lệ và không quá 300 giây. "
        "Streamlit Cloud chỉ tự làm mới khi phiên trình duyệt "
        "đang hoạt động."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    with st.sidebar:

        st.title("AI STOCK SCANNER")

        page = st.radio(
            "Điều hướng",
            [
                "Tổng quan",
                "Quét cổ phiếu",
                "Lịch sử quét",
                "Quản lý danh mục",
                "Thông tin hệ thống",
            ],
        )

        st.divider()

        st.caption(
            f"AI Stock Scanner {APP_VERSION}"
        )

        st.caption(
            "Tự làm mới: 300 giây"
        )

    if page == "Tổng quan":
        show_dashboard()

    elif page == "Quét cổ phiếu":
        show_scanner()

    elif page == "Lịch sử quét":
        show_history()

    elif page == "Quản lý danh mục":
        show_portfolio()

    else:
        show_system_info()


if __name__ == "__main__":
    main()
