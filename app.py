
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

from scan_store import (
    init_scan_db,
    get_latest_scan,
    get_scan_history,
)

from data_provider import fetch_stock_quote
from worker import run_scan


DB_PATH = os.getenv(
    "DATABASE_PATH",
    "data/stock_scanner.sqlite3"
)

st.set_page_config(
    page_title="AI Stock Scanner 2.2",
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

SIGNAL_ORDER = [
    "BUY",
    "WAIT",
    "WATCH",
    "SKIP",
    "FILTERED",
]

STRATEGY_NAMES = {
    "SWING": "Swing 2–5 phiên",
    "REVERSAL": "Đảo chiều 5–10 phiên",
    "PULLBACK": "Pullback 3–7 phiên",
    "CORE": "Bộ phân tích cũ",
}

STRATEGY_ORDER = [
    "SWING",
    "REVERSAL",
    "PULLBACK",
]


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


def safe_list(value):
    if isinstance(value, list):
        return value

    return []


def current_strategy_rows(rows):
    return [
        row for row in rows
        if row.get("strategy") in STRATEGY_ORDER
    ]


def price_status_name(status):
    return {
        "REALTIME": "Giá trong phiên",
        "D1_ONLY": "Chỉ có giá D1",
        "D1_STALE": "Dữ liệu D1 cũ",
    }.get(status, "Không rõ")


# =====================================================
# STOCK QUOTES
# =====================================================

def extract_price(raw):
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value > 0 else None

    if isinstance(raw, dict):
        for key in (
            "price",
            "close",
            "last_price",
            "lastPrice",
        ):
            try:
                value = float(raw.get(key))

                if value > 0:
                    return value

            except (TypeError, ValueError):
                continue

    return None


@st.cache_data(ttl=300, show_spinner=False)
def get_daily_quote(symbol):
    try:
        raw = fetch_stock_quote(symbol)

        return {
            "price": extract_price(raw),
            "quote_date": (
                raw.get("quote_date")
                if isinstance(raw, dict)
                else None
            ),
            "source": (
                raw.get("source")
                if isinstance(raw, dict)
                else "D1"
            ),
        }

    except Exception:
        return {
            "price": None,
            "quote_date": None,
            "source": None,
        }


# =====================================================
# PORTFOLIO
# =====================================================

def load_positions_with_prices():
    positions = get_positions(DB_PATH)

    rows = []

    for position in positions:
        symbol = position["symbol"]
        quantity = position["quantity"]

        quote = get_daily_quote(symbol)

        current_price = quote["price"]

        market_value = (
            quantity * current_price
            if current_price is not None
            else None
        )

        cost_value = position["total_cost"]

        unrealized_pnl = (
            market_value - cost_value
            if market_value is not None
            else None
        )

        pnl_percent = (
            unrealized_pnl / cost_value * 100
            if unrealized_pnl is not None
            and cost_value > 0
            else None
        )

        rows.append({
            "Mã": symbol,
            "Khối lượng": quantity,
            "Giá vốn bình quân": position["avg_cost"],
            "Tổng giá vốn": cost_value,
            "Giá đóng cửa D1": current_price,
            "Ngày dữ liệu": quote["quote_date"],
            "Giá trị thị trường": market_value,
            "Lãi/lỗ tạm tính": unrealized_pnl,
            "% lãi/lỗ": pnl_percent,
        })

    return rows


def show_portfolio_summary():
    summary = get_portfolio_summary(DB_PATH)
    positions = load_positions_with_prices()

    total_market_value = sum(
        row["Giá trị thị trường"] or 0
        for row in positions
    )

    total_unrealized = sum(
        row["Lãi/lỗ tạm tính"] or 0
        for row in positions
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
        "Lãi/lỗ tạm tính",
        money(total_unrealized),
    )

    col4.metric(
        "Lãi/lỗ đã thực hiện",
        money(summary["realized_pnl"]),
    )

    st.metric(
        "Số cổ phiếu đang nắm giữ",
        f"{summary['open_quantity']:,}",
    )

    return positions


# =====================================================
# SCAN RESULT DISPLAY
# =====================================================

def display_scan_table(rows):
    if not rows:
        st.info("Không có kết quả.")
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

    if "price_data_status" in df.columns:
        df["Trạng thái giá"] = (
            df["price_data_status"]
            .map({
                "REALTIME": "Giá trong phiên",
                "D1_ONLY": "Chỉ có giá D1",
                "D1_STALE": "Dữ liệu D1 cũ",
            })
            .fillna("Không rõ")
        )

    rename = {
        "symbol": "Mã",
        "score": "Điểm kỹ thuật",
        "price": "Giá dùng phân tích",
        "d1_close": "Giá đóng cửa D1",
        "realtime_price": "Giá trong phiên",
        "realtime_volume": "Khối lượng trong phiên",
        "quote_fetched_at": "Thời điểm lấy giá",
        "quote_source": "Nguồn giá",
        "stop_loss": "Cắt lỗ",
        "target_price": "Mục tiêu",
        "reward_risk": "RR sau phí",
        "reason": "Lý do",
        "scanned_at": "Thời gian quét",
        "data_date": "Ngày dữ liệu D1",
        "stop_distance_pct": "Rủi ro %",
        "target_distance_pct": "Tiềm năng %",
        "atr_pct": "ATR %",
        "relative_strength": "Sức mạnh vs VN-Index %",
        "market_context": "Bối cảnh thị trường",
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
        "Giá trong phiên",
        "Giá đóng cửa D1",
        "Giá dùng phân tích",
        "Cắt lỗ",
        "Mục tiêu",
        "RR sau phí",
        "Rủi ro %",
        "Tiềm năng %",
        "Ngày dữ liệu D1",
        "Thời điểm lấy giá",
        "Nguồn giá",
        "GTGD TB 20 phiên",
        "Lý do",
        "Thời gian quét",
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


def show_strategy_metrics(df):
    if df.empty:
        st.info("Chưa có kết quả.")
        return

    st.subheader("Thống kê theo chiến lược")

    cols = st.columns(3)

    for index, strategy in enumerate(STRATEGY_ORDER):
        strategy_df = df[
            df["strategy"] == strategy
        ]

        buy_count = int(
            (strategy_df["signal"] == "BUY").sum()
        )

        wait_count = int(
            (strategy_df["signal"] == "WAIT").sum()
        )

        watch_count = int(
            (strategy_df["signal"] == "WATCH").sum()
        )

        cols[index].metric(
            STRATEGY_NAMES[strategy],
            f"{buy_count} mã MUA",
        )

        cols[index].caption(
            f"Chờ: {wait_count} | Theo dõi: {watch_count}"
        )


def show_signal_explanation(row):
    st.subheader(
        f"Phân tích chi tiết: {row.get('symbol', '')}"
    )

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

    if status == "REALTIME":
        st.success("Đã nhận được giá trong phiên.")
    elif status == "D1_ONLY":
        st.warning(
            "Chưa nhận được giá trong phiên. "
            "Tín hiệu MUA chưa được xác nhận."
        )
    elif status == "D1_STALE":
        st.error("Dữ liệu D1 đã cũ.")
    else:
        st.info("Chưa xác định trạng thái dữ liệu giá.")

    col1, col2 = st.columns(2)

    col1.metric(
        "Giá trong phiên",
        money(row.get("realtime_price")),
    )

    col2.metric(
        "Giá đóng cửa D1",
        money(row.get("d1_close")),
    )

    st.write(
        f"**Giá dùng lập kế hoạch:** "
        f"{money(row.get('price'))}"
    )

    st.write(
        f"**Thời điểm lấy giá:** "
        f"{row.get('quote_fetched_at') or 'Không có'}"
    )

    st.write(
        f"**Nguồn giá:** "
        f"{row.get('quote_source') or 'Không có'}"
    )

    st.write(
        f"**Ngày dữ liệu D1:** "
        f"{row.get('data_date') or 'Không có dữ liệu'}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Cắt lỗ",
        money(row.get("stop_loss")),
    )

    col2.metric(
        "Mục tiêu",
        money(row.get("target_price")),
    )

    col3.metric(
        "RR sau phí",
        number(row.get("reward_risk")),
    )

    blockers = safe_list(row.get("blockers"))
    reasons = safe_list(row.get("reasons"))
    penalties = safe_list(row.get("penalties"))

    if reasons:
        st.markdown("**Điều kiện đã đạt**")

        for item in reasons:
            st.write(f"✓ {item}")

    if blockers:
        st.markdown("**Điều kiện chưa đạt**")

        for item in blockers:
            st.write(f"• {item}")

    if penalties:
        st.markdown("**Điểm cần thận trọng**")

        for item in penalties:
            st.write(f"• {item}")

    st.caption(
        "Điểm kỹ thuật không phải xác suất thắng. "
        "Tín hiệu MUA không bảo đảm lợi nhuận."
    )


# =====================================================
# DASHBOARD
# =====================================================

def show_dashboard():
    st.title("📈 AI STOCK SCANNER 2.2")

    st.caption(
        "Phân tích HOSE/HNX bằng chỉ báo D1, "
        "giá trong phiên và quản lý danh mục."
    )

    positions = show_portfolio_summary()

    st.divider()

    st.subheader("Danh mục hiện tại")

    if positions:
        st.dataframe(
            pd.DataFrame(positions),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Giá tham chiếu danh mục lấy từ dữ liệu D1, "
            "không phải giá realtime."
        )

    else:
        st.info("Bạn chưa có cổ phiếu trong danh mục.")

    st.divider()

    st.subheader("Kết quả quét gần nhất")

    latest = current_strategy_rows(
        get_latest_scan(DB_PATH)
    )

    if latest:
        df = pd.DataFrame(latest)

        show_strategy_metrics(df)

        realtime_count = int(
            (df.get(
                "price_data_status",
                pd.Series(dtype=str)
            ) == "REALTIME").sum()
        )

        unique_symbols = (
            df["symbol"].nunique()
            if "symbol" in df.columns
            else 0
        )

        st.caption(
            f"Có giá trong phiên: {realtime_count} dòng chiến lược "
            f"trên tổng {len(df)} dòng. "
            f"Số mã: {unique_symbols}."
        )

        st.divider()

        display_scan_table(latest[:30])

    else:
        st.info(
            "Chưa có kết quả của ba chiến lược mới. "
            "Vào mục Quét cổ phiếu để chạy quét."
        )


# =====================================================
# SCANNER
# =====================================================

def show_scanner():
    st.title("🔎 QUÉT CỔ PHIẾU")

    st.info(
        "Ba chiến lược độc lập: Swing 2–5 phiên, "
        "Đảo chiều 5–10 phiên và Pullback 3–7 phiên."
    )

    st.caption(
        "Yêu cầu tối thiểu 200 phiên lịch sử và thanh khoản đạt ngưỡng. "
        "Chỉ báo kỹ thuật tính trên D1. Giá trong phiên dùng để "
        "kiểm tra tín hiệu và lập kế hoạch giao dịch."
    )

    st.warning(
        "Điểm kỹ thuật không phải xác suất thắng. "
        "Nếu không lấy được giá trong phiên, bot không xác nhận BUY."
    )

    if st.button(
        "🚀 QUÉT THỊ TRƯỜNG NGAY",
        use_container_width=True,
        type="primary",
    ):
        progress_bar = st.progress(
            0,
            text="Chuẩn bị quét thị trường..."
        )

        progress_text = st.empty()

        try:
            def update_progress(percent, message):
                percent = max(
                    0,
                    min(100, int(percent))
                )

                progress_bar.progress(
                    percent,
                    text=f"{percent}%"
                )

                progress_text.caption(message)

            saved = run_scan(
                progress_callback=update_progress
            )

            get_daily_quote.clear()

            if saved > 0:
                st.success(
                    f"Hoàn thành lượt quét. "
                    f"Đã lưu {saved:,} dòng kết quả chiến lược."
                )
            else:
                st.warning(
                    "Lượt quét không lưu được kết quả. "
                    "Vui lòng kiểm tra Cloud logs."
                )

        except Exception as exc:
            st.error(f"Quét thất bại: {exc}")

        finally:
            progress_bar.empty()

    st.divider()

    latest = current_strategy_rows(
        get_latest_scan(DB_PATH)
    )

    if not latest:
        st.warning(
            "Chưa có kết quả của ba chiến lược. "
            "Hãy chạy quét thị trường trước."
        )
        return

    df = pd.DataFrame(latest)

    # -------------------------------------------------
    # DATA COVERAGE
    # -------------------------------------------------

    st.subheader("Tình trạng dữ liệu giá")

    if "price_data_status" in df.columns:
        realtime_count = int(
            (df["price_data_status"] == "REALTIME").sum()
        )

        d1_only_count = int(
            (df["price_data_status"] == "D1_ONLY").sum()
        )

        stale_count = int(
            (df["price_data_status"] == "D1_STALE").sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Có giá trong phiên",
            realtime_count,
        )

        col2.metric(
            "Chỉ có giá D1",
            d1_only_count,
        )

        col3.metric(
            "D1 cũ",
            stale_count,
        )

        st.caption(
            "Đây là số dòng chiến lược, không phải số mã duy nhất. "
            "Mỗi mã có thể xuất hiện trong ba chiến lược."
        )

    else:
        st.info(
            "Kết quả cũ chưa có thông tin trạng thái giá."
        )

    show_strategy_metrics(df)

    st.divider()

    # -------------------------------------------------
    # SIGNAL STATISTICS
    # -------------------------------------------------

    st.subheader("Thống kê tín hiệu")

    signal_counts = df["signal"].value_counts()

    cols = st.columns(5)

    for index, signal in enumerate(SIGNAL_ORDER):
        cols[index].metric(
            SIGNAL_NAMES[signal],
            int(signal_counts.get(signal, 0)),
        )

    st.caption(
        "Thống kê theo số dòng chiến lược, "
        "không phải số mã cổ phiếu duy nhất."
    )

    st.divider()

    # -------------------------------------------------
    # SHORTLIST
    # -------------------------------------------------

    st.subheader("⭐ Danh sách ưu tiên theo dõi")

    candidates = df[
        df["signal"].isin(["BUY", "WAIT"])
    ].copy()

    if not candidates.empty:
        candidates["_signal_rank"] = (
            candidates["signal"]
            .map({"BUY": 0, "WAIT": 1})
        )

        candidates["_rr"] = pd.to_numeric(
            candidates["reward_risk"],
            errors="coerce",
        )

        candidates = candidates.sort_values(
            [
                "_signal_rank",
                "score",
                "_rr",
            ],
            ascending=[True, False, False],
            na_position="last",
        )

        display_scan_table(
            candidates.head(15).drop(
                columns=["_signal_rank", "_rr"]
            ).to_dict("records")
        )

        st.markdown("**Xem lý do của từng mã**")

        candidate_options = [
            (
                f"{row['symbol']} - "
                f"{STRATEGY_NAMES.get(row['strategy'], row['strategy'])} "
                f"- {SIGNAL_NAMES.get(row['signal'], row['signal'])} "
                f"- {row.get('score', 0)} điểm"
            )
            for _, row in candidates.head(30).iterrows()
        ]

        selected_candidate = st.selectbox(
            "Chọn mã cần phân tích chi tiết",
            candidate_options,
            key="candidate_detail",
        )

        selected_index = candidate_options.index(
            selected_candidate
        )

        selected_row = candidates.head(30).iloc[
            selected_index
        ].to_dict()

        show_signal_explanation(selected_row)

    else:
        st.info(
            "Hiện chưa có mã MUA hoặc CHỜ. "
            "Không có tín hiệu cũng là một kết quả hợp lệ."
        )

    st.divider()

    # -------------------------------------------------
    # FILTERS
    # -------------------------------------------------

    st.subheader("Lọc kết quả")

    strategy_options = {
        "Tất cả chiến lược": "ALL",
        "Swing 2–5 phiên": "SWING",
        "Đảo chiều 5–10 phiên": "REVERSAL",
        "Pullback 3–7 phiên": "PULLBACK",
    }

    selected_strategy_label = st.selectbox(
        "Chọn chiến lược",
        list(strategy_options.keys()),
    )

    selected_strategy = strategy_options[
        selected_strategy_label
    ]

    signal_options = {
        "Tất cả tín hiệu": "ALL",
        "MUA xác nhận": "BUY",
        "Chờ xác nhận": "WAIT",
        "Theo dõi": "WATCH",
        "Bỏ qua": "SKIP",
        "Bị loại": "FILTERED",
    }

    selected_signal_label = st.selectbox(
        "Chọn tín hiệu",
        list(signal_options.keys()),
    )

    selected_signal = signal_options[
        selected_signal_label
    ]

    filtered = df.copy()

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

        display_scan_table(
            filtered.to_dict("records")
        )

    st.divider()

    # -------------------------------------------------
    # HISTORY
    # -------------------------------------------------

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
# PORTFOLIO
# =====================================================

def show_portfolio():
    st.title("💼 QUẢN LÝ DANH MỤC")

    st.warning(
        "Bot chỉ ghi nhận giao dịch do bạn nhập. "
        "Không tự động đặt lệnh mua hoặc bán."
    )

    st.subheader("Tổng quan danh mục")

    show_portfolio_summary()

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

        st.write(
            f"Khối lượng: **{pending['quantity']:,}**"
        )

        st.write(
            f"Giá khớp: **{money(pending['price'])}**"
        )

        st.write(
            f"Giá trị giao dịch: **{money(preview['gross'])}**"
        )

        st.write(
            f"Phí: **{money(preview['fee'])}**"
        )

        st.write(
            f"Thuế: **{money(preview['tax'])}**"
        )

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
            pd.DataFrame(positions),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Giá D1 là giá đóng cửa gần nhất từ nguồn dữ liệu. "
            "Lãi/lỗ chưa thực hiện chỉ là ước tính."
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

    st.caption(
        "Không sử dụng vốn mặc định cố định. "
        "Giá vốn danh mục được tính từ giá trị còn lại "
        "của các cổ phiếu đang nắm giữ."
    )

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
        save_settings({
            "risk_pct": risk_pct,
            "fee_buy": fee_buy / 100,
            "fee_sell": fee_sell / 100,
            "sell_tax": sell_tax / 100,
        }, DB_PATH)

        st.success("Đã lưu cài đặt.")
        st.rerun()

    st.caption(
        "Hãy đối chiếu phí mua, phí bán và thuế "
        "với xác nhận giao dịch thực tế của công ty chứng khoán."
    )

    st.divider()

    st.subheader("Thông tin hệ thống")

    st.write("Phiên bản: AI Stock Scanner 2.2")
    st.write("Thị trường: HOSE / HNX")
    st.write("Khung chỉ báo kỹ thuật: D1")
    st.write("Giá xác nhận: dữ liệu trong phiên nếu có")
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

        st.caption("AI Stock Scanner 2.2")
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
