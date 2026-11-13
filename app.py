
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
    page_title="AI Stock Scanner 2.0",
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
    "EARLY_BUY": "MUA SỚM - CHỜ XÁC NHẬN",
    "NEAR_BUY": "SÁT ĐIỂM MUA",
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


def money(value):
    try:
        if value is None:
            return "N/A"

        return f"{float(value):,.0f} đ"

    except (TypeError, ValueError):
        return "N/A"


def number(value, digits=2):
    try:
        if value is None:
            return "N/A"

        return f"{float(value):,.{digits}f}"

    except (TypeError, ValueError):
        return "N/A"


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
            if unrealized_pnl is not None and cost_value > 0
            else None
        )

        rows.append({
            "Mã": symbol,
            "Khối lượng": quantity,
            "Giá vốn bình quân": position["avg_cost"],
            "Giá đóng cửa D1": current_price,
            "Ngày dữ liệu": quote["quote_date"],
            "Giá trị thị trường": market_value,
            "Lãi/lỗ tạm tính": unrealized_pnl,
            "% lãi/lỗ": pnl_percent,
        })

    return rows


def current_strategy_rows(rows):
    """
    Chỉ lấy ba chiến lược mới.
    Không đưa kết quả CORE cũ vào thống kê mới.
    """

    return [
        row for row in rows
        if row.get("strategy") in STRATEGY_ORDER
    ]


def display_scan_table(rows):
    if not rows:
        st.info("Chưa có kết quả quét.")
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
        "price": "Giá D1",
        "stop_loss": "Cắt lỗ tham khảo",
        "target_price": "Mục tiêu tham khảo",
        "reward_risk": "Tỷ lệ lời/lỗ",
        "reason": "Lý do",
        "scanned_at": "Thời gian quét",
    }

    df = df.rename(columns=rename)

    preferred = [
        "Mã",
        "Chiến lược",
        "Tín hiệu",
        "Điểm",
        "Giá D1",
        "Cắt lỗ tham khảo",
        "Mục tiêu tham khảo",
        "Tỷ lệ lời/lỗ",
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
        st.info("Chưa có kết quả của ba chiến lược mới.")
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
            f"{buy_count} mã MUA"
        )

        cols[index].caption(
            f"Chờ: {wait_count} | Theo dõi: {watch_count}"
        )


def show_dashboard():
    st.title("📈 AI STOCK SCANNER 2.0")

    st.caption(
        "Phân tích cổ phiếu HOSE/HNX bằng dữ liệu D1 "
        "và quản lý danh mục đầu tư."
    )

    settings = get_settings(DB_PATH)
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

    col1, col2 = st.columns(2)

    col1.metric(
        "Vốn cấu hình",
        money(settings["capital"])
    )

    col2.metric(
        "Tổng giá vốn còn nắm giữ",
        money(summary["total_cost"])
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Giá trị thị trường",
        money(total_market_value)
    )

    col4.metric(
        "Lãi/lỗ tạm tính",
        money(total_unrealized)
    )

    col5, col6 = st.columns(2)

    col5.metric(
        "Lãi/lỗ đã thực hiện",
        money(summary["realized_pnl"])
    )

    col6.metric(
        "Số cổ phiếu đang nắm giữ",
        f"{summary['open_quantity']:,}"
    )

    st.divider()

    st.subheader("Danh mục hiện tại")

    if positions:
        st.dataframe(
            pd.DataFrame(positions),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Giá tham chiếu lấy từ nến ngày D1 gần nhất. "
            "Không phải giá realtime."
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

        st.divider()

        display_scan_table(latest[:30])

    else:
        st.info(
            "Chưa có kết quả của ba chiến lược mới. "
            "Vào mục Quét cổ phiếu để chạy quét."
        )


def show_scanner():
    st.title("🔎 QUÉT CỔ PHIẾU")

    st.info(
        "Phân tích ba chiến lược độc lập bằng nến ngày D1: "
        "Swing 2–5 phiên, Đảo chiều 5–10 phiên "
        "và Pullback 3–7 phiên."
    )

    st.caption(
        "Bộ lọc yêu cầu tối thiểu 200 phiên lịch sử, "
        "thanh khoản bình quân và trung vị đạt ngưỡng. "
        "Tín hiệu MUA còn phải đạt tỷ lệ lời/lỗ tối thiểu 2."
    )

    st.warning(
        "Tín hiệu kỹ thuật không bảo đảm lợi nhuận. "
        "Giá, điểm cắt lỗ và mục tiêu đều là tham khảo từ dữ liệu D1."
    )

    if st.button(
        "🚀 QUÉT THỊ TRƯỜNG NGAY",
        use_container_width=True,
    ):
        with st.spinner(
            "Đang quét HOSE/HNX bằng dữ liệu D1. "
            "Quá trình có thể mất vài phút..."
        ):
            try:
                saved = run_scan()

                if saved > 0:
                    st.success(
                        f"Đã hoàn thành. Lưu {saved} kết quả chiến lược."
                    )
                    get_daily_quote.clear()

                else:
                    st.warning(
                        "Lượt quét chưa lưu được kết quả. "
                        "Kiểm tra nhật ký worker hoặc nguồn dữ liệu."
                    )

                st.rerun()

            except Exception as exc:
                st.error(f"Quét thất bại: {exc}")

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

    show_strategy_metrics(df)

    st.divider()

    st.subheader("Thống kê tín hiệu")

    signal_counts = df["signal"].value_counts()

    cols = st.columns(5)

    for index, signal in enumerate(SIGNAL_ORDER):
        cols[index].metric(
            SIGNAL_NAMES[signal],
            int(signal_counts.get(signal, 0))
        )

    st.caption(
        "Thống kê tính theo số dòng chiến lược, "
        "không phải số mã cổ phiếu duy nhất."
    )

    st.divider()

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

    if "score" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(
                filtered["score"],
                errors="coerce"
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

    st.subheader("Lịch sử quét")

    history = get_scan_history(
        limit=150,
        db_path=DB_PATH,
    )

    history = current_strategy_rows(history)

    if history:
        history_df = pd.DataFrame(history)

        display_scan_table(
            history_df.to_dict("records")
        )

    else:
        st.info(
            "Chưa có lịch sử của ba chiến lược mới."
        )


def show_portfolio():
    st.title("💼 QUẢN LÝ DANH MỤC")

    st.warning(
        "Bot chỉ ghi nhận giao dịch do bạn nhập. "
        "Không tự động đặt lệnh mua hoặc bán."
    )

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


def show_settings():
    st.title("⚙️ CÀI ĐẶT HỆ THỐNG")

    settings = get_settings(DB_PATH)

    st.subheader("Quản lý vốn và rủi ro")

    with st.form("settings_form"):
        capital = st.number_input(
            "Tổng vốn đầu tư (đồng)",
            min_value=1_000_000,
            value=int(settings["capital"]),
            step=1_000_000,
        )

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
            "capital": capital,
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

    st.write("Phiên bản: AI Stock Scanner 2.0")
    st.write("Thị trường: HOSE / HNX")
    st.write("Khung phân tích: D1")
    st.write("Chiến lược: Swing / Reversal / Pullback")
    st.write("Tự động đặt lệnh: Không")


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

        st.caption("AI Stock Scanner 2.0")
        st.caption("Phân tích kỹ thuật D1")

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
