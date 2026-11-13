
import os
import streamlit as st
import pandas as pd

from portfolio import (
    init_db,
    get_settings,
    save_settings,
    get_positions,
    get_transactions,
    get_portfolio_summary,
    get_cash_summary,
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


def money(value):
    try:
        return f"{float(value):,.0f} đ"
    except (TypeError, ValueError):
        return "N/A"


def extract_price(raw):
    """Chuẩn hóa kết quả giá từ data_provider."""

    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None

    if isinstance(raw, dict):
        for key in (
            "price",
            "close",
            "last_price",
            "lastPrice",
        ):
            value = raw.get(key)
            try:
                value = float(value)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue

    return None


@st.cache_data(ttl=60, show_spinner=False)
def get_daily_price(symbol):
    try:
        raw = fetch_stock_quote(symbol)
        return extract_price(raw)
    except Exception:
        return None


def load_positions_with_prices():
    positions = get_positions(DB_PATH)
    rows = []

    for position in positions:
        symbol = position["symbol"]
        quantity = position["quantity"]
        avg_cost = position["avg_cost"]

        current_price = get_daily_price(symbol)

        market_value = (
            quantity * current_price
            if current_price is not None else None
        )

        cost_value = position["total_cost"]

        unrealized_pnl = (
            market_value - cost_value
            if market_value is not None else None
        )

        pnl_percent = (
            unrealized_pnl / cost_value * 100
            if unrealized_pnl is not None and cost_value > 0
            else None
        )

        rows.append({
            "Mã": symbol,
            "Khối lượng": quantity,
            "Giá vốn": avg_cost,
            "Giá D1": current_price,
            "Giá trị thị trường": market_value,
            "Lãi/lỗ tạm tính": unrealized_pnl,
            "% lãi/lỗ": pnl_percent,
        })

    return rows


def show_dashboard():
    st.title("📈 AI STOCK SCANNER 2.0")
    st.caption(
        "Hệ thống phân tích cổ phiếu HOSE/HNX "
        "và quản lý danh mục đầu tư."
    )

    settings = get_settings(DB_PATH)
    summary = get_portfolio_summary(DB_PATH)
    cash = get_cash_summary(DB_PATH)

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
        df = pd.DataFrame(positions)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Giá thị trường lấy từ dữ liệu đóng cửa D1. "
            "Không phải giá realtime."
        )
    else:
        st.info("Bạn chưa có cổ phiếu trong danh mục.")

    st.divider()

    st.subheader("Kết quả quét gần nhất")

    latest = get_latest_scan(DB_PATH)

    if latest:
        df = pd.DataFrame(latest)

        columns = [
            "symbol",
            "strategy",
            "signal",
            "score",
            "price",
            "stop_loss",
            "target_price",
        ]

        df = df[
            [col for col in columns if col in df.columns]
        ]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Chưa có dữ liệu quét. Hãy vào mục Quét cổ phiếu.")


def show_scanner():
    st.title("🔎 QUÉT CỔ PHIẾU")

    st.info(
        "Phân tích kỹ thuật bằng dữ liệu nến ngày D1. "
        "Chu kỳ chạy quét không đồng nghĩa với dữ liệu giá realtime."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button(
            "🚀 QUÉT THỊ TRƯỜNG NGAY",
            use_container_width=True,
        ):
            with st.spinner(
                "Đang quét HOSE/HNX. Vui lòng chờ..."
            ):
                try:
                    run_scan()
                    st.success("Đã hoàn thành lượt quét.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Quét thất bại: {exc}")

    latest = get_latest_scan(DB_PATH)

    if not latest:
        st.warning("Chưa có kết quả. Hãy chạy quét trước.")
        return

    df = pd.DataFrame(latest)

    strategy_names = {
        "SWING": "Lướt sóng 2–5 phiên",
        "REVERSAL": "Đảo chiều 5–10 phiên",
        "PULLBACK": "Điều chỉnh 3–7 phiên",
    }

    strategy_options = {
        "Tất cả chiến lược": "ALL",
        "Lướt sóng 2–5 phiên": "SWING",
        "Đảo chiều 5–10 phiên": "REVERSAL",
        "Điều chỉnh 3–7 phiên": "PULLBACK",
    }

    selected_strategy = st.selectbox(
        "Lọc chiến lược",
        list(strategy_options.keys()),
    )

    selected_code = strategy_options[selected_strategy]

    if selected_code != "ALL":
        df = df[df["strategy"] == selected_code]

    signal_options = [
        "Tất cả tín hiệu",
        "MUA",
        "CHỜ MUA",
        "THEO DÕI",
        "BỎ QUA",
    ]

    selected_signal = st.selectbox(
        "Lọc tín hiệu",
        signal_options,
    )

    if selected_signal != "Tất cả tín hiệu":
        df = df[df["signal"] == selected_signal]

    if df.empty:
        st.info("Không có cổ phiếu phù hợp bộ lọc.")
        return

    display = df.rename(columns={
        "symbol": "Mã",
        "strategy": "Chiến lược",
        "signal": "Tín hiệu",
        "score": "Điểm",
        "price": "Giá D1",
        "stop_loss": "Cắt lỗ tham khảo",
        "target_price": "Mục tiêu tham khảo",
        "reason": "Lý do",
        "scanned_at": "Thời gian quét",
    })

    if "Chiến lược" in display.columns:
        display["Chiến lược"] = (
            display["Chiến lược"].map(strategy_names)
            .fillna(display["Chiến lược"])
        )

    st.subheader(
        f"Có {len(display)} kết quả phù hợp"
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Điểm số là điểm kỹ thuật tham khảo, "
        "không phải xác suất sinh lời."
    )

    st.divider()

    st.subheader("Lịch sử quét")

    history = get_scan_history(
        limit=100,
        db_path=DB_PATH,
    )

    if history:
        st.dataframe(
            pd.DataFrame(history),
            use_container_width=True,
            hide_index=True,
        )


def show_portfolio():
    st.title("💼 QUẢN LÝ DANH MỤC")

    st.warning(
        "Giao dịch chỉ được lưu sau khi bạn xác nhận. "
        "Hệ thống không tự đặt lệnh lên sàn."
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
            pd.DataFrame(positions),
            use_container_width=True,
            hide_index=True,
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
        "Hãy kiểm tra phí thực tế theo biểu phí "
        "của công ty chứng khoán bạn sử dụng."
    )

    st.divider()

    st.subheader("Thông tin hệ thống")

    st.write("Phiên bản: AI Stock Scanner 2.0")
    st.write("Thị trường: HOSE / HNX")
    st.write("Khung phân tích: D1")
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
