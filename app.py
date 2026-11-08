import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock, scan_market, INITIAL_CAPITAL
from data_provider import (
    fetch_stock_history,
    fetch_stock_quote,
    fetch_exchange_symbols
)

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI STOCK SCANNER")

st.caption(
    "Sàng lọc cổ phiếu HOSE/HNX cho chiến lược ngắn hạn "
    "2–5 phiên. Dữ liệu nến ngày, không phải realtime."
)

with st.sidebar:
    st.header("Quản trị vốn")

    capital = st.number_input(
        "Vốn giao dịch (đồng)",
        min_value=1_000_000,
        value=INITIAL_CAPITAL,
        step=1_000_000
    )

    risk_pct = st.number_input(
        "Rủi ro tối đa mỗi lệnh (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    ) / 100

    st.caption("Mặc định vốn 10 triệu, rủi ro 2%.")
    st.divider()

    st.caption(
        "Lịch sử nến ngày: DNSE, dự phòng Vietcap. "
        "Danh sách niêm yết: Vnstock KBS."
    )


tab1, tab2 = st.tabs([
    "Phân tích một mã",
    "Quét toàn thị trường"
])


with tab1:

    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10
    ).strip().upper()

    if st.button(
        "PHÂN TÍCH CỔ PHIẾU",
        type="primary",
        use_container_width=True
    ):

        if not symbol:
            st.error("Vui lòng nhập mã cổ phiếu.")

        else:
            try:
                with st.spinner("Đang lấy dữ liệu lịch sử..."):

                    history = fetch_stock_history(
                        symbol,
                        length=260
                    )

                    result = analyze_stock(
                        history,
                        capital,
                        risk_pct
                    )

                    quote = fetch_stock_quote(symbol)

                st.subheader(f"Kết quả: {symbol}")

                st.caption(
                    f"Nến ngày gần nhất: {quote.get('quote_date')} "
                    f"| Nguồn: {quote.get('source')}"
                )

                if result.get("signal") == "BUY":
                    st.success("🟢 TÍN HIỆU BUY")
                else:
                    st.warning("🟡 TÍN HIỆU WAIT")

                st.write(result.get("reason", ""))

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá đóng cửa",
                    f"{quote['last_price']:,.0f} đ"
                )

                c2.metric(
                    "Điểm kỹ thuật",
                    f"{result.get('score', 0)}/100"
                )

                c3.metric(
                    "RSI",
                    result.get("rsi", "—")
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá cắt lỗ",
                    f"{result.get('stop_loss', 0):,.0f} đ"
                )

                c2.metric(
                    "Giá mục tiêu",
                    f"{result.get('target_price', 0):,.0f} đ"
                )

                c3.metric(
                    "Lợi nhuận/rủi ro",
                    f"{result.get('reward_risk', 0)}:1"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Khối lượng dự kiến",
                    f"{result.get('quantity', 0):,} cp"
                )

                c2.metric(
                    "Vốn sử dụng",
                    f"{result.get('capital_used', 0):,.0f} đ"
                )

                c3.metric(
                    "Rủi ro dự kiến",
                    f"{result.get('risk_money', 0):,.0f} đ"
                )

                st.subheader("Yếu tố kỹ thuật")

                for reason in result.get("reasons", []):
                    st.write("• " + reason)

                with st.expander("10 phiên gần nhất"):
                    st.dataframe(
                        history.tail(10),
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as exc:
                st.error(
                    f"Không phân tích được {symbol}: {exc}"
                )


with tab2:

    st.subheader("Quét toàn bộ mã cổ phiếu HOSE và HNX")

    st.info(
        "Một lần bấm sẽ lấy danh sách mã theo sàn và quét "
        "lần lượt toàn bộ. Lần quét đầu có thể lâu do phải "
        "tải lịch sử từng mã."
    )

    if st.button(
        "QUÉT TOÀN BỘ HOSE + HNX",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)
        status = st.empty()

        try:
            with st.spinner("Đang lấy danh sách niêm yết..."):
                symbols = fetch_exchange_symbols()

            st.write(
                f"Danh sách hiện có {len(symbols)} mã HOSE + HNX."
            )

            def update_progress(done, total):
                progress.progress(
                    done / total if total else 1.0
                )

                status.write(
                    f"Đang quét {done}/{total} mã..."
                )

            output = scan_market(
                capital=capital,
                risk_percent=risk_pct,
                max_symbols=len(symbols),
                progress_callback=update_progress
            )

            st.success(
                f"Đã xử lý {output['scanned']} mã. "
                f"Tìm thấy {output['buy_count']} mã BUY."
            )

            if output["results"]:

                display = pd.DataFrame(output["results"])

                columns = [
                    "symbol",
                    "score",
                    "price",
                    "stop_loss",
                    "target_price",
                    "reward_risk",
                    "rsi",
                    "volume_ratio",
                    "quantity",
                    "data_date"
                ]

                display = display[
                    [c for c in columns if c in display.columns]
                ]

                display = display.rename(columns={
                    "symbol": "Mã",
                    "score": "Điểm",
                    "price": "Giá",
                    "stop_loss": "Cắt lỗ",
                    "target_price": "Mục tiêu",
                    "reward_risk": "LN/RR",
                    "rsi": "RSI",
                    "volume_ratio": "KL/TB20",
                    "quantity": "Số lượng",
                    "data_date": "Ngày dữ liệu"
                })

                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info(
                    "Không có mã nào đạt đủ điều kiện BUY. "
                    "Không có tín hiệu cũng là kết quả hợp lệ."
                )

            st.caption(
                f"Số mã lỗi dữ liệu: {output['error_count']}"
            )

            if output["errors"]:

                with st.expander("Xem lỗi dữ liệu"):

                    st.dataframe(
                        pd.DataFrame(output["errors"]),
                        use_container_width=True,
                        hide_index=True
                    )

        except Exception as exc:
            st.error(
                f"Không thể quét toàn thị trường: {exc}"
            )


st.divider()

st.caption(
    "Thời điểm hiển thị: "
    + datetime.now(
        ZoneInfo("Asia/Ho_Chi_Minh")
    ).strftime("%H:%M:%S %d/%m/%Y")
)

st.caption(
    "Giá mục tiêu là mức tính theo tỷ lệ lợi nhuận/rủi ro "
    "sau chi phí, không phải dự báo. Dữ liệu nến ngày có thể "
    "trễ; kiểm tra lại giá và thanh khoản trước khi đặt lệnh."
)
