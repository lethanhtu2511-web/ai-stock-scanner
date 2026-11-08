
import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import (
    analyze_stock,
    scan_market,
    INITIAL_CAPITAL,
    RISK_PERCENT
)

from data_provider import (
    fetch_stock_history,
    fetch_stock_quote
)


st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI STOCK SCANNER")

st.caption(
    "Công cụ sàng lọc cổ phiếu HOSE/HNX "
    "cho chiến lược giao dịch ngắn hạn 2–5 phiên."
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
        "Dữ liệu phụ thuộc nhà cung cấp. "
        "Không mặc định là dữ liệu thời gian thực."
    )


tab1, tab2 = st.tabs([
    "Phân tích một mã",
    "Quét thị trường"
])


# =====================================
# PHÂN TÍCH MỘT MÃ
# =====================================

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
                with st.spinner("Đang lấy dữ liệu..."):

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
                    f"Dữ liệu lịch sử đến {result.get('data_date')} "
                    f"| Nguồn bảng giá: {quote.get('source')}"
                )

                if result.get("signal") == "BUY":
                    st.success("🟢 TÍN HIỆU BUY")
                else:
                    st.warning("🟡 TÍN HIỆU WAIT")

                st.write(result.get("reason", ""))

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá gần nhất",
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

                if result.get("reasons"):
                    for reason in result["reasons"]:
                        st.write("• " + reason)
                else:
                    st.write("Chưa có yếu tố kỹ thuật tích cực.")

                with st.expander("Chi tiết bảng giá"):

                    st.write({
                        "Sàn": quote.get("exchange"),
                        "Giá tham chiếu": quote.get("reference_price"),
                        "Giá mở cửa": quote.get("open_price"),
                        "Giá trần": quote.get("ceiling_price"),
                        "Giá sàn": quote.get("floor_price"),
                        "Giá mua tốt nhất": quote.get("bid_price_1"),
                        "Giá bán tốt nhất": quote.get("ask_price_1"),
                        "Khối lượng mua": quote.get("bid_vol_1"),
                        "Khối lượng bán": quote.get("ask_vol_1")
                    })

            except Exception as exc:
                st.error(f"Không phân tích được {symbol}: {exc}")


# =====================================
# QUÉT THỊ TRƯỜNG
# =====================================

with tab2:

    st.subheader("Quét cổ phiếu HOSE và HNX")

    st.warning(
        "Mỗi mã cần truy vấn dữ liệu lịch sử. "
        "Lượt quét mặc định giới hạn số mã để "
        "hạn chế vượt hạn mức API."
    )

    max_symbols = st.slider(
        "Số mã tối đa mỗi lượt",
        min_value=5,
        max_value=50,
        value=30,
        step=5
    )

    if st.button(
        "BẮT ĐẦU QUÉT",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)
        status = st.empty()

        def update_progress(done, total):
            progress.progress(
                done / total if total else 1.0
            )
            status.write(
                f"Đang quét {done}/{total} mã..."
            )

        try:
            with st.spinner("Đang quét thị trường..."):

                output = scan_market(
                    capital=capital,
                    risk_percent=risk_pct,
                    max_symbols=max_symbols,
                    progress_callback=update_progress
                )

            st.success(
                f"Đã quét {output['scanned']} mã. "
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
                    "Không có mã nào đạt đầy đủ điều kiện BUY. "
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
            st.error(f"Không thể quét thị trường: {exc}")


st.divider()

st.caption(
    "Thời điểm hiển thị: "
    + datetime.now(
        ZoneInfo("Asia/Ho_Chi_Minh")
    ).strftime("%H:%M:%S %d/%m/%Y")
)

st.caption(
    "Giá mục tiêu là mức tính toán theo tỷ lệ lợi nhuận/rủi ro "
    "sau chi phí, không phải dự báo giá. "
    "Lệnh thực tế có thể chịu trượt giá, gap và bước giá của sàn. "
    "Cần kiểm thử trước khi giao dịch bằng tiền thật."
)
