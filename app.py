
import streamlit as st
import pandas as pd
from scanner import analyze_stock

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI STOCK SCANNER")
st.caption("Vietnam Stock Market | Swing Trading 2–5 sessions")

st.info(
    "Phiên bản 2.0: Đã tích hợp bộ máy phân tích kỹ thuật. "
    "Hiện tại sử dụng dữ liệu CSV, chưa kết nối giá thị trường trực tiếp."
)

# Configuration
st.sidebar.header("⚙️ Cấu hình giao dịch")
capital = st.sidebar.number_input(
    "Vốn đầu tư (VNĐ)",
    min_value=1000000,
    value=10000000,
    step=1000000
)

risk_percent = st.sidebar.selectbox(
    "Rủi ro mỗi giao dịch",
    options=[0.005, 0.01, 0.02],
    index=2,
    format_func=lambda x: f"{x * 100:.1f}%"
)

st.sidebar.write("Thị trường: HOSE và HNX")
st.sidebar.write("Thời gian nắm giữ: 2–5 phiên")
st.sidebar.write("Điểm BUY tối thiểu: 85/100")

st.divider()

st.subheader("🔎 Phân tích cổ phiếu")

ticker = st.text_input(
    "Mã cổ phiếu",
    placeholder="Ví dụ: FPT, HPG, SSI..."
).strip().upper()

uploaded_file = st.file_uploader(
    "Tải dữ liệu lịch sử cổ phiếu dạng CSV",
    type=["csv"]
)

st.caption(
    "CSV cần có các cột: Date, Open, High, Low, Close, Volume. "
    "Cần tối thiểu 200 phiên dữ liệu."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.write("### Kiểm tra dữ liệu")
        st.write(f"Số dòng dữ liệu: {len(df)}")

        with st.expander("Xem dữ liệu"):
            st.dataframe(df.tail(10), use_container_width=True)

        if st.button("🚀 PHÂN TÍCH CỔ PHIẾU", use_container_width=True):
            with st.spinner("Đang phân tích dữ liệu..."):
                result = analyze_stock(
                    df,
                    capital=capital,
                    risk_percent=risk_percent
                )

            st.divider()
            st.subheader(
                f"📊 KẾT QUẢ PHÂN TÍCH: {ticker or 'CHƯA NHẬP MÃ'}"
            )

            signal = result.get("signal", "REJECT")

            if signal == "BUY":
                st.success("🟢 TÍN HIỆU: BUY")
            elif signal == "WAIT":
                st.warning("🟡 TÍN HIỆU: WAIT")
            else:
                st.error("🔴 TÍN HIỆU: REJECT")

            if "score" in result:
                st.metric("Điểm kỹ thuật", f"{result['score']}/100")

            if "reason" in result:
                st.error(result["reason"])

            if "message" in result:
                st.write(result["message"])

            if "price" in result:
                col1, col2 = st.columns(2)
                col1.metric(
                    "Giá phân tích",
                    f"{result['price']:,.0f} VNĐ"
                )
                col2.metric(
                    "RSI",
                    result.get("rsi", "N/A")
                )

                col3, col4 = st.columns(2)
                col3.metric(
                    "Giá cắt lỗ",
                    f"{result['stop_loss']:,.0f} VNĐ"
                )
                col4.metric(
                    "Giá mục tiêu",
                    f"{result['target_price']:,.0f} VNĐ"
                )

                col5, col6 = st.columns(2)
                col5.metric(
                    "Số lượng cổ phiếu",
                    f"{result['quantity']:,}"
                )
                col6.metric(
                    "Vốn sử dụng",
                    f"{result['capital_used']:,.0f} VNĐ"
                )

                st.write(
                    "Tỷ lệ lợi nhuận/rủi ro:",
                    result["reward_risk"]
                )
                st.write(
                    "Rủi ro dự kiến:",
                    f"{result['risk_money']:,.0f} VNĐ"
                )
                st.write(
                    "Ngày dữ liệu:",
                    result["data_date"]
                )

            if result.get("reasons"):
                st.write("### Lý do phân tích")
                for reason in result["reasons"]:
                    st.write(f"✅ {reason}")

    except Exception as error:
        st.error(f"Lỗi dữ liệu: {error}")

else:
    st.warning(
        "Chưa có dữ liệu cổ phiếu. "
        "Hệ thống chưa thể đưa ra khuyến nghị mua bán."
    )

st.divider()

st.caption(
    "AI STOCK SCANNER | Công cụ hỗ trợ phân tích, "
    "không tự động đặt lệnh và không đảm bảo lợi nhuận."
)
