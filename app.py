
import streamlit as st
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock
from data_provider import fetch_stock_history, fetch_stock_quote


# =====================================
# CẤU HÌNH WEBSITE
# =====================================

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI STOCK SCANNER")

st.caption(
    "Hệ thống phân tích cổ phiếu Việt Nam "
    "phục vụ giao dịch ngắn hạn 2–5 phiên."
)

st.info(
    "Dữ liệu phân tích kỹ thuật sử dụng nến ngày. "
    "Bảng giá trong phiên được lấy riêng từ nguồn KBS "
    "khi bạn thực hiện phân tích."
)


# =====================================
# LẤY DỮ LIỆU LỊCH SỬ
# =====================================

@st.cache_data(ttl=900, show_spinner=False)
def get_stock_data(symbol):
    return fetch_stock_history(symbol, length=260)


# =====================================
# CẤU HÌNH VỐN VÀ RỦI RO
# =====================================

st.sidebar.header("⚙️ Cấu hình đầu tư")

capital = st.sidebar.number_input(
    "Vốn đầu tư (đồng)",
    min_value=1_000_000,
    max_value=1_000_000_000,
    value=10_000_000,
    step=1_000_000
)

risk_percent = st.sidebar.selectbox(
    "Rủi ro tối đa mỗi giao dịch",
    options=[0.005, 0.01, 0.015, 0.02],
    index=3,
    format_func=lambda x: f"{x * 100:.1f}%"
)

st.sidebar.write(
    f"Ngân sách rủi ro: **{capital * risk_percent:,.0f} đồng**"
)

st.sidebar.divider()

st.sidebar.write("**Chiến lược hiện tại**")
st.sidebar.write("- Thị trường: HOSE và HNX")
st.sidebar.write("- Thời gian nắm giữ: 2–5 phiên")
st.sidebar.write("- Điểm BUY tối thiểu: 85/100")
st.sidebar.write("- Tỷ lệ lợi nhuận/rủi ro: 2:1")
st.sidebar.write("- Lô giao dịch: 100 cổ phiếu")


# =====================================
# NHẬP MÃ CỔ PHIẾU
# =====================================

st.subheader("🔎 Phân tích cổ phiếu")

with st.form("stock_analysis_form"):

    symbol = st.text_input(
        "Nhập mã cổ phiếu HOSE hoặc HNX",
        placeholder="Ví dụ: FPT, HPG, SSI, SHS..."
    )

    submitted = st.form_submit_button(
        "PHÂN TÍCH CỔ PHIẾU",
        type="primary",
        use_container_width=True
    )


# =====================================
# HÀM HIỂN THỊ GIÁ
# =====================================

def show_price(label, value):
    if value is None:
        return "—"

    try:
        return f"{float(value):,.0f} đ"
    except (ValueError, TypeError):
        return "—"


# =====================================
# THỰC HIỆN PHÂN TÍCH
# =====================================

if submitted:

    symbol = symbol.strip().upper()

    if not symbol:

        st.warning("Bạn chưa nhập mã cổ phiếu.")

    else:

        try:

            with st.spinner(
                f"Đang lấy dữ liệu và phân tích {symbol}..."
            ):

                # Lấy nến ngày để phân tích kỹ thuật
                df = get_stock_data(symbol)

                result = analyze_stock(
                    df,
                    capital=capital,
                    risk_percent=risk_percent
                )

            st.divider()

            st.subheader(f"Kết quả phân tích: {symbol}")

            # =====================================
            # BẢNG GIÁ TRONG PHIÊN
            # =====================================

            st.subheader("📡 Bảng giá mới nhất")

            try:

                with st.spinner("Đang lấy bảng giá KBS..."):

                    quote = fetch_stock_quote(symbol)

                current_time = datetime.now(
                    ZoneInfo("Asia/Ho_Chi_Minh")
                ).strftime("%H:%M:%S %d/%m/%Y")

                st.caption(
                    f"Nguồn: {quote['source']} | "
                    f"Thời điểm truy vấn: {current_time}"
                )

                q1, q2, q3 = st.columns(3)

                q1.metric(
                    "Giá khớp gần nhất",
                    show_price("Giá", quote["last_price"])
                )

                q2.metric(
                    "Giá tham chiếu",
                    show_price("Tham chiếu", quote["reference_price"])
                )

                q3.metric(
                    "Giá mở cửa",
                    show_price("Mở cửa", quote["open_price"])
                )

                q4, q5, q6 = st.columns(3)

                q4.metric(
                    "Giá trần",
                    show_price("Trần", quote["ceiling_price"])
                )

                q5.metric(
                    "Giá sàn",
                    show_price("Sàn", quote["floor_price"])
                )

                q6.metric(
                    "Sàn giao dịch",
                    quote["exchange"] or "Chưa xác định"
                )

                st.write("**Độ sâu bảng giá cấp 1**")

                bid, ask = st.columns(2)

                bid.metric(
                    "Giá mua tốt nhất",
                    show_price("Mua 1", quote["bid_price_1"])
                )

                bid.caption(
                    f"Khối lượng mua: "
                    f"{show_price('KL', quote['bid_vol_1'])}"
                )

                ask.metric(
                    "Giá bán tốt nhất",
                    show_price("Bán 1", quote["ask_price_1"])
                )

                ask.caption(
                    f"Khối lượng bán: "
                    f"{show_price('KL', quote['ask_vol_1'])}"
                )

                st.caption(
                    "Đây là dữ liệu bảng giá tại thời điểm truy vấn, "
                    "không phải luồng giá đẩy liên tục."
                )

            except Exception as quote_error:

                st.warning(
                    f"Chưa lấy được bảng giá trong phiên: {quote_error}"
                )

                st.caption(
                    "Phần phân tích kỹ thuật vẫn sử dụng dữ liệu lịch sử."
                )

            # =====================================
            # TÍN HIỆU
            # =====================================

            st.divider()

            signal = result["signal"]

            if signal == "BUY":

                st.success("🟢 TÍN HIỆU BUY")

            elif signal == "WAIT":

                st.warning("🟡 TÍN HIỆU WAIT")

            else:

                st.error("🔴 TÍN HIỆU REJECT")

            st.write(result["message"])

            # =====================================
            # CÁC CHỈ SỐ CHÍNH
            # =====================================

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Điểm kỹ thuật",
                f'{result["score"]}/100'
            )

            col2.metric(
                "Giá đóng cửa",
                f'{result["price"]:,.0f} đ'
            )

            col3.metric(
                "RSI",
                f'{result["rsi"]:.2f}'
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Giá cắt lỗ",
                f'{result["stop_loss"]:,.0f} đ'
            )

            col5.metric(
                "Giá mục tiêu",
                f'{result["target_price"]:,.0f} đ'
            )

            col6.metric(
                "Tỷ lệ lợi nhuận/rủi ro",
                f'{result["reward_risk"]:.2f}:1'
            )

            # =====================================
            # QUẢN TRỊ VỐN
            # =====================================

            st.subheader("💰 Quản trị vốn")

            col7, col8, col9 = st.columns(3)

            col7.metric(
                "Số lượng dự kiến",
                f'{result["quantity"]:,} cổ phiếu'
            )

            col8.metric(
                "Vốn sử dụng dự kiến",
                f'{result["capital_used"]:,.0f} đ'
            )

            col9.metric(
                "Rủi ro dự kiến",
                f'{result["risk_money"]:,.0f} đ'
            )

            st.caption(
                "Rủi ro là mức ước tính khi giá khớp tại mức cắt lỗ. "
                "Thực tế có thể cao hơn do biến động giá, "
                "trượt giá và thanh khoản."
            )

            # =====================================
            # LÝ DO CHẤM ĐIỂM
            # =====================================

            st.subheader("📊 Các yếu tố kỹ thuật")

            if result["reasons"]:

                for reason in result["reasons"]:
                    st.write(f"• {reason}")

            else:

                st.write("Chưa có yếu tố kỹ thuật tích cực.")

            st.write(
                f'**Khối lượng so với trung bình 20 phiên:** '
                f'{result["volume_ratio"]:.2f} lần'
            )

            st.write(
                f'**Ngày dữ liệu cuối:** {result["data_date"]}'
            )

            # =====================================
            # DỮ LIỆU GỐC
            # =====================================

            with st.expander("Xem dữ liệu lịch sử"):

                st.dataframe(
                    df.tail(20),
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Không thể phân tích {symbol}: {error}"
            )

            st.caption(
                "Hãy kiểm tra mã cổ phiếu, kết nối dữ liệu "
                "hoặc thử lại sau."
            )


# =====================================
# LÀM MỚI DỮ LIỆU
# =====================================

st.divider()

if st.button("🔄 Làm mới dữ liệu"):

    st.cache_data.clear()

    st.success(
        "Đã xóa bộ nhớ đệm. "
        "Hãy nhập mã cổ phiếu và phân tích lại."
    )


# =====================================
# CẢNH BÁO
# =====================================

st.divider()

st.caption(
    "AI STOCK SCANNER chỉ cung cấp tín hiệu tham khảo "
    "dựa trên dữ liệu lịch sử và chỉ báo kỹ thuật. "
    "Không đảm bảo lợi nhuận, không thay thế quyết định "
    "đầu tư của người sử dụng."
)
