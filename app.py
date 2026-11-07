
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Stock Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI STOCK SCANNER")
st.caption("Vietnam Stock Market | Swing Trading")

st.info(
    "Phiên bản 1.0: Bộ máy phân tích kỹ thuật. "
    "Dữ liệu realtime sẽ được kết nối ở bước tiếp theo."
)

st.sidebar.header("⚙️ Cấu hình")
st.sidebar.write("Chiến lược: Lướt sóng")
st.sidebar.write("Khung phân tích: 5 phút / 1 ngày")
st.sidebar.write("Thị trường: HOSE, HNX, UPCOM")

st.header("📊 Phân tích cổ phiếu")

uploaded_file = st.file_uploader(
    "Tải dữ liệu cổ phiếu CSV lên để kiểm tra",
    type=["csv"]
)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def analyze_stock(df):
    df = df.copy()

    required = ["Close", "Volume"]
    if not all(col in df.columns for col in required):
        raise ValueError("Thiếu cột Close hoặc Volume")

    for col in ["Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close", "Volume"])
    df = df.sort_values("Date") if "Date" in df.columns else df

    if len(df) < 50:
        raise ValueError("Cần ít nhất 50 phiên dữ liệu")

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"])

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    score = 0

    if last["Close"] > last["MA20"]:
        score += 10
    if last["MA20"] > last["MA50"]:
        score += 10
    if 50 <= last["RSI"] <= 70:
        score += 15
    if last["MACD"] > last["Signal"]:
        score += 15
    if last["Volume"] > last["Volume_MA20"]:
        score += 20

    if len(df) >= 21:
        previous_high = df["Close"].iloc[-21:-1].max()
        if last["Close"] > previous_high:
            score += 20

    if last["RSI"] > 75:
        score -= 10

    score = max(0, min(100, score))

    return df, score


if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(
                data["Date"], errors="coerce"
            )
            data = data.dropna(subset=["Date"])

        result, score = analyze_stock(data)
        latest = result.iloc[-1]

        col1, col2, col3 = st.columns(3)

        col1.metric("Giá đóng cửa", f"{latest['Close']:,.2f}")
        col2.metric("Điểm Scanner", f"{score}/100")
        col3.metric("RSI", f"{latest['RSI']:.1f}")

        st.subheader("📈 Đánh giá tín hiệu")

        if score >= 75:
            st.success("CỔ PHIẾU ĐÁNG THEO DÕI")
        elif score >= 60:
            st.warning("CHỜ TÍN HIỆU XÁC NHẬN")
        else:
            st.error("CHƯA ƯU TIÊN")

        st.line_chart(
            result.set_index("Date")[["Close", "MA20", "MA50"]]
            if "Date" in result.columns
            else result[["Close", "MA20", "MA50"]]
        )

        st.subheader("Thông số kỹ thuật")
        st.dataframe(result.tail(10), use_container_width=True)

    except Exception as e:
        st.error(f"Không thể phân tích dữ liệu: {e}")

else:
    st.warning(
        "Chưa có dữ liệu cổ phiếu. "
        "Hệ thống chưa đưa ra khuyến nghị mua bán."
    )
    st.subheader("Các chức năng đang xây dựng")
    st.write("🔄 Kết nối dữ liệu trực tuyến")
    st.write("🔍 Quét cổ phiếu toàn thị trường")
    st.write("🏆 Xếp hạng cổ phiếu theo điểm")
    st.write("🎯 Xác định vùng mua, chốt lời và cắt lỗ")
    st.write("📱 Giao diện tối ưu cho điện thoại")
