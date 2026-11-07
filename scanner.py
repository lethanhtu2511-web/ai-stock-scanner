import pandas as pd
import numpy as np

# ==============================
# CẤU HÌNH CHIẾN LƯỢC
# ==============================

INITIAL_CAPITAL = 10_000_000
RISK_PERCENT = 0.02
MIN_SCORE = 85
MIN_REWARD_RISK = 2.0
LOT_SIZE = 100

# Chi phí dự kiến (có thể thay đổi theo công ty chứng khoán)
BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002


# ==============================
# TÍNH CÁC CHỈ BÁO KỸ THUẬT
# ==============================

def calculate_indicators(df):

    required = [
        "Date", "Open", "High",
        "Low", "Close", "Volume"
    ]

    if df is None or df.empty:
        raise ValueError("Không có dữ liệu cổ phiếu.")

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Thiếu cột dữ liệu: " + ", ".join(missing)
        )

    df = df[required].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()
    df = df.sort_values("Date")
    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    ).reset_index(drop=True)

    if len(df) < 200:
        raise ValueError("Cần tối thiểu 200 phiên dữ liệu.")

    if (df["Close"] <= 0).any():
        raise ValueError("Giá cổ phiếu không hợp lệ.")

    # Đường trung bình
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # RSI 14
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # Xử lý trường hợp giá chỉ tăng hoặc đi ngang
    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50

    # MACD
    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Khối lượng giao dịch
    df["VOLUME_MA20"] = df["Volume"].rolling(20).mean()

    df["VOLUME_RATIO"] = (
        df["Volume"] /
        df["VOLUME_MA20"].replace(0, np.nan)
    )

    # ATR 14
    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - previous_close).abs()
    tr3 = (df["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    # Đỉnh cao nhất của 20 phiên trước
    df["HIGH20_PREVIOUS"] = (
        df["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("Không đủ dữ liệu để tính chỉ báo.")

    return df


# ==============================
# CHẤM ĐIỂM CỔ PHIẾU
# ==============================

def calculate_score(row):

    score = 0
    reasons = []

    # Xu hướng giá
    if row["Close"] > row["MA20"]:
        score += 10
        reasons.append("Giá nằm trên MA20.")

    if row["MA20"] > row["MA50"]:
        score += 10
        reasons.append("MA20 nằm trên MA50.")

    if row["MA50"] > row["MA200"]:
        score += 10
        reasons.append("Xu hướng trung hạn tích cực.")

    # RSI
    if 50 <= row["RSI"] <= 65:
        score += 15
        reasons.append("RSI ở vùng tích cực.")

    elif 65 < row["RSI"] <= 70:
        score += 10
        reasons.append("RSI mạnh nhưng cần thận trọng.")

    # MACD
    if row["MACD"] > row["MACD_SIGNAL"]:
        score += 15
        reasons.append("MACD đang tích cực.")

    # Khối lượng
    if row["VOLUME_RATIO"] >= 1.5:
        score += 15
        reasons.append("Khối lượng tăng mạnh.")

    elif row["VOLUME_RATIO"] >= 1.2:
        score += 10
        reasons.append("Khối lượng cao hơn trung bình.")

    # Breakout hoặc điều chỉnh về MA20
    if row["Close"] > row["HIGH20_PREVIOUS"]:
        score += 15
        reasons.append("Giá vượt đỉnh 20 phiên.")

    else:
        distance = abs(
            row["Close"] - row["MA20"]
        ) / row["MA20"]

        if distance <= 0.03 and row["Close"] >= row["MA20"]:
            score += 10
            reasons.append("Giá đang ở gần MA20.")

    return score, reasons


# ==============================
# PHÂN TÍCH VÀ QUẢN TRỊ RỦI RO
# ==============================

def analyze_stock(
    df,
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT
):

    result = {
        "signal": "REJECT",
        "score": 0,
        "price": 0,
        "stop_loss": 0,
        "target_price": 0,
        "quantity": 0,
        "capital_used": 0,
        "risk_money": 0,
        "reward_risk": 0,
        "rsi": 0,
        "volume_ratio": 0,
        "reasons": [],
        "data_date": None,
        "message": ""
    }

    try:

        capital = float(capital)
        risk_percent = float(risk_percent)

        if capital <= 0:
            raise ValueError("Vốn đầu tư phải lớn hơn 0.")

        if not 0 < risk_percent <= 0.02:
            raise ValueError(
                "Mức rủi ro phải lớn hơn 0 và không vượt 2%."
            )

        df = calculate_indicators(df)

        row = df.iloc[-1]

        # Kiểm tra độ mới của dữ liệu
        data_date = pd.Timestamp(row["Date"]).date()
        today = pd.Timestamp.now().date()

        age_days = (today - data_date).days

        result["data_date"] = str(data_date)

        if age_days < 0:
            raise ValueError("Ngày dữ liệu không hợp lệ.")

        if age_days > 10:
            raise ValueError(
                "Dữ liệu quá cũ. Không phát tín hiệu mua."
            )

        # Chấm điểm
        score, reasons = calculate_score(row)

        price = float(row["Close"])
        atr = float(row["ATR"])
        rsi = float(row["RSI"])
        volume_ratio = float(row["VOLUME_RATIO"])

        if not all(np.isfinite(x) for x in [
            price, atr, rsi, volume_ratio
        ]):
            raise ValueError("Chỉ báo có giá trị không hợp lệ.")

        if atr <= 0:
            raise ValueError("ATR không hợp lệ.")

        # Giá cắt lỗ
        stop_loss = price - 1.5 * atr

        if stop_loss <= 0 or stop_loss >= price:
            raise ValueError("Không xác định được mức cắt lỗ.")

        # Khoản lỗ dự kiến trên mỗi cổ phiếu:
        # Giá mua + phí mua + trượt giá mua
        # trừ giá bán tại stop sau trượt giá,
        # phí bán và thuế bán.
        estimated_buy_cost = (
            price * (1 + BUY_FEE + SLIPPAGE)
        )

        estimated_stop_proceeds = (
            stop_loss
            * (1 - SLIPPAGE)
            * (1 - SELL_FEE - SELL_TAX)
        )

        risk_per_share = (
            estimated_buy_cost -
            estimated_stop_proceeds
        )

        if risk_per_share <= 0:
            raise ValueError("Không tính được rủi ro mỗi cổ phiếu.")

        # Ngân sách rủi ro
        risk_budget = capital * risk_percent

        # Giới hạn số lượng theo rủi ro
        quantity_by_risk = int(
            risk_budget // risk_per_share
        )

        # Giới hạn số lượng theo vốn, bao gồm phí mua
        quantity_by_capital = int(
            capital // estimated_buy_cost
        )

        quantity = min(
            quantity_by_risk,
            quantity_by_capital
        )

        # Làm tròn xuống lô 100 cổ phiếu
        quantity = (
            quantity // LOT_SIZE
        ) * LOT_SIZE

        # Giá mục tiêu theo tỷ lệ 2:1
        target_price = price + (
            price - stop_loss
        ) * MIN_REWARD_RISK

        # Tính lợi nhuận dự kiến sau chi phí
        estimated_target_proceeds = (
            target_price
            * (1 - SLIPPAGE)
            * (1 - SELL_FEE - SELL_TAX)
        )

        estimated_reward_per_share = (
            estimated_target_proceeds -
            estimated_buy_cost
        )

        reward_risk = (
            estimated_reward_per_share /
            risk_per_share
        )

        capital_used = (
            quantity * estimated_buy_cost
        )

        actual_risk = (
            quantity * risk_per_share
        )

        # Cập nhật kết quả
        result.update({
            "score": int(score),
            "price": round(price, 2),
            "stop_loss": round(stop_loss, 2),
            "target_price": round(target_price, 2),
            "quantity": int(quantity),
            "capital_used": round(capital_used),
            "risk_money": round(actual_risk),
            "reward_risk": round(reward_risk, 2),
            "rsi": round(rsi, 2),
            "volume_ratio": round(volume_ratio, 2),
            "reasons": reasons
        })

        # Điều kiện từ chối
        if rsi > 75:
            result["signal"] = "WAIT"
            result["message"] = (
                "RSI quá cao, cổ phiếu có thể đã tăng nóng."
            )
            return result

        if quantity < LOT_SIZE:
            result["signal"] = "WAIT"
            result["message"] = (
                "Vốn hoặc mức rủi ro không đủ để mua một lô."
            )
            return result

        if score < MIN_SCORE:
            result["signal"] = "WAIT"
            result["message"] = (
                f"Điểm chỉ đạt {score}/100, "
                f"chưa đủ ngưỡng {MIN_SCORE}."
            )
            return result

        if price <= row["MA20"]:
            result["signal"] = "WAIT"
            result["message"] = (
                "Giá chưa nằm trên MA20."
            )
            return result

        if reward_risk < MIN_REWARD_RISK:
            result["signal"] = "WAIT"
            result["message"] = (
                "Tỷ lệ lợi nhuận/rủi ro sau chi phí chưa đạt 2:1."
            )
            return result

        if actual_risk > risk_budget:
            result["signal"] = "WAIT"
            result["message"] = (
                "Rủi ro dự kiến vượt ngân sách cho phép."
            )
            return result

        # Đạt toàn bộ điều kiện
        result["signal"] = "BUY"
        result["message"] = (
            "Đạt các điều kiện kỹ thuật và quản trị rủi ro. "
            "Đây là tín hiệu tham khảo, không đảm bảo lợi nhuận."
        )

        return result

    except Exception as error:

        result["signal"] = "REJECT"
        result["message"] = str(error)

        return result
