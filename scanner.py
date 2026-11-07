
import pandas as pd
import numpy as np


# ==========================================
# AI STOCK SCANNER - TECHNICAL ENGINE
# Strategy: Swing trading 2-5 sessions
# Markets: HOSE / HNX
# ==========================================

INITIAL_CAPITAL = 10_000_000
RISK_PERCENT = 0.02
MIN_SCORE = 85
MIN_REWARD_RISK = 2.0
LOT_SIZE = 100


def calculate_indicators(df):
    """Calculate technical indicators from daily OHLCV data."""

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]

    if not all(col in df.columns for col in required):
        raise ValueError(
            "Dữ liệu cần có: Date, Open, High, Low, Close, Volume"
        )

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required)
    data = data.sort_values("Date").drop_duplicates("Date")

    if len(data) < 200:
        raise ValueError(
            "Cần ít nhất 200 phiên dữ liệu để phân tích đáng tin cậy."
        )

    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Dữ liệu giá không hợp lệ.")

    if (data["Volume"] < 0).any():
        raise ValueError("Khối lượng giao dịch không hợp lệ.")

    close = data["Close"]

    # Moving averages
    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()
    data["MA200"] = close.rolling(200).mean()

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    data.loc[
        (avg_loss == 0) & (avg_gain > 0), "RSI"
    ] = 100

    data.loc[
        (avg_loss == 0) & (avg_gain == 0), "RSI"
    ] = 50

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    # Volume
    data["VOL_MA20"] = data["Volume"].rolling(20).mean()

    # Average True Range
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = tr.rolling(14).mean()

    # Previous 20-session high, excluding current session
    data["PREV_HIGH20"] = (
        data["High"].shift(1).rolling(20).max()
    )

    return data.dropna()


def calculate_score(row):
    """Score a stock from 0 to 100."""

    score = 0
    reasons = []

    # Trend
    if row["Close"] > row["MA20"]:
        score += 10
        reasons.append("Giá trên MA20")

    if row["MA20"] > row["MA50"]:
        score += 10
        reasons.append("MA20 trên MA50")

    if row["MA50"] > row["MA200"]:
        score += 10
        reasons.append("Xu hướng trung hạn tích cực")

    # RSI
    if 50 <= row["RSI"] <= 65:
        score += 15
        reasons.append("RSI tích cực")

    elif 65 < row["RSI"] <= 70:
        score += 10
        reasons.append("RSI mạnh")

    # MACD
    if row["MACD"] > row["MACD_SIGNAL"]:
        score += 15
        reasons.append("MACD xác nhận xu hướng tăng")

    # Volume
    if row["VOL_MA20"] > 0:
        volume_ratio = row["Volume"] / row["VOL_MA20"]

        if volume_ratio >= 1.5:
            score += 15
            reasons.append("Khối lượng tăng mạnh")

        elif volume_ratio >= 1.2:
            score += 10
            reasons.append("Khối lượng cải thiện")

    # Breakout
    if row["Close"] > row["PREV_HIGH20"]:
        score += 15
        reasons.append("Vượt đỉnh 20 phiên")

    # Pullback near MA20
    elif (
        row["Close"] >= row["MA20"]
        and row["Close"] <= row["MA20"] * 1.03
    ):
        score += 10
        reasons.append("Giá điều chỉnh gần MA20")

    return min(score, 100), reasons


def analyze_stock(
    df,
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT
):
    """
    Analyze one stock.
    Returns WAIT/BUY/REJECT and a trading plan.
    Input must be daily OHLCV data.
    """

    try:
        if capital <= 0 or not 0 < risk_percent <= 0.02:
            return {
                "signal": "REJECT",
                "reason": "Vốn hoặc mức rủi ro không hợp lệ."
            }

        data = calculate_indicators(df)
        row = data.iloc[-1]

        latest_date = row["Date"].date()
        today = pd.Timestamp.now().date()

        # Reject stale or future data
        age = (today - latest_date).days

        if age < 0 or age > 7:
            return {
                "signal": "REJECT",
                "reason": "Dữ liệu quá cũ hoặc ngày dữ liệu không hợp lệ.",
                "data_date": str(latest_date)
            }

        score, reasons = calculate_score(row)

        price = float(row["Close"])
        atr = float(row["ATR"])

        if not np.isfinite(atr) or atr <= 0:
            return {
                "signal": "REJECT",
                "reason": "Không tính được ATR hợp lệ."
            }

        # Initial stop loss
        stop_loss = price - 1.5 * atr

        # Position sizing
        risk_money = capital * risk_percent
        risk_per_share = price - stop_loss

        if risk_per_share <= 0:
            return {
                "signal": "REJECT",
                "reason": "Không xác định được mức cắt lỗ."
            }

        quantity_by_risk = int(
            risk_money // risk_per_share
        )

        quantity_by_capital = int(
            capital // price
        )

        # Vietnamese stock lot size
        quantity = min(
            quantity_by_risk,
            quantity_by_capital
        )

        quantity = (quantity // LOT_SIZE) * LOT_SIZE

        target_price = price + (
            risk_per_share * MIN_REWARD_RISK
        )

        actual_rr = (
            (target_price - price) / risk_per_share
        )

        # Avoid chasing overheated prices
        if row["RSI"] > 75:
            return {
                "signal": "WAIT",
                "score": score,
                "reason": "RSI quá cao, tránh mua đuổi.",
                "data_date": str(latest_date)
            }

        # Buy only when all conditions are satisfied
        if (
            score >= MIN_SCORE
            and actual_rr >= MIN_REWARD_RISK
            and quantity >= LOT_SIZE
            and price > row["MA20"]
        ):
            signal = "BUY"
            message = "Đủ điều kiện kỹ thuật ban đầu."

        else:
            signal = "WAIT"
            message = "Chưa đủ điều kiện mua."

        return {
            "signal": signal,
            "score": int(score),
            "price": round(price, 2),
            "stop_loss": round(stop_loss, 2),
            "target_price": round(target_price, 2),
            "quantity": quantity,
            "capital_used": round(quantity * price, 0),
            "risk_money": round(
                quantity * risk_per_share, 0
            ),
            "reward_risk": round(actual_rr, 2),
            "rsi": round(float(row["RSI"]), 2),
            "volume_ratio": round(
                float(row["Volume"] / row["VOL_MA20"]), 2
            ),
            "reasons": reasons,
            "data_date": str(latest_date),
            "message": message
        }

    except Exception as error:
        return {
            "signal": "REJECT",
            "reason": str(error)
        }
