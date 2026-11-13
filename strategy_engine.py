
import numpy as np
import pandas as pd


STRATEGIES = {
    "SWING": "LƯỚT SÓNG 2–5 PHIÊN",
    "REVERSAL": "ĐẢO CHIỀU 5–10 PHIÊN",
    "PULLBACK": "MUA KHI ĐIỀU CHỈNH 3–7 PHIÊN",
}


def _prepare(df):
    if df is None or len(df) < 60:
        raise ValueError("Cần tối thiểu 60 phiên dữ liệu D1.")

    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]

    required = ["open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Thiếu dữ liệu: {', '.join(missing)}")

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required).copy()
    data = data.sort_index().reset_index(drop=True)

    if len(data) < 60:
        raise ValueError("Không đủ dữ liệu D1 hợp lệ.")

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    data["ema20"] = close.ewm(span=20, adjust=False).mean()
    data["ema50"] = close.ewm(span=50, adjust=False).mean()
    data["ma200"] = close.rolling(200).mean()

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
    data["rsi"] = 100 - (100 / (1 + rs))
    data["rsi"] = data["rsi"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(
        span=9, adjust=False
    ).mean()

    data["vol_ma20"] = volume.rolling(20).mean()

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(
        typical_price > typical_price.shift(1), 0
    ).rolling(14).sum()

    negative_flow = money_flow.where(
        typical_price < typical_price.shift(1), 0
    ).rolling(14).sum()

    money_ratio = positive_flow / negative_flow.replace(0, np.nan)
    data["mfi"] = 100 - (100 / (1 + money_ratio))
    data["mfi"] = data["mfi"].fillna(50)

    data["obv"] = (
        np.sign(close.diff()).fillna(0) * volume
    ).cumsum()

    data["obv_ma10"] = data["obv"].rolling(10).mean()

    previous_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)

    data["atr"] = true_range.rolling(14).mean()

    return data


def _relative_strength(stock_df, market_df=None):
    if market_df is None:
        return None

    try:
        market = market_df.copy()
        market.columns = [str(c).lower() for c in market.columns]

        if "close" not in market.columns or len(market) < 21:
            return None

        stock_return = (
            stock_df["close"].iloc[-1]
            / stock_df["close"].iloc[-21] - 1
        )

        market_close = pd.to_numeric(
            market["close"], errors="coerce"
        ).dropna()

        market_return = (
            market_close.iloc[-1]
            / market_close.iloc[-21] - 1
        )

        return float(stock_return - market_return)

    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _base_metrics(data):
    last = data.iloc[-1]
    previous = data.iloc[-2]

    close = float(last["close"])
    atr = float(last["atr"]) if pd.notna(last["atr"]) else 0

    recent_high = float(data["high"].iloc[-21:-1].max())
    recent_low = float(data["low"].iloc[-21:-1].min())

    average_volume = float(last["vol_ma20"]) if pd.notna(
        last["vol_ma20"]
    ) else 0

    volume_ratio = (
        float(last["volume"]) / average_volume
        if average_volume > 0 else 0
    )

    return {
        "close": close,
        "previous_close": float(previous["close"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "ma200": (
            float(last["ma200"])
            if pd.notna(last["ma200"]) else None
        ),
        "rsi": float(last["rsi"]),
        "macd": float(last["macd"]),
        "macd_signal": float(last["macd_signal"]),
        "mfi": float(last["mfi"]),
        "volume_ratio": volume_ratio,
        "atr": atr,
        "support": recent_low,
        "resistance": recent_high,
        "obv_rising": (
            float(last["obv"]) > float(last["obv_ma10"])
            if pd.notna(last["obv_ma10"]) else False
        ),
    }


def _score_swing(m):
    score = 0
    reasons = []

    if m["close"] > m["ema20"] > m["ema50"]:
        score += 20
        reasons.append("Giá trên EMA20 và EMA50")

    if m["rsi"] >= 50 and m["rsi"] <= 70:
        score += 15
        reasons.append("RSI xác nhận xu hướng")

    if m["macd"] > m["macd_signal"]:
        score += 15
        reasons.append("MACD tích cực")

    if m["volume_ratio"] >= 1.3:
        score += 15
        reasons.append("Thanh khoản tăng")

    if m["obv_rising"]:
        score += 10
        reasons.append("OBV cải thiện")

    if m["close"] > m["resistance"]:
        score += 15
        reasons.append("Vượt kháng cự 20 phiên")

    if m["mfi"] > 50:
        score += 10
        reasons.append("Dòng tiền MFI tích cực")

    return min(score, 100), reasons


def _score_reversal(m, data):
    score = 0
    reasons = []

    recent_rsi = data["rsi"].iloc[-5:]

    if m["rsi"] < 40 and m["rsi"] > recent_rsi.iloc[0]:
        score += 20
        reasons.append("RSI phục hồi từ vùng yếu")

    if m["close"] > m["previous_close"]:
        score += 10
        reasons.append("Giá đóng cửa tăng")

    if m["macd"] > m["macd_signal"]:
        score += 20
        reasons.append("MACD xác nhận cải thiện")

    if m["volume_ratio"] >= 1.2:
        score += 15
        reasons.append("Thanh khoản xác nhận")

    if m["obv_rising"]:
        score += 10
        reasons.append("OBV phục hồi")

    if m["close"] > m["ema20"]:
        score += 15
        reasons.append("Giá lấy lại EMA20")

    if m["mfi"] > 40:
        score += 10
        reasons.append("MFI cải thiện")

    return min(score, 100), reasons


def _score_pullback(m):
    score = 0
    reasons = []

    if m["ema20"] > m["ema50"]:
        score += 20
        reasons.append("Xu hướng trung hạn tích cực")

    distance = abs(m["close"] - m["ema20"]) / m["ema20"]

    if distance <= 0.025:
        score += 20
        reasons.append("Giá đang gần EMA20")

    if m["close"] >= m["ema20"] * 0.98:
        score += 10
        reasons.append("Giá giữ vùng hỗ trợ EMA20")

    if 40 <= m["rsi"] <= 60:
        score += 15
        reasons.append("RSI phù hợp nhịp điều chỉnh")

    if m["macd"] > m["macd_signal"]:
        score += 10
        reasons.append("MACD duy trì tích cực")

    if m["obv_rising"]:
        score += 10
        reasons.append("OBV chưa suy yếu")

    if m["volume_ratio"] <= 1.2:
        score += 10
        reasons.append("Khối lượng điều chỉnh không quá lớn")

    if m["close"] > m["ema50"]:
        score += 5
        reasons.append("Giá vẫn trên EMA50")

    return min(score, 100), reasons


def analyze_stock(symbol, df, market_df=None):
    """
    Phân tích một cổ phiếu bằng dữ liệu D1.

    Trả về điểm số và tín hiệu riêng cho từng chiến lược.
    """

    data = _prepare(df)
    m = _base_metrics(data)

    relative_strength = _relative_strength(data, market_df)

    if relative_strength is not None:
        if relative_strength > 0.03:
            rs_bonus = 5
        elif relative_strength < -0.03:
            rs_bonus = -5
        else:
            rs_bonus = 0
    else:
        rs_bonus = 0

    stop_loss = (
        m["close"] - 1.5 * m["atr"]
        if m["atr"] > 0 else None
    )

    target_price = (
        m["close"] + 3 * m["atr"]
        if m["atr"] > 0 else None
    )

    result = {
        "symbol": str(symbol).upper(),
        "price": m["close"],
        "support": m["support"],
        "resistance": m["resistance"],
        "rsi": m["rsi"],
        "macd": m["macd"],
        "mfi": m["mfi"],
        "volume_ratio": m["volume_ratio"],
        "atr": m["atr"],
        "stop_loss": stop_loss,
        "target_price": target_price,
        "relative_strength": relative_strength,
        "strategies": {},
    }

    scorers = {
        "SWING": lambda: _score_swing(m),
        "REVERSAL": lambda: _score_reversal(m, data),
        "PULLBACK": lambda: _score_pullback(m),
    }

    for strategy, scorer in scorers.items():
        score, reasons = scorer()
        score = max(0, min(100, score + rs_bonus))

        if score >= 75:
            signal = "MUA"
        elif score >= 60:
            signal = "CHỜ MUA"
        elif score >= 40:
            signal = "THEO DÕI"
        else:
            signal = "BỎ QUA"

        result["strategies"][strategy] = {
            "name": STRATEGIES[strategy],
            "score": score,
            "signal": signal,
            "reasons": reasons,
        }

    best_strategy = max(
        result["strategies"].items(),
        key=lambda item: item[1]["score"]
    )

    result["best_strategy"] = best_strategy[0]
    result["best_score"] = best_strategy[1]["score"]
    result["best_signal"] = best_strategy[1]["signal"]

    return result


def analyze_market_data(symbol, df, market_df=None):
    """
    Hàm tương thích tên gọi khác nếu app cần sử dụng.
    """

    return analyze_stock(symbol, df, market_df)
