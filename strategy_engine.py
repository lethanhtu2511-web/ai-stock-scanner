
import numpy as np
import pandas as pd


STRATEGIES = {
    "SWING": "LƯỚT SÓNG 2–5 PHIÊN",
    "REVERSAL": "ĐẢO CHIỀU 5–10 PHIÊN",
    "PULLBACK": "MUA KHI ĐIỀU CHỈNH 3–7 PHIÊN",
}

MIN_HISTORY = 200
MIN_AVG_TURNOVER = 5_000_000_000
MIN_MEDIAN_TURNOVER = 3_000_000_000

BUY_SCORE = 75
WATCH_SCORE = 60


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _prepare(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dữ liệu D1 không hợp lệ.")

    data = df.copy()
    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    required = [
        "open", "high", "low",
        "close", "volume"
    ]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu dữ liệu: {', '.join(missing)}"
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if "date" in data.columns:
        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )
        data = data.dropna(subset=["date"])
        data = data.sort_values("date")

    data = (
        data.dropna(subset=required)
        .drop_duplicates(
            subset=["date"] if "date" in data.columns else None,
            keep="last"
        )
        .reset_index(drop=True)
    )

    valid = (
        (data["open"] > 0)
        & (data["high"] > 0)
        & (data["low"] > 0)
        & (data["close"] > 0)
        & (data["volume"] >= 0)
        & (data["high"] >= data[["open", "close", "low"]].max(axis=1))
        & (data["low"] <= data[["open", "close"]].min(axis=1))
    )

    data = data.loc[valid].reset_index(drop=True)

    if len(data) < MIN_HISTORY:
        raise ValueError(
            f"Cần tối thiểu {MIN_HISTORY} phiên D1 hợp lệ; "
            f"hiện có {len(data)} phiên."
        )

    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]

    # Xu hướng
    data["ema20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    data["ema50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    data["ma200"] = close.rolling(200).mean()

    # RSI Wilder
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["rsi"] = 100 - 100 / (1 + rs)

    data.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi"
    ] = 100

    data.loc[
        (avg_gain == 0) & (avg_loss > 0),
        "rsi"
    ] = 0

    data["rsi"] = data["rsi"].fillna(50)

    # MACD
    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    data["macd"] = ema12 - ema26

    data["macd_signal"] = data["macd"].ewm(
        span=9, adjust=False
    ).mean()

    data["macd_hist"] = (
        data["macd"] - data["macd_signal"]
    )

    data["macd_hist_prev"] = data["macd_hist"].shift(1)

    # Khối lượng: so sánh với 20 phiên trước,
    # không đưa phiên hiện tại vào đường trung bình.
    data["volume_avg20"] = volume.shift(1).rolling(20).mean()

    data["volume_ratio"] = (
        volume / data["volume_avg20"].replace(0, np.nan)
    )

    data["volume_avg5_prev"] = volume.shift(1).rolling(5).mean()

    data["volume_dryup"] = (
        data["volume_avg5_prev"]
        / data["volume_avg20"].replace(0, np.nan)
    )

    # Giá trị giao dịch
    data["turnover"] = close * volume

    data["avg_turnover20"] = (
        data["turnover"].shift(1).rolling(20).mean()
    )

    data["median_turnover20"] = (
        data["turnover"].shift(1).rolling(20).median()
    )

    # ATR Wilder
    previous_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    data["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14
    ).mean()

    data["atr_pct"] = (
        data["atr"] / close * 100
    )

    # MFI
    typical_price = (high + low + close) / 3
    raw_flow = typical_price * volume

    direction = typical_price.diff()

    positive_flow = raw_flow.where(
        direction > 0, 0
    ).rolling(14).sum()

    negative_flow = raw_flow.where(
        direction < 0, 0
    ).rolling(14).sum()

    money_ratio = (
        positive_flow
        / negative_flow.replace(0, np.nan)
    )

    data["mfi"] = 100 - 100 / (1 + money_ratio)

    data.loc[
        (negative_flow == 0) & (positive_flow > 0),
        "mfi"
    ] = 100

    data["mfi"] = data["mfi"].fillna(50)

    # OBV
    direction_close = np.sign(close.diff()).fillna(0)

    data["obv"] = (
        direction_close * volume
    ).cumsum()

    data["obv_avg10"] = data["obv"].rolling(10).mean()

    # Kháng cự / hỗ trợ quá khứ, không bao gồm phiên hiện tại
    data["high20_prev"] = high.shift(1).rolling(20).max()
    data["high50_prev"] = high.shift(1).rolling(50).max()
    data["low20_prev"] = low.shift(1).rolling(20).min()

    data["low10_prev"] = low.shift(1).rolling(10).min()

    data["high20_current"] = high.rolling(20).max()
    data["low20_current"] = low.rolling(20).min()

    # Độ dốc EMA20
    data["ema20_prev5"] = data["ema20"].shift(5)

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data.reset_index(drop=True)


def _relative_strength(stock_df, market_df=None):
    if market_df is None:
        return None

    try:
        market = market_df.copy()

        market.columns = [
            str(column).strip().lower()
            for column in market.columns
        ]

        if "close" not in market.columns:
            return None

        market_close = pd.to_numeric(
            market["close"],
            errors="coerce"
        ).dropna()

        stock_close = stock_df["close"].dropna()

        if len(market_close) < 21 or len(stock_close) < 21:
            return None

        stock_return = (
            stock_close.iloc[-1]
            / stock_close.iloc[-21] - 1
        )

        market_return = (
            market_close.iloc[-1]
            / market_close.iloc[-21] - 1
        )

        return float(stock_return - market_return)

    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _base_metrics(data):
    last = data.iloc[-1]

    close = _safe_float(last["close"])
    atr = _safe_float(last["atr"])

    ema20 = _safe_float(last["ema20"])
    ema50 = _safe_float(last["ema50"])
    ma200 = _safe_float(last["ma200"])

    volume_ratio = _safe_float(last["volume_ratio"])
    avg_turnover = _safe_float(last["avg_turnover20"])
    median_turnover = _safe_float(last["median_turnover20"])

    high20 = _safe_float(last["high20_prev"])
    high50 = _safe_float(last["high50_prev"])
    low20 = _safe_float(last["low20_prev"])
    low10 = _safe_float(last["low10_prev"])

    candle_range = max(
        _safe_float(last["high"]) - _safe_float(last["low"]),
        1e-9
    )

    close_position = (
        close - _safe_float(last["low"])
    ) / candle_range

    ema20_slope = (
        ema20 / _safe_float(last["ema20_prev5"]) - 1
        if _safe_float(last["ema20_prev5"]) > 0
        else 0
    )

    return {
        "close": close,
        "open": _safe_float(last["open"]),
        "high": _safe_float(last["high"]),
        "low": _safe_float(last["low"]),
        "ema20": ema20,
        "ema50": ema50,
        "ma200": ma200,
        "rsi": _safe_float(last["rsi"], 50),
        "macd": _safe_float(last["macd"]),
        "macd_signal": _safe_float(last["macd_signal"]),
        "macd_hist": _safe_float(last["macd_hist"]),
        "macd_hist_prev": _safe_float(last["macd_hist_prev"]),
        "mfi": _safe_float(last["mfi"], 50),
        "volume_ratio": volume_ratio,
        "volume_dryup": _safe_float(last["volume_dryup"]),
        "atr": atr,
        "atr_pct": _safe_float(last["atr_pct"]),
        "avg_turnover20": avg_turnover,
        "median_turnover20": median_turnover,
        "high20": high20,
        "high50": high50,
        "low20": low20,
        "low10": low10,
        "close_position": close_position,
        "ema20_slope": ema20_slope,
        "obv_rising": (
            _safe_float(last["obv"])
            > _safe_float(last["obv_avg10"])
        ),
        "breakout20": close > high20 > 0,
        "above_ema20": close > ema20 > 0,
        "above_ema50": close > ema50 > 0,
        "above_ma200": close > ma200 > 0,
    }


def _score_swing(m):
    score = 0
    reasons = []
    penalties = []

    def add(points, reason):
        nonlocal score
        score += points
        reasons.append(reason)

    def penalize(points, reason):
        nonlocal score
        score -= points
        penalties.append(reason)

    if m["close"] > m["ema20"] > m["ema50"]:
        add(15, "Giá trên EMA20 và EMA50")
    elif m["close"] > m["ema20"]:
        add(8, "Giá lấy lại EMA20")
    else:
        penalize(10, "Giá dưới EMA20")

    if m["ema50"] > m["ma200"] > 0:
        add(10, "Xu hướng trung hạn trên MA200")

    if m["rsi"] >= 50 and m["rsi"] <= 68:
        add(12, "RSI xác nhận động lượng")
    elif m["rsi"] > 75:
        penalize(10, "RSI quá nóng")

    if m["macd"] > m["macd_signal"]:
        add(12, "MACD tích cực")

    if m["volume_ratio"] >= 1.3:
        add(12, "Khối lượng tăng xác nhận")
    elif m["volume_ratio"] >= 1:
        add(6, "Khối lượng duy trì")

    if m["obv_rising"]:
        add(8, "OBV cải thiện")

    if m["breakout20"]:
        add(15, "Vượt đỉnh 20 phiên")
    elif m["high20"] > 0 and m["close"] >= m["high20"] * 0.97:
        add(8, "Áp sát đỉnh 20 phiên")

    if m["close_position"] >= 0.7:
        add(6, "Nến đóng cửa gần đỉnh")

    if 1 <= m["atr_pct"] <= 5:
        add(5, "Biến động phù hợp lướt sóng")
    elif m["atr_pct"] > 7:
        penalize(10, "Biến động quá lớn")

    return score, reasons, penalties


def _score_reversal(m, data):
    score = 0
    reasons = []
    penalties = []

    def add(points, reason):
        nonlocal score
        score += points
        reasons.append(reason)

    def penalize(points, reason):
        nonlocal score
        score -= points
        penalties.append(reason)

    recent_rsi = data["rsi"].iloc[-5:]
    recent_close = data["close"].iloc[-5:]

    rsi_recovering = (
        m["rsi"] > recent_rsi.iloc[0]
        and m["rsi"] < 55
    )

    price_recovering = (
        m["close"] > m["open"]
        and m["close"] > m["low20"]
    )

    if 25 <= m["rsi"] <= 45 and rsi_recovering:
        add(18, "RSI phục hồi từ vùng yếu")
    elif m["rsi"] < 25:
        penalize(8, "RSI quá thấp, cần thêm xác nhận")
    elif m["rsi"] > 60:
        penalize(8, "Giá có thể đã hồi phục xa")

    if price_recovering:
        add(12, "Giá có phản ứng hồi phục")

    if m["close"] > m["ema20"]:
        add(15, "Giá lấy lại EMA20")
    elif m["close"] >= m["ema20"] * 0.98:
        add(8, "Giá áp sát EMA20")
    else:
        penalize(8, "Chưa lấy lại EMA20")

    if m["macd_hist"] > m["macd_hist_prev"]:
        add(12, "MACD histogram cải thiện")

    if m["macd"] > m["macd_signal"]:
        add(8, "MACD xác nhận tích cực")

    if m["volume_ratio"] >= 1.2:
        add(10, "Khối lượng xác nhận hồi phục")
    elif m["volume_ratio"] < 0.6:
        penalize(8, "Khối lượng hồi phục yếu")

    if m["obv_rising"]:
        add(8, "OBV cải thiện")

    if m["mfi"] > 40:
        add(7, "MFI cải thiện")

    if m["close"] > recent_close.iloc[-2]:
        add(5, "Giá duy trì nhịp hồi")

    if m["close"] < m["low20"]:
        penalize(12, "Giá vẫn dưới đáy 20 phiên")

    return score, reasons, penalties


def _score_pullback(m, data):
    score = 0
    reasons = []
    penalties = []

    def add(points, reason):
        nonlocal score
        score += points
        reasons.append(reason)

    def penalize(points, reason):
        nonlocal score
        score -= points
        penalties.append(reason)

    if m["ema20"] > m["ema50"] > m["ma200"] > 0:
        add(20, "Xu hướng tăng đồng thuận")
    elif m["ema20"] > m["ema50"]:
        add(12, "Xu hướng trung hạn tích cực")
    else:
        penalize(12, "Xu hướng chưa đủ mạnh")

    distance = (
        abs(m["close"] - m["ema20"]) / m["ema20"]
        if m["ema20"] > 0 else 1
    )

    if distance <= 0.025:
        add(18, "Giá điều chỉnh gần EMA20")
    elif distance <= 0.045:
        add(10, "Giá nằm trong vùng EMA20")
    else:
        penalize(8, "Giá chưa ở vùng pullback phù hợp")

    if m["close"] >= m["ema20"] * 0.98:
        add(8, "Giá giữ vùng EMA20")
    else:
        penalize(8, "Giá mất vùng EMA20")

    if 40 <= m["rsi"] <= 60:
        add(12, "RSI phù hợp nhịp điều chỉnh")
    elif m["rsi"] < 35:
        penalize(8, "RSI suy yếu sâu")

    if m["macd_hist"] > m["macd_hist_prev"]:
        add(8, "MACD histogram cải thiện")

    if m["obv_rising"]:
        add(8, "OBV duy trì tích cực")

    if 0 < m["volume_dryup"] <= 0.85:
        add(10, "Khối lượng co hẹp khi điều chỉnh")
    elif m["volume_ratio"] >= 1.5:
        penalize(8, "Khối lượng tăng mạnh, cần kiểm tra áp lực bán")

    if m["close"] > m["ema50"]:
        add(8, "Giá vẫn trên EMA50")

    if m["close_position"] >= 0.65 and m["close"] > m["open"]:
        add(8, "Nến hồi phục có chất lượng")

    return score, reasons, penalties


def _strategy_signal(score, confirmed, blockers):
    if confirmed and score >= BUY_SCORE:
        return "MUA"

    if score >= WATCH_SCORE:
        return "CHỜ XÁC NHẬN"

    if score >= 40:
        return "THEO DÕI"

    return "BỎ QUA"


def _strategy_result(score, reasons, penalties, confirmed, blockers):
    score = max(0, min(100, int(round(score))))

    return {
        "score": score,
        "signal": _strategy_signal(
            score,
            confirmed,
            blockers
        ),
        "confirmed": bool(confirmed),
        "reasons": reasons,
        "penalties": penalties,
        "blockers": blockers,
    }


def analyze_stock(symbol, df, market_df=None):
    """
    Phân tích ba chiến lược độc lập bằng dữ liệu D1.

    MUA chỉ được phát ra khi:
    - Điểm đạt ngưỡng.
    - Điều kiện xác nhận riêng của chiến lược đạt.
    - Thanh khoản đạt yêu cầu.
    """

    data = _prepare(df)
    m = _base_metrics(data)

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Thiếu mã cổ phiếu.")

    liquidity_blockers = []

    if m["avg_turnover20"] < MIN_AVG_TURNOVER:
        liquidity_blockers.append(
            "GTGD bình quân 20 phiên dưới 5 tỷ"
        )

    if m["median_turnover20"] < MIN_MEDIAN_TURNOVER:
        liquidity_blockers.append(
            "GTGD trung vị 20 phiên dưới 3 tỷ"
        )

    relative_strength = _relative_strength(
        data,
        market_df
    )

    rs_bonus = 0

    if relative_strength is not None:
        if relative_strength >= 0.03:
            rs_bonus = 5
        elif relative_strength <= -0.03:
            rs_bonus = -5

    swing_score, swing_reasons, swing_penalties = _score_swing(m)
    reversal_score, reversal_reasons, reversal_penalties = _score_reversal(m, data)
    pullback_score, pullback_reasons, pullback_penalties = _score_pullback(m, data)

    swing_score += rs_bonus
    reversal_score += rs_bonus
    pullback_score += rs_bonus

    swing_blockers = []
    if not m["close"] > m["ema20"]:
        swing_blockers.append("Giá chưa trên EMA20")
    if not m["rsi"] < 70:
        swing_blockers.append("RSI quá cao")
    if not m["macd"] > m["macd_signal"]:
        swing_blockers.append("MACD chưa xác nhận")
    if not (m["volume_ratio"] >= 0.8):
        swing_blockers.append("Khối lượng chưa đạt")
    if m["atr_pct"] > 7:
        swing_blockers.append("Biến động quá cao")

    swing_confirmed = not swing_blockers

    reversal_blockers = []
    if not m["close"] > m["open"]:
        reversal_blockers.append("Chưa có nến hồi phục")
    if not m["close"] > m["close"] * 0.98:
        reversal_blockers.append("Giá chưa xác nhận hồi phục")
    if not (m["macd_hist"] > m["macd_hist_prev"]):
        reversal_blockers.append("MACD histogram chưa cải thiện")
    if not (m["rsi"] > 30 and m["rsi"] < 60):
        reversal_blockers.append("RSI chưa xác nhận vùng hồi phục")
    if not (m["volume_ratio"] >= 0.8):
        reversal_blockers.append("Khối lượng chưa xác nhận")

    reversal_confirmed = not reversal_blockers

    pullback_blockers = []
    if not m["ema20"] > m["ema50"]:
        pullback_blockers.append("EMA20 chưa trên EMA50")
    if not m["close"] >= m["ema20"] * 0.98:
        pullback_blockers.append("Giá chưa giữ được EMA20")
    if not (m["rsi"] >= 40 and m["rsi"] <= 65):
        pullback_blockers.append("RSI chưa phù hợp")
    if not (m["macd_hist"] > m["macd_hist_prev"]):
        pullback_blockers.append("MACD chưa cải thiện")
    if not (m["volume_ratio"] <= 1.5):
        pullback_blockers.append("Khối lượng bất thường")

    pullback_confirmed = not pullback_blockers

    if liquidity_blockers:
        swing_blockers.extend(liquidity_blockers)
        reversal_blockers.extend(liquidity_blockers)
        pullback_blockers.extend(liquidity_blockers)

        swing_confirmed = False
        reversal_confirmed = False
        pullback_confirmed = False

    strategies = {
        "SWING": {
            "name": STRATEGIES["SWING"],
            **_strategy_result(
                swing_score,
                swing_reasons,
                swing_penalties,
                swing_confirmed,
                swing_blockers
            )
        },
        "REVERSAL": {
            "name": STRATEGIES["REVERSAL"],
            **_strategy_result(
                reversal_score,
                reversal_reasons,
                reversal_penalties,
                reversal_confirmed,
                reversal_blockers
            )
        },
        "PULLBACK": {
            "name": STRATEGIES["PULLBACK"],
            **_strategy_result(
                pullback_score,
                pullback_reasons,
                pullback_penalties,
                pullback_confirmed,
                pullback_blockers
            )
        },
    }

    best_strategy = max(
        strategies.items(),
        key=lambda item: item[1]["score"]
    )

    atr = m["atr"]

    stop_loss = (
        round(m["close"] - 1.5 * atr)
        if atr > 0 else None
    )

    target_price = (
        round(m["close"] + 3 * atr)
        if atr > 0 else None
    )

    return {
        "symbol": symbol,
        "price": round(m["close"]),
        "rsi": round(m["rsi"], 2),
        "macd": round(m["macd"], 4),
        "mfi": round(m["mfi"], 2),
        "volume_ratio": round(m["volume_ratio"], 2),
        "atr": round(atr, 2),
        "atr_pct": round(m["atr_pct"], 2),
        "support": round(m["low20"], 2),
        "resistance": round(m["high20"], 2),
        "relative_strength": (
            round(relative_strength * 100, 2)
            if relative_strength is not None
            else None
        ),
        "liquidity_ok": not bool(liquidity_blockers),
        "liquidity_blockers": liquidity_blockers,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "strategies": strategies,
        "best_strategy": best_strategy[0],
        "best_score": best_strategy[1]["score"],
        "best_signal": best_strategy[1]["signal"],
    }


def analyze_market_data(symbol, df, market_df=None):
    return analyze_stock(
        symbol,
        df,
        market_df
    )
