
import numpy as np
import pandas as pd

from candlestick_patterns import detect_candlestick_patterns

STRATEGIES = {
    "SWING": "LƯỚT SÓNG 2–5 PHIÊN",
    "REVERSAL": "ĐẢO CHIỀU 5–10 PHIÊN",
    "PULLBACK": "MUA KHI ĐIỀU CHỈNH 3–7 PHIÊN",
}

MIN_HISTORY = 200
MIN_AVG_TURNOVER = 5_000_000_000
MIN_MEDIAN_TURNOVER = 3_000_000_000

# Điểm kỹ thuật, không phải xác suất thắng.
BUY_SCORE = 78
WAIT_SCORE = 60
WATCH_SCORE = 40


def _num(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _prepare(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dữ liệu D1 không hợp lệ.")

    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in d.columns]

    if missing:
        raise ValueError(f"Thiếu dữ liệu: {', '.join(missing)}")

    for c in required:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date")
        d = d.drop_duplicates("date", keep="last")

    d = d.dropna(subset=required).reset_index(drop=True)

    valid = (
        (d["open"] > 0)
        & (d["close"] > 0)
        & (d["high"] >= d[["open", "close", "low"]].max(axis=1))
        & (d["low"] <= d[["open", "close"]].min(axis=1))
        & (d["volume"] >= 0)
    )
    d = d.loc[valid].reset_index(drop=True)

    if len(d) < MIN_HISTORY:
        raise ValueError(
            f"Cần tối thiểu {MIN_HISTORY} phiên D1; "
            f"hiện có {len(d)} phiên."
        )

    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ma200"] = c.rolling(200).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)
    d.loc[(loss == 0) & (gain > 0), "rsi"] = 100
    d.loc[(gain == 0) & (loss > 0), "rsi"] = 0
    d["rsi"] = d["rsi"].fillna(50)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()

    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(
        span=9, adjust=False
    ).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["macd_hist_prev"] = d["macd_hist"].shift(1)

    d["volume_avg20"] = v.shift(1).rolling(20).mean()
    d["volume_ratio"] = (
        v / d["volume_avg20"].replace(0, np.nan)
    )

    d["volume_avg5_prev"] = v.shift(1).rolling(5).mean()
    d["volume_dryup"] = (
        d["volume_avg5_prev"]
        / d["volume_avg20"].replace(0, np.nan)
    )

    d["turnover"] = c * v
    d["avg_turnover20"] = (
        d["turnover"].shift(1).rolling(20).mean()
    )
    d["median_turnover20"] = (
        d["turnover"].shift(1).rolling(20).median()
    )

    previous_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - previous_close).abs(),
        (l - previous_close).abs(),
    ], axis=1).max(axis=1)

    d["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    d["atr_pct"] = d["atr"] / c * 100

    typical = (h + l + c) / 3
    flow = typical * v
    direction = typical.diff()

    positive = flow.where(direction > 0, 0).rolling(14).sum()
    negative = flow.where(direction < 0, 0).rolling(14).sum()

    ratio = positive / negative.replace(0, np.nan)
    d["mfi"] = 100 - 100 / (1 + ratio)
    d.loc[(negative == 0) & (positive > 0), "mfi"] = 100
    d["mfi"] = d["mfi"].fillna(50)

    d["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    d["obv_avg10"] = d["obv"].rolling(10).mean()

    d["high20_prev"] = h.shift(1).rolling(20).max()
    d["high50_prev"] = h.shift(1).rolling(50).max()
    d["low20_prev"] = l.shift(1).rolling(20).min()
    d["low10_prev"] = l.shift(1).rolling(10).min()

    d["ema20_prev5"] = d["ema20"].shift(5)
    d["ema50_prev5"] = d["ema50"].shift(5)

    return d.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def _metrics(d):
    r = d.iloc[-1]

    close = _num(r["close"])
    open_price = _num(r["open"])
    high = _num(r["high"])
    low = _num(r["low"])
    ema20 = _num(r["ema20"])
    ema50 = _num(r["ema50"])
    ma200 = _num(r["ma200"])
    atr = _num(r["atr"])

    candle_range = max(high - low, 1e-9)

    ema20_prev5 = _num(r["ema20_prev5"])
    ema50_prev5 = _num(r["ema50_prev5"])

    return {
        "close": close,
        "open": open_price,
        "high": high,
        "low": low,
        "ema20": ema20,
        "ema50": ema50,
        "ma200": ma200,
        "rsi": _num(r["rsi"], 50),
        "macd": _num(r["macd"]),
        "macd_signal": _num(r["macd_signal"]),
        "macd_hist": _num(r["macd_hist"]),
        "macd_hist_prev": _num(r["macd_hist_prev"]),
        "mfi": _num(r["mfi"], 50),
        "volume_ratio": _num(r["volume_ratio"]),
        "volume_dryup": _num(r["volume_dryup"]),
        "avg_turnover20": _num(r["avg_turnover20"]),
        "median_turnover20": _num(r["median_turnover20"]),
        "atr": atr,
        "atr_pct": _num(r["atr_pct"]),
        "high20": _num(r["high20_prev"]),
        "high50": _num(r["high50_prev"]),
        "low20": _num(r["low20_prev"]),
        "low10": _num(r["low10_prev"]),
        "close_position": (close - low) / candle_range,
        "ema20_slope": (
            ema20 / ema20_prev5 - 1 if ema20_prev5 > 0 else 0
        ),
        "ema50_slope": (
            ema50 / ema50_prev5 - 1 if ema50_prev5 > 0 else 0
        ),
        "obv_rising": (
            _num(r["obv"]) > _num(r["obv_avg10"])
        ),
    }


def _score_swing(m):
    score = 0
    reasons = []
    penalties = []
    blockers = []

    if m["close"] > m["ema20"] > m["ema50"]:
        score += 20
        reasons.append("Giá trên EMA20 và EMA50")
    else:
        blockers.append("Chưa có cấu trúc giá trên EMA20 và EMA50")

    if m["ema50"] > m["ma200"] > 0:
        score += 10
        reasons.append("EMA50 trên MA200")
    else:
        penalties.append("Xu hướng dài hạn chưa đồng thuận")

    if m["ema20_slope"] > 0 and m["ema50_slope"] > 0:
        score += 10
        reasons.append("EMA20 và EMA50 đang hướng lên")
    else:
        blockers.append("Đường trung bình chưa cùng hướng tăng")

    if 50 <= m["rsi"] <= 68:
        score += 12
        reasons.append("RSI xác nhận động lượng")
    elif m["rsi"] > 75:
        penalties.append("RSI quá nóng")
    else:
        blockers.append("RSI chưa xác nhận động lượng")

    if m["macd"] > m["macd_signal"] and m["macd_hist"] > 0:
        score += 12
        reasons.append("MACD dương và trên đường tín hiệu")
    else:
        blockers.append("MACD chưa xác nhận xu hướng tăng")

    if m["volume_ratio"] >= 1.2:
        score += 12
        reasons.append("Khối lượng tăng xác nhận")
    elif m["volume_ratio"] >= 0.9:
        score += 6
        reasons.append("Khối lượng duy trì")
    else:
        blockers.append("Khối lượng chưa đủ xác nhận")

    if m["obv_rising"]:
        score += 8
        reasons.append("OBV cải thiện")

    if m["high20"] > 0 and m["close"] > m["high20"]:
        score += 10
        reasons.append("Vượt đỉnh 20 phiên")
    elif m["high20"] > 0 and m["close"] >= m["high20"] * 0.97:
        score += 5
        reasons.append("Áp sát đỉnh 20 phiên")

    distance = (
        m["close"] / m["ema20"] - 1
        if m["ema20"] > 0 else 0
    )

    if distance > 0.06:
        blockers.append("Giá đã tăng quá xa EMA20")
    elif distance < 0:
        blockers.append("Giá còn dưới EMA20")

    if m["close_position"] >= 0.65:
        score += 6
        reasons.append("Nến đóng gần vùng cao")

    if not 0 < m["atr_pct"] <= 7:
        blockers.append("Biến động ATR không phù hợp")

    return score, reasons, penalties, blockers


def _score_reversal(m, d):
    score = 0
    reasons = []
    penalties = []
    blockers = []

    recent_rsi = d["rsi"].iloc[-5:]
    recent_close = d["close"].iloc[-5:]

    rsi_recovering = (
        m["rsi"] > _num(recent_rsi.iloc[0])
        and m["rsi"] < 60
    )

    if 25 <= m["rsi"] <= 48 and rsi_recovering:
        score += 18
        reasons.append("RSI phục hồi từ vùng yếu")
    else:
        blockers.append("RSI chưa thể hiện quá trình hồi phục rõ")

    if m["close"] > m["open"] and m["close_position"] >= 0.6:
        score += 12
        reasons.append("Nến tăng đóng cửa tích cực")
    else:
        blockers.append("Chưa có nến tăng xác nhận")

    if m["close"] > m["ema20"]:
        score += 15
        reasons.append("Giá lấy lại EMA20")
    elif m["close"] >= m["ema20"] * 0.98:
        score += 6
        blockers.append("Giá mới áp sát EMA20, chưa lấy lại")
    else:
        blockers.append("Giá còn cách xa EMA20")

    if m["macd_hist"] > m["macd_hist_prev"]:
        score += 12
        reasons.append("MACD histogram cải thiện")
    else:
        blockers.append("MACD histogram chưa cải thiện")

    if m["macd"] > m["macd_signal"]:
        score += 8
        reasons.append("MACD xác nhận tích cực")

    if m["volume_ratio"] >= 1.0:
        score += 10
        reasons.append("Khối lượng xác nhận hồi phục")
    else:
        blockers.append("Khối lượng hồi phục chưa đủ")

    if m["obv_rising"]:
        score += 8
        reasons.append("OBV cải thiện")

    if m["mfi"] > 45:
        score += 7
        reasons.append("MFI cải thiện")

    if (
        len(recent_close) >= 2
        and recent_close.iloc[-1] > recent_close.iloc[-2]
    ):
        score += 5
        reasons.append("Giá duy trì nhịp hồi")

    if m["low20"] > 0 and m["close"] < m["low20"]:
        blockers.append("Giá phá đáy 20 phiên")

    if m["atr_pct"] > 7:
        blockers.append("Biến động quá cao")

    return score, reasons, penalties, blockers


def _score_pullback(m):
    score = 0
    reasons = []
    penalties = []
    blockers = []

    if m["ema20"] > m["ema50"] > m["ma200"] > 0:
        score += 20
        reasons.append("Xu hướng tăng đồng thuận")
    elif m["ema20"] > m["ema50"] > 0:
        score += 12
        reasons.append("Xu hướng trung hạn tích cực")
    else:
        blockers.append("EMA20 chưa nằm trên EMA50")

    distance = (
        abs(m["close"] - m["ema20"]) / m["ema20"]
        if m["ema20"] > 0 else 1
    )

    if distance <= 0.025:
        score += 15
        reasons.append("Giá nằm gần EMA20")
    elif distance <= 0.045:
        score += 8
        reasons.append("Giá nằm trong vùng EMA20")
    else:
        blockers.append("Giá chưa ở vùng pullback hợp lý")

    if m["close"] > m["ema20"]:
        score += 10
        reasons.append("Giá đóng cửa trên EMA20")
    else:
        blockers.append("Giá chưa lấy lại EMA20")

    if m["close"] > m["open"] and m["close_position"] >= 0.6:
        score += 10
        reasons.append("Nến hồi phục đóng cửa tích cực")
    else:
        blockers.append("Chưa có nến hồi phục xác nhận")

    if 42 <= m["rsi"] <= 65:
        score += 10
        reasons.append("RSI phù hợp nhịp điều chỉnh")
    else:
        blockers.append("RSI chưa phù hợp")

    if m["macd_hist"] > m["macd_hist_prev"]:
        score += 10
        reasons.append("MACD histogram cải thiện")
    else:
        blockers.append("MACD chưa cải thiện")

    if m["obv_rising"]:
        score += 8
        reasons.append("OBV duy trì tích cực")

    if 0 < m["volume_dryup"] <= 0.85:
        score += 8
        reasons.append("Khối lượng co hẹp trong nhịp điều chỉnh")
    elif m["volume_ratio"] > 1.5:
        blockers.append("Khối lượng tăng đột biến, cần kiểm tra áp lực bán")

    if m["close"] > m["ema50"]:
        score += 7
        reasons.append("Giá vẫn trên EMA50")
    else:
        blockers.append("Giá đã mất EMA50")

    if m["atr_pct"] > 7:
        blockers.append("Biến động ATR quá cao")

    return score, reasons, penalties, blockers


def _relative_strength(stock_df, market_df=None):
    """So sánh hiệu suất 20 phiên trên các ngày giao dịch chung."""
    if market_df is None:
        return None

    try:
        market = market_df.copy()
        market.columns = [
            str(c).strip().lower() for c in market.columns
        ]

        stock = stock_df.copy()
        stock.columns = [
            str(c).strip().lower() for c in stock.columns
        ]

        if "close" not in market.columns or "close" not in stock.columns:
            return None

        market_close = pd.to_numeric(
            market["close"], errors="coerce"
        )
        stock_close = pd.to_numeric(
            stock["close"], errors="coerce"
        )

        # Đồng bộ theo ngày để tránh so sánh lệch phiên khi một nguồn
        # thiếu nến hoặc có lịch nghỉ giao dịch khác nhau.
        if "date" in market.columns and "date" in stock.columns:
            market_dates = pd.to_datetime(
                market["date"], errors="coerce"
            )
            stock_dates = pd.to_datetime(
                stock["date"], errors="coerce"
            )
            market_series = pd.DataFrame({
                "date": market_dates,
                "market_close": market_close,
            }).dropna()
            stock_series = pd.DataFrame({
                "date": stock_dates,
                "stock_close": stock_close,
            }).dropna()

            aligned = (
                stock_series.merge(market_series, on="date", how="inner")
                .sort_values("date")
                .drop_duplicates("date", keep="last")
            )
            if len(aligned) < 21:
                return None

            stock_start = float(aligned["stock_close"].iloc[-21])
            stock_end = float(aligned["stock_close"].iloc[-1])
            market_start = float(aligned["market_close"].iloc[-21])
            market_end = float(aligned["market_close"].iloc[-1])
        else:
            market_values = market_close.dropna()
            stock_values = stock_close.dropna()
            if len(market_values) < 21 or len(stock_values) < 21:
                return None
            stock_start = float(stock_values.iloc[-21])
            stock_end = float(stock_values.iloc[-1])
            market_start = float(market_values.iloc[-21])
            market_end = float(market_values.iloc[-1])

        if min(stock_start, market_start) <= 0:
            return None

        stock_return = stock_end / stock_start - 1
        market_return = market_end / market_start - 1
        return float(stock_return - market_return)

    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _make_result(
    score,
    reasons,
    penalties,
    blockers,
    liquidity_blockers,
):
    score = max(0, min(100, int(round(score))))

    blockers = list(dict.fromkeys(
        blockers + liquidity_blockers
    ))

    if not blockers and score >= BUY_SCORE:
        signal = "MUA"
    elif score >= WAIT_SCORE:
        signal = "CHỜ XÁC NHẬN"
    elif score >= WATCH_SCORE:
        signal = "THEO DÕI"
    else:
        signal = "BỎ QUA"

    return {
        "score": score,
        "signal": signal,
        "confirmed": not bool(blockers),
        "reasons": reasons,
        "penalties": penalties,
        "blockers": blockers,
    }


def analyze_stock(symbol, df, market_df=None):
    data = _prepare(df)
    m = _metrics(data)

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

    relative_strength = _relative_strength(data, market_df)

    rs_bonus = 0
    rs_penalty = 0

    if relative_strength is not None:
        if relative_strength >= 0.03:
            rs_bonus = 5
        elif relative_strength <= -0.03:
            rs_penalty = 5

    swing = _score_swing(m)
    reversal = _score_reversal(m, data)
    pullback = _score_pullback(m)
    candle = detect_candlestick_patterns(data)
    bullish_confirmed = candle.get("confirmed_bullish", [])
    bearish_confirmed = candle.get("confirmed_bearish", [])

    # Nến chỉ bổ trợ điểm khi có xác nhận vị trí và khối lượng.
    # Nến giảm đã xác nhận sẽ chặn tín hiệu mua mới của chiến lược đó.
    if bullish_confirmed:
        reversal[0] += 8
        reversal[1].append("Mô hình nến tăng xác nhận: " + ", ".join(bullish_confirmed))
        pullback[0] += 5
        pullback[1].append("Nến hồi phục xác nhận tại vùng hỗ trợ: " + ", ".join(bullish_confirmed))
        if "Three White Soldiers" in bullish_confirmed:
            swing[0] += 4
            swing[1].append("Three White Soldiers được xác nhận")
    if bearish_confirmed:
        warning = "Mô hình nến giảm xác nhận: " + ", ".join(bearish_confirmed)
        for raw in (swing, reversal, pullback):
            raw[0] -= 8
            raw[2].append(warning)
            raw[3].append("Tạm chặn mua mới do xuất hiện mô hình nến giảm được xác nhận")

    strategies_raw = {
        "SWING": swing,
        "REVERSAL": reversal,
        "PULLBACK": pullback,
    }

    strategies = {}

    for key, raw in strategies_raw.items():
        score, reasons, penalties, blockers = raw

        adjusted_score = score + rs_bonus - rs_penalty

        if rs_bonus:
            reasons = reasons + ["Sức mạnh tương đối vượt VN-Index"]

        if rs_penalty:
            penalties = penalties + ["Yếu hơn VN-Index trong 20 phiên"]

        strategies[key] = {
            "name": STRATEGIES[key],
            **_make_result(
                adjusted_score,
                reasons,
                penalties,
                blockers,
                liquidity_blockers,
            ),
        }

    best = max(
        strategies.items(),
        key=lambda item: item[1]["score"]
    )

    atr = m["atr"]

    return {
        "symbol": symbol,
        "price": round(m["close"]),
        "rsi": round(m["rsi"], 2),
        "mfi": round(m["mfi"], 2),
        "volume_ratio": round(m["volume_ratio"], 2),
        "atr": round(atr, 2),
        "atr_pct": round(m["atr_pct"], 2),
        "support": round(m["low20"], 2),
        "resistance": round(m["high20"], 2),
        "relative_strength": (
            round(relative_strength * 100, 2)
            if relative_strength is not None else None
        ),
        "liquidity_ok": not bool(liquidity_blockers),
        "liquidity_blockers": liquidity_blockers,
        "stop_loss": (
            round(m["close"] - 1.5 * atr)
            if atr > 0 else None
        ),
        "target_price": (
            round(m["close"] + 3 * atr)
            if atr > 0 else None
        ),
        "strategies": strategies,
        "candlestick_analysis": candle,
        "best_strategy": best[0],
        "best_score": best[1]["score"],
        "best_signal": best[1]["signal"],
    }


def analyze_market_data(symbol, df, market_df=None):
    return analyze_stock(symbol, df, market_df)
