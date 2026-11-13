
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

    d = d.dropna(subset=required).reset_index(drop=True)

    valid = (
        (d["open"] > 0)
        & (d["high"] >= d[["open", "close", "low"]].max(axis=1))
        & (d["low"] <= d[["open", "close"]].min(axis=1))
        & (d["close"] > 0)
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
        (l - previous_close).abs()
    ], axis=1).max(axis=1)

    d["atr"] = tr.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    d["atr_pct"] = d["atr"] / c * 100

    d["mfi_typical"] = (h + l + c) / 3
    flow = d["mfi_typical"] * v
    direction = d["mfi_typical"].diff()

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
    ema20_slope = (
        ema20 / ema20_prev5 - 1
        if ema20_prev5 > 0 else 0
    )

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
        "ema20_slope": ema20_slope,
        "obv_rising": (
            _num(r["obv"]) > _num(r["obv_avg10"])
        ),
    }


def _score_swing(m):
    score, reasons, penalties = 0, [], []

    if m["close"] > m["ema20"] > m["ema50"]:
        score += 15
        reasons.append("Giá trên EMA20 và EMA50")
    elif m["close"] > m["ema20"]:
        score += 8
        reasons.append("Giá lấy lại EMA20")
    else:
        score -= 10
        penalties.append("Giá dưới EMA20")

    if m["ema50"] > m["ma200"] > 0:
        score += 10
        reasons.append("EMA50 trên MA200")

    if 50 <= m["rsi"] <= 68:
        score += 12
        reasons.append("RSI xác nhận động lượng")
    elif m["rsi"] > 75:
        score -= 10
        penalties.append("RSI quá nóng")

    if m["macd"] > m["macd_signal"]:
        score += 12
        reasons.append("MACD tích cực")

    if m["volume_ratio"] >= 1.3:
        score += 12
        reasons.append("Khối lượng tăng xác nhận")
    elif m["volume_ratio"] >= 1:
        score += 6
        reasons.append("Khối lượng duy trì")

    if m["obv_rising"]:
        score += 8
        reasons.append("OBV cải thiện")

    if m["high20"] > 0 and m["close"] > m["high20"]:
        score += 15
        reasons.append("Vượt đỉnh 20 phiên")
    elif m["high20"] > 0 and m["close"] >= m["high20"] * 0.97:
        score += 8
        reasons.append("Áp sát đỉnh 20 phiên")

    if m["close_position"] >= 0.7:
        score += 6
        reasons.append("Nến đóng gần đỉnh")

    if 1 <= m["atr_pct"] <= 5:
        score += 5
        reasons.append("Biến động phù hợp")
    elif m["atr_pct"] > 7:
        score -= 10
        penalties.append("Biến động quá lớn")

    blockers = []
    if m["close"] <= m["ema20"]:
        blockers.append("Giá chưa trên EMA20")
    if not 45 <= m["rsi"] <= 70:
        blockers.append("RSI chưa phù hợp")
    if m["macd"] <= m["macd_signal"]:
        blockers.append("MACD chưa xác nhận")
    if m["volume_ratio"] < 0.8:
        blockers.append("Khối lượng thấp")
    if m["atr_pct"] > 7:
        blockers.append("Biến động quá cao")

    return score, reasons, penalties, blockers


def _score_reversal(m, d):
    score, reasons, penalties = 0, [], []

    recent_rsi = d["rsi"].iloc[-5:]
    recent_close = d["close"].iloc[-5:]

    rsi_recovering = (
        m["rsi"] > recent_rsi.iloc[0]
        and m["rsi"] < 55
    )

    if 25 <= m["rsi"] <= 45 and rsi_recovering:
        score += 18
        reasons.append("RSI phục hồi từ vùng yếu")
    elif m["rsi"] < 25:
        score -= 8
        penalties.append("RSI quá thấp, cần xác nhận thêm")
    elif m["rsi"] > 60:
        score -= 8
        penalties.append("Nhịp hồi có thể đã đi xa")

    if m["close"] > m["open"]:
        score += 12
        reasons.append("Nến ngày tăng")
    else:
        penalties.append("Chưa có nến hồi phục")

    if m["close"] > m["ema20"]:
        score += 15
        reasons.append("Giá lấy lại EMA20")
    elif m["close"] >= m["ema20"] * 0.98:
        score += 8
        reasons.append("Giá áp sát EMA20")
    else:
        score -= 8
        penalties.append("Chưa lấy lại EMA20")

    if m["macd_hist"] > m["macd_hist_prev"]:
        score += 12
        reasons.append("MACD histogram cải thiện")

    if m["macd"] > m["macd_signal"]:
        score += 8
        reasons.append("MACD xác nhận tích cực")

    if m["volume_ratio"] >= 1.2:
        score += 10
        reasons.append("Khối lượng xác nhận hồi phục")
    elif m["volume_ratio"] < 0.6:
        score -= 8
        penalties.append("Khối lượng hồi phục yếu")

    if m["obv_rising"]:
        score += 8
        reasons.append("OBV cải thiện")

    if m["mfi"] > 40:
        score += 7
        reasons.append("MFI cải thiện")

    if len(recent_close) >= 2 and recent_close.iloc[-1] > recent_close.iloc[-2]:
        score += 5
        reasons.append("Giá duy trì nhịp hồi")

    blockers = []
    if m["close"] <= m["open"]:
        blockers.append("Chưa có nến tăng xác nhận")
    if not (m["macd_hist"] > m["macd_hist_prev"]):
        blockers.append("MACD histogram chưa cải thiện")
    if not (30 < m["rsi"] < 60):
        blockers.append("RSI chưa ở vùng hồi phục")
    if m["volume_ratio"] < 0.8:
        blockers.append("Khối lượng chưa xác nhận")
    if m["low20"] > 0 and m["close"] < m["low20"]:
        blockers.append("Giá dưới đáy 20 phiên")

    return score, reasons, penalties, blockers


def _score_pullback(m):
    score, reasons, penalties = 0, [], []

    if m["ema20"] > m["ema50"] > m["ma200"] > 0:
        score += 20
        reasons.append("Xu hướng tăng đồng thuận")
    elif m["ema20"] > m["ema50"]:
        score += 12
        reasons.append("Xu hướng trung hạn tích cực")
    else:
        score -= 12
        penalties.append("Xu hướng chưa đủ mạnh")

    distance = (
        abs(m["close"] - m["ema20"]) / m["ema20"]
        if m["ema20"] > 0 else 1
    )

    if distance <= 0.025:
        score += 18
        reasons.append("Giá điều chỉnh gần EMA20")
    elif distance <= 0.045:
        score += 10
        reasons.append("Giá trong vùng EMA20")
    else:
        score -= 8
        penalties.append("Giá chưa ở vùng pullback")

    if m["close"] >= m["ema20"] * 0.98:
        score += 8
        reasons.append("Giá giữ vùng EMA20")

    if 40 <= m["rsi"] <= 60:
        score += 12
        reasons.append("RSI phù hợp nhịp điều chỉnh")

    if m["macd_hist"] > m["macd_hist_prev"]:
        score += 8
        reasons.append("MACD histogram cải thiện")

    if m["obv_rising"]:
        score += 8
        reasons.append("OBV duy trì tích cực")

    if 0 < m["volume_dryup"] <= 0.85:
        score += 10
        reasons.append("Khối lượng co hẹp khi điều chỉnh")
    elif m["volume_ratio"] >= 1.5:
        score -= 8
        penalties.append("Khối lượng tăng mạnh, cần kiểm tra áp lực bán")

    if m["close"] > m["ema50"]:
        score += 8
        reasons.append("Giá vẫn trên EMA50")

    if m["close_position"] >= 0.65 and m["close"] > m["open"]:
        score += 8
        reasons.append("Nến hồi phục tích cực")

    blockers = []
    if m["ema20"] <= m["ema50"]:
        blockers.append("EMA20 chưa trên EMA50")
    if m["close"] < m["ema20"] * 0.98:
        blockers.append("Giá chưa giữ được EMA20")
    if not 40 <= m["rsi"] <= 65:
        blockers.append("RSI chưa phù hợp")
    if m["macd_hist"] <= m["macd_hist_prev"]:
        blockers.append("MACD chưa cải thiện")
    if m["volume_ratio"] > 1.5:
        blockers.append("Khối lượng bất thường")

    return score, reasons, penalties, blockers


def _relative_strength(stock_df, market_df=None):
    if market_df is None:
        return None

    try:
        market = market_df.copy()
        market.columns = [str(c).strip().lower() for c in market.columns]

        if "close" not in market.columns:
            return None

        market_close = pd.to_numeric(
            market["close"], errors="coerce"
        ).dropna()

        stock_close = stock_df["close"].dropna()

        if len(market_close) < 21 or len(stock_close) < 21:
            return None

        stock_return = stock_close.iloc[-1] / stock_close.iloc[-21] - 1
        market_return = market_close.iloc[-1] / market_close.iloc[-21] - 1

        return float(stock_return - market_return)

    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _make_result(score, reasons, penalties, blockers, liquidity_blockers):
    score = max(0, min(100, int(round(score))))
    blockers = list(dict.fromkeys(blockers + liquidity_blockers))

    if not blockers and score >= 75:
        signal = "MUA"
    elif score >= 60:
        signal = "CHỜ XÁC NHẬN"
    elif score >= 40:
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
        liquidity_blockers.append("GTGD bình quân 20 phiên dưới 5 tỷ")

    if m["median_turnover20"] < MIN_MEDIAN_TURNOVER:
        liquidity_blockers.append("GTGD trung vị 20 phiên dưới 3 tỷ")

    relative_strength = _relative_strength(data, market_df)
    rs_bonus = 0

    if relative_strength is not None:
        if relative_strength >= 0.03:
            rs_bonus = 5
        elif relative_strength <= -0.03:
            rs_bonus = -5

    swing = _score_swing(m)
    reversal = _score_reversal(m, data)
    pullback = _score_pullback(m)

    strategies = {
        "SWING": {
            "name": STRATEGIES["SWING"],
            **_make_result(
                swing[0] + rs_bonus,
                swing[1], swing[2], swing[3],
                liquidity_blockers
            )
        },
        "REVERSAL": {
            "name": STRATEGIES["REVERSAL"],
            **_make_result(
                reversal[0] + rs_bonus,
                reversal[1], reversal[2], reversal[3],
                liquidity_blockers
            )
        },
        "PULLBACK": {
            "name": STRATEGIES["PULLBACK"],
            **_make_result(
                pullback[0] + rs_bonus,
                pullback[1], pullback[2], pullback[3],
                liquidity_blockers
            )
        },
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
        "stop_loss": round(m["close"] - 1.5 * atr) if atr > 0 else None,
        "target_price": round(m["close"] + 3 * atr) if atr > 0 else None,
        "strategies": strategies,
        "best_strategy": best[0],
        "best_score": best[1]["score"],
        "best_signal": best[1]["signal"],
    }


def analyze_market_data(symbol, df, market_df=None):
    return analyze_stock(symbol, df, market_df)
