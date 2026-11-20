"""Deterministic D1 candlestick pattern recognition for the live scanner.

Patterns are descriptive evidence only. They are confirmed by location,
trend context and volume before affecting a strategy score.
"""
import numpy as np
import pandas as pd


BULLISH = {
    "Hammer", "Inverted Hammer", "Bullish Engulfing", "Piercing Line",
    "Morning Star", "Three White Soldiers",
}
BEARISH = {
    "Shooting Star", "Hanging Man", "Bearish Engulfing",
    "Dark Cloud Cover", "Evening Star", "Three Black Crows",
}
NEUTRAL = {"Doji", "Spinning Top"}


def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def detect_candlestick_patterns(df):
    """Detect patterns on the latest completed D1 candle.

    Accepts either lowercase or title-case OHLCV column names.
    Returns pattern names, direction, contextual confirmation and notes.
    """
    if not isinstance(df, pd.DataFrame) or len(df) < 25:
        return {
            "patterns": [], "bullish": [], "bearish": [], "neutral": [],
            "context": "INSUFFICIENT_DATA", "confirmed_bullish": [],
            "confirmed_bearish": [], "notes": ["Không đủ dữ liệu nến"],
        }

    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    required = ["open", "high", "low", "close", "volume"]
    if any(c not in d.columns for c in required):
        return {
            "patterns": [], "bullish": [], "bearish": [], "neutral": [],
            "context": "INVALID_DATA", "confirmed_bullish": [],
            "confirmed_bearish": [], "notes": ["Thiếu OHLCV"],
        }
    d = d[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 25:
        return {
            "patterns": [], "bullish": [], "bearish": [], "neutral": [],
            "context": "INSUFFICIENT_DATA", "confirmed_bullish": [],
            "confirmed_bearish": [], "notes": ["Không đủ dữ liệu nến hợp lệ"],
        }

    rows = d.iloc[-3:].to_dict("records")
    p, q, r = rows[-3], rows[-2], rows[-1]

    def parts(x):
        o, h, l, c = (_safe(x[k]) for k in ("open", "high", "low", "close"))
        span = max(h - l, 1e-9)
        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l
        return o, h, l, c, span, body, upper, lower

    o, h, l, c, span, body, upper, lower = parts(r)
    po, ph, pl, pc, pspan, pbody, pupper, plower = parts(q)
    ppo, pph, ppl, ppc, ppspan, ppbody, ppupper, pplower = parts(p)

    patterns = []
    # Single-candle patterns.
    if body / span <= 0.10:
        patterns.append("Doji")
    elif body / span <= 0.30 and upper / span >= 0.25 and lower / span >= 0.25:
        patterns.append("Spinning Top")

    if lower >= max(body * 2.0, span * 0.45) and upper <= span * 0.20 and c >= l + span * 0.60:
        patterns.append("Hammer")
    if upper >= max(body * 2.0, span * 0.45) and lower <= span * 0.20 and c <= l + span * 0.40:
        patterns.append("Shooting Star")
    if upper >= max(body * 2.0, span * 0.45) and lower <= span * 0.20 and c > o:
        patterns.append("Inverted Hammer")
    if lower >= max(body * 2.0, span * 0.45) and upper <= span * 0.20 and c < o:
        patterns.append("Hanging Man")

    # Two-candle patterns.
    if pc < po and c > o and o <= pc and c >= po:
        patterns.append("Bullish Engulfing")
    if pc > po and c < o and o >= pc and c <= po:
        patterns.append("Bearish Engulfing")
    if pc < po and c > o and o < pl + pspan * 0.50 and c > (po + pc) / 2 and c < po:
        patterns.append("Piercing Line")
    if pc > po and c < o and o > pc and c < (po + pc) / 2 and c > po:
        patterns.append("Dark Cloud Cover")

    # Three-candle patterns. Middle candle must be relatively small.
    small_mid = pbody <= pspan * 0.35
    if ppc < ppo and small_mid and c > o and c > (ppo + ppc) / 2:
        patterns.append("Morning Star")
    if ppc > ppo and small_mid and c < o and c < (ppo + ppc) / 2:
        patterns.append("Evening Star")
    if (ppc > ppo and pc > po and c > o
            and pc > ppc and c > pc
            and pbody / ppspan >= 0.45 and body / span >= 0.45
            and ppbody / ppspan >= 0.45):
        patterns.append("Three White Soldiers")
    if (ppc < ppo and pc < po and c < o
            and pc < ppc and c < pc
            and pbody / ppspan >= 0.45 and body / span >= 0.45
            and ppbody / ppspan >= 0.45):
        patterns.append("Three Black Crows")

    close = d["close"]
    volume = d["volume"]
    avg_volume = _safe(volume.iloc[-21:-1].mean())
    volume_ratio = _safe(volume.iloc[-1] / avg_volume) if avg_volume > 0 else 0
    support = _safe(d["low"].iloc[-21:-1].min())
    resistance = _safe(d["high"].iloc[-21:-1].max())
    near_support = support > 0 and (c - support) / c <= 0.035
    near_resistance = resistance > 0 and (resistance - c) / c <= 0.035
    prior_return = (
        _safe(close.iloc[-1] / close.iloc[-11] - 1)
        if _safe(close.iloc[-11]) > 0 else 0
    )

    bullish = [x for x in patterns if x in BULLISH]
    bearish = [x for x in patterns if x in BEARISH]
    neutral = [x for x in patterns if x in NEUTRAL]

    confirmed_bullish = []
    confirmed_bearish = []
    notes = []
    for name in bullish:
        # Hammer/engulfing/morning-star type signals need support or a decline;
        # breakout continuation is allowed when price clears prior resistance.
        location_ok = near_support or prior_return <= -0.04 or (
            name in {"Three White Soldiers"} and c > resistance
        )
        volume_ok = volume_ratio >= 0.80
        if location_ok and volume_ok:
            confirmed_bullish.append(name)
    for name in bearish:
        location_ok = near_resistance or prior_return >= 0.04
        volume_ok = volume_ratio >= 0.80
        if location_ok and volume_ok:
            confirmed_bearish.append(name)

    if bullish and not confirmed_bullish:
        notes.append("Có nến tăng nhưng chưa ở vùng hỗ trợ/xu hướng phù hợp hoặc thiếu khối lượng")
    if bearish and not confirmed_bearish:
        notes.append("Có nến giảm nhưng chưa ở vùng kháng cự/xu hướng phù hợp hoặc thiếu khối lượng")
    if confirmed_bullish:
        notes.append("Mô hình nến tăng được xác nhận bởi vị trí giá và khối lượng")
    if confirmed_bearish:
        notes.append("Mô hình nến giảm được xác nhận bởi vị trí giá và khối lượng")

    return {
        "patterns": patterns,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "confirmed_bullish": confirmed_bullish,
        "confirmed_bearish": confirmed_bearish,
        "context": {
            "near_support": bool(near_support),
            "near_resistance": bool(near_resistance),
            "prior_10d_return_pct": round(prior_return * 100, 2),
            "volume_ratio": round(volume_ratio, 2),
        },
        "notes": notes,
    }
