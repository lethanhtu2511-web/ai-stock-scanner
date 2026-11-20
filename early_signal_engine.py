"""Early-warning layer for HOSE/HNX swing setups.

This module deliberately separates WATCH_EARLY from confirmed BUY.
It uses only completed D1 bars plus a verified live quote. It does not
interpret a missing/partial quote as real-time evidence.
"""
from __future__ import annotations

import math
import pandas as pd
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo('Asia/Ho_Chi_Minh')


def _safe(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def evaluate_early_signal(history: pd.DataFrame, quote: dict | None = None) -> dict:
    """Return an explainable early setup state and a 0-100 setup score."""
    if not isinstance(history, pd.DataFrame):
        return {"early_signal": "UNAVAILABLE", "early_score": 0,
                "early_reasons": ["Thiếu dữ liệu D1 hợp lệ."]}
    d = history.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    required = {"high", "low", "close", "volume"}
    if not required.issubset(d.columns) or len(d) < 60:
        return {"early_signal": "UNAVAILABLE", "early_score": 0,
                "early_reasons": ["Cần tối thiểu 60 phiên D1 hợp lệ."]}
    for col in required:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=list(required)).reset_index(drop=True)
    if len(d) < 60:
        return {"early_signal": "UNAVAILABLE", "early_score": 0,
                "early_reasons": ["Dữ liệu D1 không đủ sau chuẩn hóa."]}

    # Ignore the still-forming daily candle during market hours.
    if "date" in d.columns and len(d):
        dates = pd.to_datetime(d["date"], errors="coerce")
        now = datetime.now(VN_TZ)
        in_session = now.weekday() < 5 and (
            dt_time(9, 0) <= now.time() <= dt_time(11, 30)
            or dt_time(13, 0) <= now.time() <= dt_time(15, 0)
        )
        if in_session and pd.notna(dates.iloc[-1]) and dates.iloc[-1].date() == now.date():
            d = d.iloc[:-1].reset_index(drop=True)
    if len(d) < 60:
        return {"early_signal": "UNAVAILABLE", "early_score": 0,
                "early_reasons": ["Cần 60 phiên D1 đã hoàn tất."]}

    c, h, l, v = d.close, d.high, d.low, d.volume
    close = _safe(c.iloc[-1])
    if close <= 0:
        return {"early_signal": "UNAVAILABLE", "early_score": 0,
                "early_reasons": ["Giá đóng cửa D1 không hợp lệ."]}

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    avg_vol20 = v.shift(1).rolling(20).mean()
    avg_vol5 = v.shift(1).rolling(5).mean()
    high20 = h.shift(1).rolling(20).max()
    high55 = h.shift(1).rolling(55).max()
    low20 = l.shift(1).rolling(20).min()

    score, reasons, risks = 0, [], []
    if close > _safe(ema20.iloc[-1]) and _safe(ema20.iloc[-1]) > _safe(ema50.iloc[-1]):
        score += 18
        reasons.append("Cấu trúc xu hướng ngắn hạn tích cực: giá > EMA20 > EMA50.")
    elif close >= _safe(ema20.iloc[-1]) * 0.985 and _safe(ema20.iloc[-1]) > _safe(ema50.iloc[-1]):
        score += 10
        reasons.append("Giá đang áp sát EMA20 trong xu hướng EMA20 > EMA50.")
    else:
        risks.append("Xu hướng EMA chưa ủng hộ.")

    resistance = _safe(high20.iloc[-1])
    if resistance > 0:
        distance = (resistance / close - 1) * 100
        if 0 <= distance <= 2.0:
            score += 16
            reasons.append("Giá nằm sát vùng đỉnh 20 phiên; có thể kích hoạt breakout.")
        elif close > resistance:
            score += 20
            reasons.append("Giá D1 đã vượt đỉnh 20 phiên.")
    if _safe(high55.iloc[-1]) > 0 and close >= _safe(high55.iloc[-1]) * 0.97:
        score += 8
        reasons.append("Giá áp sát vùng đỉnh 55 phiên.")

    vol_base = _safe(avg_vol20.iloc[-1])
    vol_ratio = _safe(v.iloc[-1] / vol_base) if vol_base > 0 else 0
    if vol_ratio >= 1.5:
        score += 15
        reasons.append(f"Khối lượng D1 cao bất thường ({vol_ratio:.2f} lần TB20).")
    elif vol_ratio >= 1.1:
        score += 8
        reasons.append(f"Khối lượng D1 cải thiện ({vol_ratio:.2f} lần TB20).")

    if len(c) >= 6:
        momentum5 = (close / _safe(c.iloc[-6]) - 1) * 100 if _safe(c.iloc[-6]) else 0
        if 0 < momentum5 <= 8:
            score += 8
            reasons.append("Động lượng 5 phiên dương nhưng chưa tăng quá nóng.")
        elif momentum5 > 10:
            risks.append("Đã tăng nhanh 5 phiên; tránh mua đuổi.")

    recent_low = _safe(low20.iloc[-1])
    if recent_low > 0 and (close / recent_low - 1) * 100 <= 4:
        score += 8
        reasons.append("Giá đang ở gần vùng hỗ trợ 20 phiên; theo dõi phản ứng cầu.")

    quote_verified = bool(isinstance(quote, dict) and quote.get("quote_is_fresh") is True)
    live_price = _safe((quote or {}).get("price"))
    if quote_verified and live_price > 0:
        live_change = (live_price / close - 1) * 100
        if -0.5 <= live_change <= 2.5:
            score += 12
            reasons.append("Giá trong phiên giữ được nền, chưa tăng quá xa giá D1.")
        elif live_change < -1.5:
            score -= 12
            risks.append("Giá trong phiên suy yếu rõ so với giá D1.")
        elif live_change > 4:
            score -= 10
            risks.append("Giá tăng nóng trong phiên; nguy cơ mua đuổi.")
    else:
        risks.append("Chưa có quote realtime được xác minh; cảnh báo chỉ dựa trên D1.")

    score = max(0, min(100, int(round(score))))
    if score >= 65 and not any("suy yếu rõ" in x or "tăng nóng" in x for x in risks):
        state = "EARLY_BREAKOUT"
    elif score >= 48:
        state = "PREPARE"
    else:
        state = "NONE"

    return {
        "early_signal": state,
        "early_score": score,
        "early_reasons": reasons,
        "early_risks": risks,
        "early_volume_ratio": round(vol_ratio, 2),
        "early_resistance": round(resistance, 2) if resistance else None,
        "early_quote_verified": quote_verified,
    }
