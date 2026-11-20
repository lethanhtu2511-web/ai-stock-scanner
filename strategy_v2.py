"""Experimental multi-setup daily strategy from the user specification.

Research-only: this module emits candidates and risk plans, never broker orders.
All signal decisions use the last completed daily bar; entries are next-session
open (Pullback/Swing) or a stop trigger above the confirmation candle (Reversal).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

import numpy as np
import pandas as pd

SETUP_NAMES = ("PULLBACK", "SWING_BREAKOUT", "REVERSAL")
FEE_RATE = 0.0015
SELL_TAX = 0.001
SLIPPAGE_HIGH = 0.001
SLIPPAGE_LOW = 0.0025


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dữ liệu phải là DataFrame.")
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {', '.join(sorted(missing))}")
    for c in required:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "date" in d:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    d = d.dropna(subset=list(required)).reset_index(drop=True)
    valid = ((d.open > 0) & (d.close > 0) & (d.high >= d[["open", "close", "low"]].max(axis=1))
             & (d.low <= d[["open", "close"]].min(axis=1)) & (d.volume >= 0))
    return d.loc[valid].reset_index(drop=True)


def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = _clean(df)
    c, h, l, v = d.close, d.high, d.low, d.volume
    for n in (18, 21):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    for n in (50, 150, 200):
        d[f"sma{n}"] = c.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - 100/(1+rs)).fillna(50)
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d["vol20"] = v.shift(1).rolling(20).mean()
    d["vol_ratio"] = v / d.vol20.replace(0, np.nan)
    # Wilder-style ADX
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_s = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd_hist"] = (ema12-ema26) - (ema12-ema26).ewm(span=9, adjust=False).mean()
    d["macd_hist_prev"] = d.macd_hist.shift(1)
    d["sma50_prev"] = d.sma50.shift(5)
    d["sma150_prev"] = d.sma150.shift(5)
    d["sma200_prev"] = d.sma200.shift(5)
    d["high22_prev"] = h.shift(1).rolling(22).max()
    d["low5_prev"] = l.shift(1).rolling(5).min()
    d["high20_prev"] = h.shift(1).rolling(20).max()
    d["low20_prev"] = l.shift(1).rolling(20).min()
    d["turnover"] = c * v
    d["avg_turnover20"] = d.turnover.shift(1).rolling(20).mean()
    d["avg_volume20"] = v.shift(1).rolling(20).mean()
    return d.replace([np.inf, -np.inf], np.nan)


def market_regime(market_df: pd.DataFrame | None) -> str:
    if market_df is None:
        return "UNKNOWN"
    d = _indicators(market_df)
    if len(d) < 200 or pd.isna(d.iloc[-1][["sma50", "sma200"]]).any():
        return "UNKNOWN"
    r = d.iloc[-1]
    if r.close < r.sma200:
        return "BEARISH"
    if r.close > r.sma200 and r.sma50 > r.sma200:
        return "BULLISH"
    if r.close > r.sma200 and r.sma50 < r.sma200:
        return "NEUTRAL"
    return "NEUTRAL"


def _universe_check_indicators(d: pd.DataFrame, *, suspended: bool = False, controlled: bool = False) -> tuple[bool, list[str]]:
    reasons = []
    if len(d) < 250:
        reasons.append("Lịch sử dưới 250 phiên")
    if d.empty:
        return False, reasons + ["Không có dữ liệu"]
    r = d.iloc[-1]
    if r.close < 10_000:
        reasons.append("Giá dưới 10.000 đồng")
    if pd.isna(r.avg_volume20) or r.avg_volume20 < 500_000:
        reasons.append("Khối lượng bình quân 20 phiên dưới 500.000 cp")
    if pd.isna(r.avg_turnover20) or r.avg_turnover20 < 10_000_000_000:
        reasons.append("GTGD bình quân 20 phiên dưới 10 tỷ đồng")
    if suspended:
        reasons.append("Mã đang bị đình chỉ")
    if controlled:
        reasons.append("Mã thuộc diện kiểm soát")
    return not reasons, reasons


def universe_check(df: pd.DataFrame, *, suspended: bool = False, controlled: bool = False) -> tuple[bool, list[str]]:
    return _universe_check_indicators(_indicators(df), suspended=suspended, controlled=controlled)


def trend_filter(r: pd.Series) -> bool:
    vals = [r.get(k, np.nan) for k in ("sma50", "sma150", "sma200", "sma150_prev", "sma200_prev")]
    if any(pd.isna(x) for x in vals):
        return False
    return bool(r.close > r.sma150 and r.sma150 > r.sma150_prev
                and r.sma50 > r.sma200 and r.sma200 >= r.sma200_prev)


def _near(price: float, level: float, atr: float, mult: float) -> bool:
    return atr > 0 and abs(price-level) <= mult*atr


def _pivot_divergence(d: pd.DataFrame) -> bool:
    # Compare two confirmed local lows separated by at least 5 bars.
    if len(d) < 20:
        return False
    lows = d.low.to_numpy()
    rsi = d.rsi.to_numpy()
    pivots = [i for i in range(2, len(d)-2) if lows[i] <= min(lows[i-2:i+3])]
    if len(pivots) < 2:
        return False
    a, b = pivots[-2], pivots[-1]
    return bool(b-a >= 5 and lows[b] < lows[a] and rsi[b] > rsi[a])


def _candle_reversal(r: pd.Series, prev: pd.Series) -> bool:
    body = abs(r.close-r.open)
    rng = max(r.high-r.low, 1e-9)
    lower_wick = min(r.open, r.close)-r.low
    hammer = lower_wick >= 2*max(body, rng*0.05) and (r.close-r.low)/rng >= 0.6
    engulf = r.close > r.open and prev.close < prev.open and r.open <= prev.close and r.close >= prev.open
    piercing = r.close > r.open and prev.close < prev.open and r.open < prev.low and r.close > (prev.open+prev.close)/2
    return bool(hammer or engulf or piercing)


def _support_zone(d: pd.DataFrame, r: pd.Series) -> bool:
    # Structural support approximated by prior 20/60-session lows or 0.618-0.786 retracement.
    recent = d.iloc[-61:-1]
    if recent.empty:
        return False
    near_old_low = min(abs(r.close-float(recent.low.min())), abs(r.close-float(recent.close.min()))) <= 1.5*r.atr
    wave = d.iloc[-120:-1]
    if wave.empty:
        return bool(near_old_low)
    swing_low, swing_high = float(wave.low.min()), float(wave.high.max())
    retrace_618 = swing_high - 0.618*(swing_high-swing_low)
    retrace_786 = swing_high - 0.786*(swing_high-swing_low)
    fib = retrace_786 <= r.close <= retrace_618
    return bool(near_old_low or fib)


def _risk_plan(entry: float, stop: float, atr: float, *, setup: str, liquidity_high: bool = True) -> dict[str, Any]:
    risk = entry-stop
    if entry <= 0 or stop <= 0 or risk <= 0:
        return {"valid": False, "reason": "Stop không hợp lệ"}
    r_mult = 1.5 if setup == "REVERSAL" else 2.0
    return {
        "valid": True, "entry_reference": round(entry, 2), "stop_loss": round(stop, 2),
        "risk_per_share": round(risk, 2), "R": round(risk, 2),
        "target_1": round(entry+r_mult*risk, 2), "target_1_R": r_mult,
        "chandelier_initial": round(entry-3*atr, 2),
        "slippage_rate": SLIPPAGE_HIGH if liquidity_high else SLIPPAGE_LOW,
        "fee_buy": FEE_RATE, "fee_sell": FEE_RATE, "sell_tax": SELL_TAX,
    }


def analyze_daily_setup(symbol: str, df: pd.DataFrame, market_df: pd.DataFrame | None,
                        *, suspended: bool = False, controlled: bool = False,
                        sector: str | None = None, liquidity_high: bool = True,
                        indicators_precomputed: bool = False,
                        market_regime_override: str | None = None) -> dict[str, Any]:
    d = df.copy() if indicators_precomputed else _indicators(df)
    if len(d) < 250:
        return {"symbol": symbol, "candidates": [], "blocked": ["Cần tối thiểu 250 phiên"]}
    r, prev = d.iloc[-1], d.iloc[-2]
    regime = market_regime_override if market_regime_override is not None else market_regime(market_df)
    eligible, universe_blockers = _universe_check_indicators(d, suspended=suspended, controlled=controlled)
    out = {"symbol": str(symbol).upper(), "market_regime": regime, "sector": sector,
           "universe_eligible": eligible, "universe_blockers": universe_blockers, "candidates": []}
    if not eligible or regime in ("BEARISH", "UNKNOWN"):
        out["blocked"] = universe_blockers + [f"Không mở lệnh mới trong regime {regime}"]
        return out
    trend_ok = trend_filter(r)
    risk_factor = 1.0 if regime == "BULLISH" else 0.6
    if regime == "NEUTRAL":
        allowed = {"PULLBACK", "SWING_BREAKOUT"}
    else:
        allowed = set(SETUP_NAMES)

    # Pullback: score out of 8, strict trend and ADX prerequisite.
    if "PULLBACK" in allowed:
        score, why = 0, []
        pullback = (r.close < r.high22_prev and (r.close <= r.ema18+0.5*r.atr)
                    and (r.close >= r.ema21-0.5*r.atr))
        if trend_ok and r.adx > 20 and pullback:
            if _near(r.close, r.ema18, r.atr, 0.5) or _near(r.close, r.ema21, r.atr, 0.5):
                score += 2; why.append("Giá trong vùng ±0,5 ATR quanh EMA18/21")
            elif _near(r.close, r.sma50, r.atr, 1.0):
                score += 1; why.append("Giá trong vùng ±1 ATR quanh SMA50")
            if r.close > r.open and r.close > r.ema18:
                score += 2; why.append("Nến xanh đóng trên EMA18")
            if r.vol_ratio >= 1.2:
                score += 2; why.append("Khối lượng >=1,2 lần trung bình")
            if 40 <= r.rsi <= 65:
                score += 1; why.append("RSI trong vùng 40–65")
            reclaim = ((prev.close < prev.ema18 and r.close > r.ema18)
                       or (prev.close < prev.sma50 and r.close > r.sma50))
            if reclaim:
                score += 1; why.append("Rũ bỏ rồi lấy lại đường trung bình")
            if score >= 5:
                entry = float(r.close)
                stop = min(float(d.low.iloc[-5:].min()), entry-1.5*float(r.atr))-0.2*float(r.atr)
                plan = _risk_plan(entry, stop, float(r.atr), setup="PULLBACK", liquidity_high=liquidity_high)
                out["candidates"].append({"setup":"PULLBACK","score":score,"max_score":8,
                    "priority_note":"Biến thể reclaim được ưu tiên trong ngày" if reclaim else "",
                    "reasons":why,"entry_type":"NEXT_OPEN","gap_limit_atr":1.0,"atr":float(r.atr),
                    "holding_sessions":[3,20],"risk_factor":risk_factor,"risk_plan":plan,
                    "exit_rule":"50% tại 2R; phần còn lại Chandelier 22 phiên - 3 ATR; thoát tối đa 20 phiên nếu chưa đạt 1R và mất EMA21"})
    # Swing breakout: base range and volume contraction.
    if "SWING_BREAKOUT" in allowed and trend_ok:
        base = d.iloc[-26:-1]
        if len(base) >= 15:
            base_high, base_low = float(base.high.max()), float(base.low.min())
            base_range = (base_high/base_low-1) if base_low > 0 else 99
            prior_vol = float(d.volume.iloc[-51:-26].mean()) if len(d) >= 51 else np.nan
            contracting = bool(pd.notna(prior_vol) and base.volume.mean() < prior_vol)
            breakout = r.close > base_high
            score, why = 0, []
            if breakout: score += 2; why.append("Đóng cửa vượt đỉnh nền")
            if r.vol_ratio >= 1.5: score += 2; why.append("Khối lượng breakout >=1,5 lần")
            if len(base) >= 25: score += 1; why.append("Nền tích lũy từ 5 tuần")
            if abs(r.close-r.sma50) <= 2*r.atr: score += 1; why.append("Giá cách SMA50 không quá 2 ATR")
            if 55 <= r.rsi <= 75: score += 1; why.append("RSI trong vùng 55–75")
            if base_range <= 0.12 and contracting and breakout and score >= 5:
                entry = float(r.close)
                stop = min(base_low, entry-2*float(r.atr))
                plan = _risk_plan(entry, stop, float(r.atr), setup="SWING_BREAKOUT", liquidity_high=liquidity_high)
                out["candidates"].append({"setup":"SWING_BREAKOUT","score":score,"max_score":7,
                    "reasons":why+["Nền <=12%, khối lượng co hẹp"],"base_range_pct":round(base_range*100,2),
                    "entry_type":"NEXT_OPEN","gap_limit_atr":1.5,"atr":float(r.atr),"holding_sessions":[3,30],"base_low":base_low,
                    "risk_factor":risk_factor,"risk_plan":plan,
                    "exit_rule":"1/3 tại 2R, 1/3 tại 3R hoặc swing high; phần còn lại Chandelier; tối đa 30 phiên"})
    # Reversal: separate, higher-risk satellite; no trend filter by design.
    if "REVERSAL" in allowed:
        decline = float(r.close / d.close.iloc[-61:-30].max()-1) if len(d) >= 61 else 0.0
        candle = _candle_reversal(r, prev) and _support_zone(d, r)
        divergence = _pivot_divergence(d)
        macd_up = bool((d.macd_hist.iloc[-3:] > d.macd_hist.iloc[-3:].shift(1)).sum() >= 2)
        macd_turn = bool(r.macd_hist > 0 or macd_up)
        score, why = 0, []
        if decline <= -0.15:
            if divergence: score += 2; why.append("RSI phân kỳ tăng")
            if candle: score += 2; why.append("Nến đảo chiều tại hỗ trợ cấu trúc")
            if macd_turn: score += 1; why.append("MACD histogram chuyển dương/đang dốc lên")
            if r.vol_ratio >= 1.5: score += 1; why.append("Khối lượng xác nhận >=1,5 lần")
            structural = candle
            momentum = divergence or macd_turn
            if score >= 4 and structural and momentum:
                trigger = float(r.high)
                stop = float(r.low)-0.5*float(r.atr)
                plan = _risk_plan(trigger, stop, float(r.atr), setup="REVERSAL", liquidity_high=liquidity_high)
                out["candidates"].append({"setup":"REVERSAL","score":score,"max_score":6,
                    "reasons":why,"prior_decline_pct":round(decline*100,2),
                    "entry_type":"STOP_ABOVE_CONFIRMATION_HIGH","entry_trigger":trigger,
                    "cancel_if_close_below_confirmation_low":float(r.low),
                    "confirmation_low":float(r.low),
                    "holding_sessions":[3,10],"risk_factor":risk_factor*0.5,
                    "risk_plan":plan,"exit_rule":"50% tại 1,5R; phần còn lại trail dưới EMA21; thoát tối đa 10 phiên nếu không bật"})
    out["candidates"].sort(key=lambda x: (x["score"]/max(x["max_score"],1), x["risk_factor"]), reverse=True)
    out["best_candidate"] = out["candidates"][0] if out["candidates"] else None
    return out


def position_size(*, capital: float, entry: float, stop: float, market_factor: float = 1.0,
                  drawdown_factor: float = 1.0, consecutive_loss_factor: float = 1.0,
                  avg_turnover: float | None = None, max_position_pct: float = 0.10,
                  risk_pct: float = 0.0075, lot_size: int = 100) -> dict[str, Any]:
    """Risk-based sizing with 1% hard cap, 10% position cap and 1% ADV cap."""
    if capital <= 0 or entry <= 0 or stop <= 0 or entry <= stop:
        return {"valid": False, "quantity": 0, "reason": "Thông số vốn/entry/stop không hợp lệ"}
    effective_risk = min(0.01, risk_pct) * max(0, market_factor) * max(0, drawdown_factor) * max(0, consecutive_loss_factor)
    risk_cash = capital * effective_risk
    qty_by_risk = floor(risk_cash/(entry-stop))
    qty_by_position = floor((capital*max_position_pct)/entry)
    qty = min(qty_by_risk, qty_by_position)
    if avg_turnover and avg_turnover > 0:
        qty = min(qty, floor((avg_turnover*0.01)/entry))
    qty = max(0, (qty//lot_size)*lot_size)
    return {"valid": qty >= lot_size, "quantity": qty, "risk_cash": round(risk_cash,2),
            "effective_risk_pct": round(effective_risk*100,4),
            "position_value": round(qty*entry,2),
            "reason": "" if qty >= lot_size else "Không đủ điều kiện tối thiểu một lô 100 cổ phiếu"}
