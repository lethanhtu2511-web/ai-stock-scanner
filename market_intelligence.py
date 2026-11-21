"""Evidence-aware market and sector intelligence helpers.

These functions only calculate from caller-supplied, timestamp-aligned data.
They do not guess sector membership or fill missing market breadth.
"""
from __future__ import annotations

import pandas as pd


def _clean_close(frame: pd.DataFrame) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(dtype=float)
    d = frame.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "close" not in d:
        return pd.Series(dtype=float)
    if "date" in d:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return pd.to_numeric(d["close"], errors="coerce").dropna()


def calculate_market_breadth(histories: dict[str, pd.DataFrame], as_of=None) -> dict:
    """Calculate equal-weight breadth from supplied completed D1 histories."""
    above20 = above50 = advancing = declining = 0
    returns = []
    valid = 0
    for frame in (histories or {}).values():
        d = frame.copy()
        d.columns = [str(c).strip().lower() for c in d.columns]
        if "date" in d and as_of is not None:
            dates = pd.to_datetime(d["date"], errors="coerce")
            d = d.loc[dates <= pd.Timestamp(as_of)]
        close = _clean_close(d)
        if len(close) < 51:
            continue
        valid += 1
        last = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        prior = float(close.iloc[-2])
        above20 += last > ma20
        above50 += last > ma50
        advancing += last > prior
        declining += last < prior
        returns.append(last / float(close.iloc[-6]) - 1 if float(close.iloc[-6]) else 0)
    if not valid:
        return {"status": "UNAVAILABLE", "sample_size": 0}
    pct20, pct50 = above20 / valid * 100, above50 / valid * 100
    ad_ratio = advancing / declining if declining else (float("inf") if advancing else None)
    if pct20 >= 65 and pct50 >= 55:
        regime = "BROAD_BULLISH"
    elif pct20 <= 35 and pct50 <= 40:
        regime = "BROAD_BEARISH"
    else:
        regime = "MIXED"
    return {
        "status": "AVAILABLE", "sample_size": valid,
        "above_ema20_pct": round(pct20, 2),
        "above_ema50_pct": round(pct50, 2),
        "advancing": advancing, "declining": declining,
        "advance_decline_ratio": round(ad_ratio, 2) if ad_ratio is not None and pd.notna(ad_ratio) else None,
        "average_5d_return_pct": round(sum(returns) / len(returns) * 100, 2),
        "regime": regime,
        "note": "Equal-weight breadth of supplied symbols only; not the full exchange unless the input universe is complete."
    }


def calculate_sector_strength(histories: dict[str, pd.DataFrame], symbol_to_sector: dict[str, str], as_of=None) -> dict:
    """Rank sectors by equal-weight 20-session return; mapping must be explicit."""
    grouped: dict[str, list[float]] = {}
    for symbol, frame in (histories or {}).items():
        sector = (symbol_to_sector or {}).get(str(symbol).upper())
        if not sector:
            continue
        d = frame.copy()
        d.columns = [str(c).strip().lower() for c in d.columns]
        if "date" in d and as_of is not None:
            dates = pd.to_datetime(d["date"], errors="coerce")
            d = d.loc[dates <= pd.Timestamp(as_of)]
        close = _clean_close(d)
        if len(close) < 21 or float(close.iloc[-21]) <= 0:
            continue
        grouped.setdefault(str(sector), []).append(float(close.iloc[-1] / close.iloc[-21] - 1))
    if not grouped:
        return {"status": "UNAVAILABLE", "sectors": []}
    sectors = [
        {"sector": name, "symbols": len(values),
         "return_20d_pct": round(sum(values) / len(values) * 100, 2)}
        for name, values in grouped.items()
    ]
    sectors.sort(key=lambda x: x["return_20d_pct"], reverse=True)
    return {"status": "AVAILABLE", "sectors": sectors,
            "note": "Equal-weight return of mapped symbols; sectors without an explicit mapping are excluded."}
