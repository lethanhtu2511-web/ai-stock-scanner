"""Settled foreign-investor flow adapter from the existing vn-market-data stack."""
from __future__ import annotations

import time
import pandas as pd

_CACHE = {}
CACHE_SECONDS = 900


def _num(value):
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0


def normalize_foreign_flow(raw):
    if raw is None:
        return pd.DataFrame()
    d = pd.DataFrame(raw).copy()
    if d.empty:
        return pd.DataFrame()
    d.columns = [str(c).strip().lower() for c in d.columns]
    aliases = {
        "date": "time", "trading_date": "time", "tradingdate": "time",
        "buy_volume": "buy_vol", "buy_value": "buy_val",
        "sell_volume": "sell_vol", "sell_value": "sell_val",
        "net_volume": "net_vol", "net_value": "net_val",
        "foreign_buy_volume": "buy_vol", "foreign_buy_value": "buy_val",
        "foreign_sell_volume": "sell_vol", "foreign_sell_value": "sell_val",
        "foreign_net_volume": "net_vol", "foreign_net_value": "net_val",
        "fr_buy_volume": "buy_vol", "fr_buy_value": "buy_val",
        "fr_sell_volume": "sell_vol", "fr_sell_value": "sell_val",
        "fr_net_volume": "net_vol", "fr_net_value": "net_val",
    }
    for old, new in aliases.items():
        if old in d.columns and new not in d.columns:
            d = d.rename(columns={old: new})
    if "time" not in d.columns:
        return pd.DataFrame()
    parsed = pd.to_datetime(d["time"], errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
    d["date"] = parsed.dt.normalize()
    for col in ("buy_vol", "buy_val", "sell_vol", "sell_val", "net_vol", "net_val"):
        if col not in d.columns:
            d[col] = pd.NA
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if d["net_val"].isna().all() and d["buy_val"].notna().any() and d["sell_val"].notna().any():
        d["net_val"] = d["buy_val"] - d["sell_val"]
    if d["net_vol"].isna().all() and d["buy_vol"].notna().any() and d["sell_vol"].notna().any():
        d["net_vol"] = d["buy_vol"] - d["sell_vol"]
    d = d.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    return d[["date", "buy_vol", "buy_val", "sell_vol", "sell_val", "net_vol", "net_val"]].reset_index(drop=True)


def fetch_foreign_flow(symbol: str, start: str | None = None, end: str | None = None, length: int = 120, force_refresh: bool = False):
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()
    if not end:
        end = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").strftime("%Y-%m-%d")
    if not start:
        # CafeF settled archive is approximately 63 sessions deep.
        start = (pd.Timestamp(end) - pd.Timedelta(days=max(120, int(length) * 2))).strftime("%Y-%m-%d")
    cache_key = (symbol, start, end)
    cached = _CACHE.get(cache_key)
    if not force_refresh and cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1].copy()
    try:
        from vn_market_data import get_foreign_history
        response = get_foreign_history(symbol, start, end)
        source, rows = response if isinstance(response, tuple) and len(response) == 2 else ("vn-market-data", response)
        result = normalize_foreign_flow(rows)
        result.attrs["source"] = str(source)
        if not result.empty:
            _CACHE[cache_key] = (time.monotonic(), result.copy())
            return result
        raise RuntimeError(f"{source}: không có dữ liệu trong {start} đến {end}")
    except Exception as exc:
        raise RuntimeError(f"Không lấy được dữ liệu khối ngoại: {exc}") from exc


def summarize_foreign_flow(flow: pd.DataFrame, as_of=None, market_turnover_5d=None) -> dict:
    empty = {
        "foreign_flow_status": "UNAVAILABLE",
        "foreign_buy_value": None, "foreign_sell_value": None,
        "foreign_net_value": None, "foreign_buy_volume": None,
        "foreign_sell_volume": None, "foreign_net_volume": None,
        "foreign_net_3d": None, "foreign_net_5d": None,
        "foreign_net_10d": None, "foreign_net_ratio": None,
        "foreign_signal": "UNKNOWN", "foreign_score_adjustment": 0,
        "foreign_data_date": None,
    }
    if flow is None or flow.empty:
        return empty
    d = normalize_foreign_flow(flow)
    if as_of is not None:
        d = d[d["date"] <= pd.Timestamp(as_of).normalize()]
    if d.empty:
        return empty
    row = d.iloc[-1]
    net = d["net_val"]
    net3 = float(net.tail(3).sum()) if net.tail(3).notna().all() else None
    net5 = float(net.tail(5).sum()) if net.tail(5).notna().all() else None
    net10 = float(net.tail(10).sum()) if net.tail(10).notna().all() else None
    ratio = net5 / float(market_turnover_5d) if (
        net5 is not None and market_turnover_5d and float(market_turnover_5d) > 0
    ) else None
    score = 0
    if ratio is not None:
        if ratio >= 0.02:
            score = 6
        elif ratio >= 0.005:
            score = 3
        elif ratio <= -0.02:
            score = -6
        elif ratio <= -0.005:
            score = -3
    elif net5 is not None:
        score = 3 if net5 > 0 else (-3 if net5 < 0 else 0)
    return {
        "foreign_flow_status": "AVAILABLE",
        "foreign_buy_value": _num(row["buy_val"]),
        "foreign_sell_value": _num(row["sell_val"]),
        "foreign_net_value": _num(row["net_val"]),
        "foreign_buy_volume": _num(row["buy_vol"]),
        "foreign_sell_volume": _num(row["sell_vol"]),
        "foreign_net_volume": _num(row["net_vol"]),
        "foreign_net_3d": net3, "foreign_net_5d": net5,
        "foreign_net_10d": net10, "foreign_net_ratio": ratio,
        "foreign_signal": "ACCUMULATING" if score > 0 else ("DISTRIBUTING" if score < 0 else "NEUTRAL"),
        "foreign_score_adjustment": score,
        "foreign_data_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
    }
