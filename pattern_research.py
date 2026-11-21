"""Point-in-time event study to discover recurring pre-move patterns.

This is research only: it does not create BUY signals or alter production strategy rules.
Every feature uses information available through the signal-date close. Outcomes begin
at the next session open and include estimated fees, tax and slippage.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

COSTS = {"buy_fee": 0.0015, "sell_fee": 0.0015, "sell_tax": 0.0010, "slippage": 0.0020}
HORIZONS = {"SWING": 5, "REVERSAL": 10, "PULLBACK": 7}
NUMERIC_FEATURES = [
    "ret_1d_pct", "ret_5d_pct", "ret_10d_pct", "ret_20d_pct",
    "dist_ema20_pct", "dist_ema50_pct", "dist_ma200_pct",
    "rsi14", "macd_hist_pct", "volume_ratio20", "turnover20_bn",
    "atr14_pct", "body_pct", "close_location",
    "foreign_net5_ratio_pct",
]


def _safe_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _prepare_ohlcv(frame):
    if frame.attrs.get("_pattern_research_clean"):
        return frame
    d = frame.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d.columns and "datetime" in d.columns:
        d = d.rename(columns={"datetime": "date"})
    required = ["date", "open", "high", "low", "close", "volume"]
    if any(c not in d.columns for c in required):
        return pd.DataFrame()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for col in required[1:]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=required).sort_values("date").drop_duplicates("date", keep="last")
    valid = (
        (d["open"] > 0) & (d["high"] >= d[["open", "close", "low"]].max(axis=1))
        & (d["low"] <= d[["open", "close"]].min(axis=1)) & (d["volume"] >= 0)
    )
    return d.loc[valid].reset_index(drop=True)


def _foreign_asof(flow, date):
    if flow is None or flow.empty:
        return np.nan
    d = flow.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d.columns or "net_val" not in d.columns:
        return np.nan
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["net_val"] = pd.to_numeric(d["net_val"], errors="coerce")
    d = d[(d["date"] <= pd.Timestamp(date)) & d["net_val"].notna()].sort_values("date")
    if len(d) < 5:
        return np.nan
    return float(d["net_val"].tail(5).sum())


def extract_point_in_time_features(frame, index, foreign_flow=None, market_regime="UNKNOWN"):
    """Feature row at index; no observations after index are read."""
    d = _prepare_ohlcv(frame)
    if d.empty or index < 200 or index >= len(d):
        return None
    x = d.iloc[: index + 1].copy()
    close, high, low, volume = x["close"], x["high"], x["low"], x["volume"]
    c = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    if float(loss.iloc[-1] or 0) == 0 and float(gain.iloc[-1] or 0) > 0:
        rsi_value = 100.0
    else:
        rsi_value = _safe_float(rsi.iloc[-1])
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    prev_volume_avg = volume.shift(1).rolling(20).mean().iloc[-1]
    tr = pd.concat([
        high - low, (high - close.shift()).abs(), (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1]
    bar = x.iloc[-1]
    turnover20 = (close * volume).tail(20).mean()
    foreign5 = _foreign_asof(foreign_flow, bar["date"])
    foreign_ratio = foreign5 / float((close * volume).tail(5).sum()) if (
        pd.notna(foreign5) and float((close * volume).tail(5).sum()) > 0
    ) else np.nan
    candle_range = float(bar["high"] - bar["low"])
    return {
        "date": pd.Timestamp(bar["date"]).strftime("%Y-%m-%d"),
        "market_regime": market_regime,
        "ret_1d_pct": (c / float(close.iloc[-2]) - 1) * 100,
        "ret_5d_pct": (c / float(close.iloc[-6]) - 1) * 100,
        "ret_10d_pct": (c / float(close.iloc[-11]) - 1) * 100,
        "ret_20d_pct": (c / float(close.iloc[-21]) - 1) * 100,
        "dist_ema20_pct": (c / ema20 - 1) * 100 if ema20 else np.nan,
        "dist_ema50_pct": (c / ema50 - 1) * 100 if ema50 else np.nan,
        "dist_ma200_pct": (c / ma200 - 1) * 100 if ma200 else np.nan,
        "rsi14": rsi_value,
        "macd_hist_pct": float(macd_hist.iloc[-1]) / c * 100 if c else np.nan,
        "volume_ratio20": float(bar["volume"]) / float(prev_volume_avg) if pd.notna(prev_volume_avg) and prev_volume_avg > 0 else np.nan,
        "turnover20_bn": float(turnover20) / 1_000_000_000 if pd.notna(turnover20) else np.nan,
        "atr14_pct": float(atr) / c * 100 if c else np.nan,
        "body_pct": (float(bar["close"]) - float(bar["open"])) / c * 100 if c else np.nan,
        "close_location": (float(bar["close"]) - float(bar["low"])) / candle_range if candle_range > 0 else 0.5,
        "foreign_net5_ratio_pct": foreign_ratio * 100 if pd.notna(foreign_ratio) else np.nan,
        "above_ema20": bool(c > ema20),
        "ema20_above_ema50": bool(ema20 > ema50),
        "above_ma200": bool(c > ma200),
        "volume_expansion": bool(pd.notna(prev_volume_avg) and prev_volume_avg > 0 and float(bar["volume"]) / float(prev_volume_avg) >= 1.5),
        "foreign_net_buy_5d": bool(pd.notna(foreign5) and foreign5 > 0),
    }


def label_forward_outcome(frame, index, horizon, costs=None):
    """Label only after feature date; next open entry, horizon close exit."""
    d = _prepare_ohlcv(frame)
    costs = {**COSTS, **(costs or {})}
    entry_idx, exit_idx = index + 1, index + horizon
    if d.empty or exit_idx >= len(d):
        return None
    entry_open = float(d.iloc[entry_idx]["open"])
    exit_close = float(d.iloc[exit_idx]["close"])
    if entry_open <= 0:
        return None
    buy = entry_open * (1 + costs["slippage"]) * (1 + costs["buy_fee"])
    sell = exit_close * (1 - costs["slippage"]) * (1 - costs["sell_fee"] - costs["sell_tax"])
    net_pct = (sell / buy - 1) * 100
    future = d.iloc[entry_idx: exit_idx + 1]
    mfe_pct = (float(future["high"].max()) / entry_open - 1) * 100
    mae_pct = (float(future["low"].min()) / entry_open - 1) * 100
    return {
        "entry_date": pd.Timestamp(d.iloc[entry_idx]["date"]).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(d.iloc[exit_idx]["date"]).strftime("%Y-%m-%d"),
        "entry_open": entry_open,
        "exit_close": exit_close,
        "net_return_pct": net_pct,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "positive": bool(net_pct > 0),
        "quality_win": bool(net_pct >= 3.0 and mae_pct > -5.0),
    }


def _metrics(group):
    r = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
    wins, losses = r[r > 0], r[r < 0]
    gross_loss = abs(float(losses.sum()))
    return {
        "samples": int(len(r)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 2) if len(r) else None,
        "quality_win_rate_pct": round(float(group["quality_win"].mean() * 100), 2) if len(r) else None,
        "avg_net_return_pct": round(float(r.mean()), 3) if len(r) else None,
        "median_net_return_pct": round(float(r.median()), 3) if len(r) else None,
        "profit_factor": round(float(wins.sum()) / gross_loss, 3) if gross_loss else None,
        "avg_mfe_pct": round(float(group["mfe_pct"].mean()), 3) if len(r) else None,
        "avg_mae_pct": round(float(group["mae_pct"].mean()), 3) if len(r) else None,
    }


def analyze_patterns(events, min_bucket_samples=30):
    """Univariate bins + a small, interpretable confluence scan; research only."""
    if events is None or events.empty:
        return {"by_strategy": {}, "feature_relationships": [], "confluence_patterns": []}
    output = {}
    relationships = []
    confluences = []
    for strategy, frame in events.groupby("strategy"):
        output[strategy] = _metrics(frame)
        for feature in NUMERIC_FEATURES:
            valid = frame[[feature, "net_return_pct", "quality_win", "mfe_pct", "mae_pct"]].dropna()
            if len(valid) < min_bucket_samples * 3 or valid[feature].nunique() < 4:
                continue
            try:
                valid = valid.copy()
                valid["bucket"] = pd.qcut(valid[feature], q=5, duplicates="drop")
            except ValueError:
                continue
            for bucket, group in valid.groupby("bucket", observed=True):
                if len(group) < min_bucket_samples:
                    continue
                relationships.append({
                    "strategy": strategy, "feature": feature,
                    "bucket": str(bucket), **_metrics(group),
                })
        # Test a limited set of intuitive, predeclared combinations to avoid
        # unconstrained data-mining across hundreds of arbitrary combinations.
        masks = {
            "xu_huong_tang": frame["above_ema20"] & frame["ema20_above_ema50"],
            "xu_huong_tang_va_volume": frame["above_ema20"] & frame["ema20_above_ema50"] & frame["volume_expansion"],
            "xu_huong_tang_va_khoi_ngoai_mua": frame["above_ema20"] & frame["ema20_above_ema50"] & frame["foreign_net_buy_5d"],
            "volume_expansion_va_close_manh": frame["volume_expansion"] & (frame["close_location"] >= 0.7),
            "market_bullish_va_trend_bullish": (frame["market_regime"] == "BULLISH") & frame["above_ema20"] & frame["ema20_above_ema50"],
        }
        for name, mask in masks.items():
            group = frame.loc[mask]
            if len(group) >= min_bucket_samples:
                confluences.append({"strategy": strategy, "pattern": name, **_metrics(group)})
    relationships.sort(key=lambda x: (x["strategy"], -(x["quality_win_rate_pct"] or 0), -x["samples"]))
    confluences.sort(key=lambda x: (x["strategy"], -(x["avg_net_return_pct"] or -999), -x["samples"]))
    return {
        "by_strategy": output,
        "feature_relationships": relationships,
        "confluence_patterns": confluences,
        "interpretation_rules": [
            "Các bảng mô tả mối liên hệ trong mẫu lịch sử, không chứng minh quan hệ nhân quả.",
            "Mẫu có ít hơn 30 quan sát không được xếp hạng để tránh kết luận từ mẫu quá nhỏ.",
            "Các ngày tín hiệu liền kề có thể chồng lấn cửa sổ lợi nhuận; số mẫu không đồng nghĩa số giao dịch độc lập.",
            "Chỉ đưa điều kiện vào chiến lược sau khi xác nhận trên giai đoạn ngoài mẫu và nhiều trạng thái thị trường.",
        ],
    }
