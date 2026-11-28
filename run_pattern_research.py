"""Run point-in-time pattern discovery over liquid current HOSE/HNX names.

Research only. Selection uses recent turnover; features/outcomes remain point-in-time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from data_provider import fetch_exchange_symbols, fetch_stock_history
from foreign_flow_provider import fetch_foreign_flow
from adaptive_strategy import classify_market_regime
from pattern_research import HORIZONS, walk_forward_validation, extract_point_in_time_features, label_forward_outcome

OUT = Path("pattern_research_output")
OUT.mkdir(exist_ok=True)

MIN_HISTORY_SESSIONS = 200
MIN_AVG_TURNOVER_20D = 5_000_000_000
MIN_MEDIAN_TURNOVER_20D = 3_000_000_000


def clean(raw):
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d.columns and "Date" in raw.columns:
        d["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = (
        d.dropna(subset=["date", "open", "high", "low", "close", "volume"])
        .sort_values("date").drop_duplicates("date").reset_index(drop=True)
    )
    d.attrs["_pattern_research_clean"] = True
    return d


def _config():
    max_symbols = int(os.getenv("PATTERN_RESEARCH_MAX_SYMBOLS", "30"))
    lookback = int(os.getenv("PATTERN_RESEARCH_LOOKBACK", "420"))
    # Keep UI/CLI bounded to prevent accidental provider overload.
    max_symbols = max(1, min(max_symbols, 120))
    lookback = max(260, min(lookback, 800))
    return max_symbols, lookback


def main():
    max_symbols, lookback = _config()
    horizon_max = max(HORIZONS.values())
    # Request a full event window plus indicator warm-up and exit-label bars.
    history_length = lookback + MIN_HISTORY_SESSIONS + horizon_max
    symbols = fetch_exchange_symbols()
    market = clean(fetch_stock_history("VNINDEX", length=history_length, is_index=True))
    histories, errors, ranks, rejected = {}, [], [], []

    for i, symbol in enumerate(symbols, 1):
        try:
            d = clean(fetch_stock_history(symbol, length=history_length))
            if len(d) < MIN_HISTORY_SESSIONS + 30:
                rejected.append({"symbol": symbol, "reason": f"Thiếu lịch sử: {len(d)} phiên"})
                continue

            # Universe liquidity filter is based on the latest completed D1 data.
            turnover = (d["close"] * d["volume"]).tail(20)
            avg_turnover = float(turnover.mean())
            median_turnover = float(turnover.median())
            if avg_turnover < MIN_AVG_TURNOVER_20D or median_turnover < MIN_MEDIAN_TURNOVER_20D:
                rejected.append({
                    "symbol": symbol,
                    "reason": "Không đạt bộ lọc thanh khoản 20 phiên",
                    "avg_turnover_20d": avg_turnover,
                    "median_turnover_20d": median_turnover,
                })
                continue
            ranks.append((symbol, avg_turnover, median_turnover))
            histories[symbol] = d
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:250]})
        if i % 25 == 0:
            print(f"Đã tải lịch sử {i}/{len(symbols)} mã")

    selected = [s for s, _, _ in sorted(ranks, key=lambda x: (x[1], x[2]), reverse=True)[:max_symbols]]
    rows = []
    for n, symbol in enumerate(selected, 1):
        d = histories[symbol]
        try:
            foreign = fetch_foreign_flow(symbol, length=history_length)
        except Exception as exc:
            foreign = pd.DataFrame()
            errors.append({"symbol": symbol, "error": f"Foreign flow: {str(exc)[:200]}"})

        # lookback means the number of signal dates studied; 200 sessions are
        # retained as indicator warm-up, with future horizon bars for labels.
        start = max(MIN_HISTORY_SESSIONS, len(d) - lookback - horizon_max)
        last_signal_idx = len(d) - horizon_max
        for idx in range(start, last_signal_idx):
            asof = d.iloc[idx]["date"]
            market_asof = market.loc[market["date"] <= asof]
            regime = classify_market_regime(market_asof)
            features = extract_point_in_time_features(d, idx, foreign, regime)
            if not features:
                continue
            for strategy, horizon in HORIZONS.items():
                outcome = label_forward_outcome(d, idx, horizon)
                if outcome is None:
                    continue
                rows.append({"symbol": symbol, "strategy": strategy, **features, **outcome})
        if n % 10 == 0:
            print(f"Đã nghiên cứu {n}/{len(selected)} mã; số quan sát: {len(rows)}")

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError(
            "Không tạo được quan sát nghiên cứu. Kiểm tra nguồn D1, bộ lọc thanh khoản "
            "và số phiên lịch sử trả về."
        )
    events.to_csv(OUT / "pattern_events.csv", index=False)
    report = walk_forward_validation(events)
    report.update({
        "universe_discovered": len(symbols),
        "symbols_with_history": len(histories),
        "symbols_passed_liquidity": len(ranks),
        "symbols_tested": len(selected),
        "symbols_rejected_liquidity_or_history": len(rejected),
        "rejected_sample": rejected[:60],
        "data_errors_count": len(errors),
        "data_errors_sample": errors[:40],
        "observations": len(events),
        "data_start": str(events["date"].min()),
        "data_end": str(events["date"].max()),
        "requested_lookback_sessions": lookback,
        "actual_signal_dates_per_symbol_max": lookback,
        "minimum_history_sessions": MIN_HISTORY_SESSIONS,
        "liquidity_filter": {
            "average_turnover_20d_min_vnd": MIN_AVG_TURNOVER_20D,
            "median_turnover_20d_min_vnd": MIN_MEDIAN_TURNOVER_20D,
        },
        "method": (
            "Point-in-time event study: D1 features through signal-date close; entry next "
            "session open; exit after strategy horizon (Swing 5, Pullback 7, Reversal 10 "
            "sessions); buy/sell fees, sell tax and 0.20% slippage each side. "
            "MFE/MAE from post-entry bars. No production BUY signals generated."
        ),
        "costs": {"buy_fee": 0.0015, "sell_fee": 0.0015, "sell_tax": 0.001, "slippage_each_side": 0.002},
        "universe_warning": (
            "Universe is based on currently listed HOSE/HNX stocks and recent liquidity, "
            "not a historical point-in-time universe. Delisted stocks are absent "
            "(survivorship bias); current turnover ranking is for research coverage only."
        ),
        "foreign_flow_warning": (
            "Foreign-flow history may be shorter than price history depending on provider. "
            "Missing values remain UNKNOWN/NaN and are not interpreted as net buying."
        ),
    })
    (OUT / "pattern_research_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "universe_discovered": report["universe_discovered"],
        "symbols_passed_liquidity": report["symbols_passed_liquidity"],
        "symbols_tested": report["symbols_tested"],
        "observations": report["observations"],
        "data_start": report["data_start"],
        "data_end": report["data_end"],
        "walk_forward_status": report["status"],
        "discovery_cutoff": report.get("discovery_cutoff"),
        "training_observations": report.get("training_observations"),
        "holdout_observations": report.get("holdout_observations"),
        "holdout_by_strategy": report.get("holdout_by_strategy"),
        "pattern_holdout": report.get("pattern_holdout"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
