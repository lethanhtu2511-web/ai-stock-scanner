"""Run point-in-time pattern discovery over the most liquid current HOSE/HNX names."""
from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd

from data_provider import fetch_exchange_symbols, fetch_stock_history
from foreign_flow_provider import fetch_foreign_flow
from adaptive_strategy import classify_market_regime
from pattern_research import HORIZONS, walk_forward_validation, extract_point_in_time_features, label_forward_outcome

OUT = Path("pattern_research_output")
OUT.mkdir(exist_ok=True)


def clean(raw):
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d.columns and "Date" in raw.columns:
        d["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ("open","high","low","close","volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["date","open","high","low","close","volume"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    d.attrs["_pattern_research_clean"] = True
    return d


def main():
    max_symbols = int(os.getenv("PATTERN_RESEARCH_MAX_SYMBOLS", "120"))
    lookback = int(os.getenv("PATTERN_RESEARCH_LOOKBACK", "420"))
    symbols = fetch_exchange_symbols()
    market = clean(fetch_stock_history("VNINDEX", length=lookback, is_index=True))
    histories, errors, ranks = {}, [], []
    for i, symbol in enumerate(symbols, 1):
        try:
            d = clean(fetch_stock_history(symbol, length=lookback))
            if len(d) < 230:
                raise ValueError(f"Chỉ có {len(d)} phiên")
            cutoff_idx = max(200, len(d) - min(lookback, 300) - max(HORIZONS.values()))
            known_turnover = (d["close"] * d["volume"]).iloc[:cutoff_idx + 1].tail(20)
            ranks.append((symbol, float(known_turnover.mean())))
            histories[symbol] = d
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:250]})
        if i % 25 == 0:
            print(f"Đã tải lịch sử {i}/{len(symbols)}")
    selected = [s for s, _ in sorted(ranks, key=lambda x: x[1], reverse=True)[:max_symbols]]
    rows = []
    for n, symbol in enumerate(selected, 1):
        d = histories[symbol]
        try:
            foreign = fetch_foreign_flow(symbol, length=lookback)
        except Exception:
            foreign = pd.DataFrame()
        start = max(200, len(d) - min(lookback, 300) - max(HORIZONS.values()))
        for idx in range(start, len(d) - max(HORIZONS.values())):
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
                rows.append({
                    "symbol": symbol, "strategy": strategy, **features, **outcome,
                })
        if n % 10 == 0:
            print(f"Đã nghiên cứu {n}/{len(selected)} mã; số quan sát: {len(rows)}")
    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError("Không tạo được quan sát nghiên cứu; kiểm tra nguồn dữ liệu.")
    events.to_csv(OUT / "pattern_events.csv", index=False)
    report = walk_forward_validation(events)
    report.update({
        "universe_discovered": len(symbols),
        "symbols_with_history": len(histories),
        "symbols_tested": len(selected),
        "data_errors_count": len(errors),
        "data_errors_sample": errors[:40],
        "observations": len(events),
        "data_start": str(events["date"].min()),
        "data_end": str(events["date"].max()),
        "method": "Point-in-time event study: D1 features through signal close; entry next session open; close exit after 5/7/10 sessions; buy/sell fees, sell tax and 0.20% slippage each side. MFE/MAE from post-entry bars. No production BUY signals generated.",
        "costs": {"buy_fee":0.0015,"sell_fee":0.0015,"sell_tax":0.001,"slippage_each_side":0.002},
        "universe_warning": "Universe is current HOSE/HNX listings, so delisted stocks are absent (survivorship bias). Recent top-liquidity selection is for coverage, not an investable historical universe.",
    })
    (OUT / "pattern_research_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "universe_discovered": report["universe_discovered"],
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
