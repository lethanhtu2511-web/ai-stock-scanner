"""Run a reproducible representative-universe validation and write artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from historical_backtest import run_historical_backtest


# Representative liquid names from both HOSE and HNX. This is a sample,
# not a claim that the list equals every currently listed security.
SAMPLE_SYMBOLS = [
    "ACB", "BCM", "BID", "CTG", "DCM", "DGC", "DPM", "FPT", "GAS",
    "GVR", "HDB", "HPG", "MBB", "MSN", "MWG", "NLG", "PLX", "PVS",
    "SAB", "SHB", "SHS", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
    "VIC", "VND", "VNM", "VPB", "VRE",
]


def main() -> None:
    output = Path("backtest_output")
    output.mkdir(exist_ok=True)

    result = run_historical_backtest(
        symbols=SAMPLE_SYMBOLS,
        lookback_sessions=120,
        max_symbols=len(SAMPLE_SYMBOLS),
        strategy_filter="ALL",
    )

    trades = result["trades"]
    trades.to_csv(output / "trades.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_type": "representative_sample_not_full_market",
        "sample_symbols": SAMPLE_SYMBOLS,
        "summary": result["summary"],
        "by_strategy": result["by_strategy"],
        "symbols_requested": result["symbols_requested"],
        "symbols_tested": result["symbols_tested"],
        "data_errors": result["data_errors"],
        "selected_symbols": result["selected_symbols"],
        "costs": result["costs"],
        "lookback_sessions": result["lookback_sessions"],
        "data_end_date": result.get("data_end_date"),
        "benchmark_available": result.get("benchmark_available"),
        "foreign_flow_symbols": result.get("foreign_flow_symbols"),
        "foreign_flow_missing_symbols": result.get("foreign_flow_missing_symbols"),
        "method_note": result["method_note"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result["symbols_tested"] < 10:
        raise SystemExit(
            "Không đủ dữ liệu: chỉ kiểm định được dưới 10 mã. "
            "Không được coi kết quả này là báo cáo hợp lệ."
        )


if __name__ == "__main__":
    main()
