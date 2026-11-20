"""Run reproducible representative-universe validation and write artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from historical_backtest import run_historical_backtest
from validation_report import build_validation_report


SAMPLE_SYMBOLS = [
    # Banks
    "ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "OCB", "SHB",
    "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB",
    # Brokers
    "BSI", "CTS", "FTS", "HCM", "MBS", "SHS", "SSI", "VCI", "VND",
    # Real estate, construction and industrial parks
    "BCM", "CEO", "CII", "DIG", "DXG", "HDC", "HDG", "KBC", "KDH",
    "NLG", "NVL", "PDR", "SZC", "VCG", "VHM", "VRE",
    # Steel, materials and chemicals
    "DGC", "DCM", "DPM", "GVR", "HPG", "NKG", "PHR", "PLC",
    # Consumer, retail and technology
    "FPT", "FRT", "MCH", "MSN", "MWG", "PNJ", "SAB", "VNM",
    # Energy, oil and gas, transport
    "BSR", "GAS", "PVS", "PVD", "PLX", "POW", "PVT",
    # Large cap / diversified
    "BVH", "VIC", "VJC", "VTP",
]
SAMPLE_SYMBOLS = list(dict.fromkeys(SAMPLE_SYMBOLS))


def grouped_performance(frame: pd.DataFrame, group_columns: list[str]) -> dict:
    """Summarize net trade outcomes by strategy and regime."""
    if frame.empty or not set(group_columns).issubset(frame.columns):
        return {}
    output = {}
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = " | ".join(str(value) for value in keys)
        returns = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
        wins, losses = returns[returns > 0], returns[returns < 0]
        output[label] = {
            "trades": int(len(returns)),
            "win_rate_pct": round(float((returns > 0).mean() * 100), 2) if len(returns) else None,
            "avg_net_return_pct": round(float(returns.mean()), 3) if len(returns) else None,
            "median_net_return_pct": round(float(returns.median()), 3) if len(returns) else None,
            "profit_factor": round(float(wins.sum() / abs(losses.sum())), 3)
                if len(losses) and abs(losses.sum()) > 0 else None,
        }
    return output

def main() -> None:
    output = Path("backtest_output")
    output.mkdir(exist_ok=True)

    result = run_historical_backtest(
        symbols=SAMPLE_SYMBOLS,
        lookback_sessions=250,
        max_symbols=len(SAMPLE_SYMBOLS),
        strategy_filter="ALL",
    )

    trades = result["trades"]
    trades.to_csv(output / "all_signal_outcomes.csv", index=False, encoding="utf-8-sig")
    validation = build_validation_report(trades)
    executable = trades[trades["strategy"].astype(str).str.upper().isin(
        {"SWING", "REVERSAL", "PULLBACK"}
    )].copy() if not trades.empty else trades.copy()
    early = trades[trades["strategy"].astype(str).str.upper().isin(
        {"EARLY_BREAKOUT", "PREPARE"}
    )].copy() if not trades.empty else trades.copy()
    executable.to_csv(output / "executable_trades.csv", index=False, encoding="utf-8-sig")
    early.to_csv(output / "early_warning_research.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_type": "expanded_liquid_representative_sample_not_full_market",
        "validation_design": "250_session_walk_forward_with_regime_breakdown",
        "sample_symbols": SAMPLE_SYMBOLS,
        "legacy_combined_signal_summary": result["summary"],
        "portfolio": result.get("portfolio", {}),
        "validation": validation,
        "by_strategy": result["by_strategy"],
        "executable_performance_by_market_regime": grouped_performance(
            executable, ["market_regime", "strategy"]
        ),
        "adaptive_score_bucket_outcomes": grouped_performance(
            executable.assign(adaptive_score_bucket=pd.cut(
                pd.to_numeric(executable["adaptive_score"], errors="coerce"),
                bins=[-1, 39, 54, 69, 84, 100],
                labels=["0-39", "40-54", "55-69", "70-84", "85-100"],
            )) if "adaptive_score" in executable.columns else executable,
            ["adaptive_score_bucket", "strategy"],
        ) if "adaptive_score" in executable.columns else {},
        "symbols_requested": result["symbols_requested"],
        "symbols_tested": result["symbols_tested"],
        "data_errors": result["data_errors"],
        "signal_evaluation_error_count": result.get("signal_evaluation_error_count", 0),
        "signal_evaluation_errors_sample": result.get("signal_evaluation_errors_sample", []),
        "signal_diagnostics": result.get("signal_diagnostics", {}),
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
