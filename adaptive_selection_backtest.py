"""Adaptive, research-only top-three selection experiment.

Compares fixed score ranking with point-in-time relative-strength ranking and
a conservative quality gate. Uses the full available history through each
checkpoint for indicators; 250 bars are only the minimum warm-up required by
the existing SMA200-based strategy. No production signals or orders are changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_provider import fetch_exchange_symbols, fetch_stock_history
from strategy_v2 import _indicators, analyze_daily_setup, universe_check
from strategy_v2_backtest import clean, simulate
from portfolio_execution import simulate_cash_portfolio

WINDOWS = 60
CHECKPOINT_STEP = 2
FORWARD_SESSIONS = 5
HISTORY_SESSIONS = 1500
MIN_HISTORY_SESSIONS = 250
MAX_PICKS = 3
MIN_SCORE = 0.65
INITIAL_CAPITAL = 10_000_000


def safe_float(value, default=None):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def make_checkpoints(session_count: int, windows: int = WINDOWS,
                     step: int = CHECKPOINT_STEP,
                     forward: int = FORWARD_SESSIONS,
                     minimum_history: int = MIN_HISTORY_SESSIONS) -> list[int]:
    """Build recent checkpoints while retaining enough warm-up and forward bars."""
    if windows < 1 or step < 1 or forward < 1:
        raise ValueError("windows, step and forward must be positive")
    last = session_count - forward - 1
    first = last - (windows - 1) * step
    if first < minimum_history - 1:
        raise ValueError("Not enough sessions for requested checkpoints and warm-up")
    return list(range(first, last + 1, step))


def point_in_time_features(past: pd.DataFrame, market_past: pd.DataFrame) -> dict:
    """Features use only bars available at the checkpoint."""
    row = past.iloc[-1]
    close = safe_float(row.close, 0.0) or 0.0
    atr = safe_float(row.get("atr"), 0.0) or 0.0
    ema21 = safe_float(row.get("ema21"))
    result = {
        "rsi": safe_float(row.get("rsi")),
        "adx": safe_float(row.get("adx")),
        "vol_ratio": safe_float(row.get("vol_ratio")),
        "atr_pct": (atr / close * 100) if close > 0 and atr > 0 else None,
        "close_to_ema21_atr": ((close - ema21) / atr)
            if ema21 is not None and atr > 0 else None,
    }
    for n in (20, 60):
        if len(past) > n and len(market_past) > n:
            stock_ret = close / float(past.iloc[-n-1].close) - 1
            market_ret = (float(market_past.iloc[-1].close) /
                          float(market_past.iloc[-n-1].close) - 1)
            result[f"rs{n}_pct"] = (stock_ret - market_ret) * 100
        else:
            result[f"rs{n}_pct"] = None
    return result


def rank_candidates(candidates: list[dict], arm: str) -> list[dict]:
    """Rank the same point-in-time candidate pool under a predeclared research arm."""
    rows = [dict(x) for x in candidates]
    if arm == "RS60_POSITIVE":
        rows = [x for x in rows if safe_float(x["features"].get("rs60_pct"), -999) > 0]
        rows.sort(key=lambda x: (
            x["normalized_score"],
            safe_float(x["features"].get("rs60_pct"), -999),
            x["avg_turnover20"],
        ), reverse=True)
    elif arm == "QUALITY_GATE":
        def passes(x):
            f = x["features"]
            adx, rsi = safe_float(f.get("adx")), safe_float(f.get("rsi"))
            gap = safe_float(f.get("close_to_ema21_atr"))
            rs60 = safe_float(f.get("rs60_pct"))
            return (
                x["normalized_score"] >= 0.75
                and rs60 is not None and rs60 > 0
                and adx is not None and 18 <= adx <= 40
                and rsi is not None and rsi <= 75
                and gap is not None and gap <= 2.0
            )
        rows = [x for x in rows if passes(x)]
        rows.sort(key=lambda x: (
            x["normalized_score"],
            safe_float(x["features"].get("rs60_pct"), -999),
            x["avg_turnover20"],
        ), reverse=True)
    elif arm == "CROSS_SECTIONAL_BLEND":
        if rows:
            frame = pd.DataFrame([{
                "score": x["normalized_score"],
                "rs60": safe_float(x["features"].get("rs60_pct"), -20),
                "adx": safe_float(x["features"].get("adx"), 0),
                "liquidity": x["avg_turnover20"],
                "idx": i,
            } for i, x in enumerate(rows)])
            # Percentile ranks are computed only among candidates on this date.
            frame["blend"] = (
                0.55 * frame["score"].rank(pct=True)
                + 0.25 * frame["rs60"].rank(pct=True)
                + 0.10 * frame["adx"].rank(pct=True)
                + 0.10 * frame["liquidity"].rank(pct=True)
            )
            scores = dict(zip(frame["idx"], frame["blend"]))
            for i, item in enumerate(rows):
                item["research_rank_score"] = float(scores[i])
            rows.sort(key=lambda x: x["research_rank_score"], reverse=True)
    elif arm == "BASELINE":
        rows.sort(key=lambda x: (
            x["normalized_score"] * 100 + x["risk_factor"] * 5,
            x["avg_turnover20"],
        ), reverse=True)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return rows[:MAX_PICKS]


def trade_metrics(rows: list[dict]) -> dict:
    values = [safe_float(x.get("net_return_pct")) for x in rows]
    values = [x for x in values if x is not None]
    gains = sum(x for x in values if x > 0)
    losses = abs(sum(x for x in values if x < 0))
    excess = [safe_float(x.get("benchmark_excess_pct")) for x in rows]
    excess = [x for x in excess if x is not None]
    return {
        "executed_trades": len(values),
        "win_rate_pct": round(sum(x > 0 for x in values) / len(values) * 100, 2) if values else None,
        "average_net_return_pct": round(float(np.mean(values)), 3) if values else None,
        "median_net_return_pct": round(float(np.median(values)), 3) if values else None,
        "profit_factor": round(gains / losses, 3) if losses > 0 else None,
        "average_matched_vnindex_excess_pct": round(float(np.mean(excess)), 3) if excess else None,
    }


def main():
    out = Path("adaptive_selection_output")
    out.mkdir(exist_ok=True)
    symbols = sorted(set(fetch_exchange_symbols()))
    market = clean(fetch_stock_history("VNINDEX", length=HISTORY_SESSIONS, is_index=True))
    market_ind = _indicators(market)
    checkpoints = make_checkpoints(len(market_ind))
    market_dates = pd.to_datetime(market_ind.date).to_numpy()

    histories = {}
    errors = []
    for number, symbol in enumerate(symbols, 1):
        try:
            raw = clean(fetch_stock_history(symbol, length=HISTORY_SESSIONS))
            if len(raw) < MIN_HISTORY_SESSIONS + FORWARD_SESSIONS + 1:
                errors.append({"symbol": symbol, "error": f"history={len(raw)}"})
                continue
            histories[symbol] = _indicators(raw)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:180]})
        if number % 25 == 0:
            print(f"Downloaded {number}/{len(symbols)}; usable histories={len(histories)}")

    arms = ("BASELINE", "RS60_POSITIVE", "CROSS_SECTIONAL_BLEND", "QUALITY_GATE")
    selected_rows = []
    window_rows = []
    for window_no, market_i in enumerate(checkpoints, 1):
        asof = pd.Timestamp(market_ind.iloc[market_i].date)
        mr = market_ind.iloc[market_i]
        if pd.isna(mr.sma50) or pd.isna(mr.sma200):
            regime = "UNKNOWN"
        elif mr.close < mr.sma200:
            regime = "BEARISH"
        elif mr.sma50 > mr.sma200:
            regime = "BULLISH"
        else:
            regime = "NEUTRAL"

        pool = []
        for symbol, full in histories.items():
            dates = pd.to_datetime(full.date).to_numpy()
            index = int(dates.searchsorted(asof.to_datetime64(), side="right")) - 1
            if index < MIN_HISTORY_SESSIONS - 1 or index + FORWARD_SESSIONS >= len(full):
                continue
            if pd.Timestamp(full.iloc[index].date).normalize() != asof.normalize():
                continue
            past = full.iloc[:index + 1].copy()
            eligible, _ = universe_check(past)
            if not eligible:
                continue
            analysis = analyze_daily_setup(
                symbol, past, None, indicators_precomputed=True,
                market_regime_override=regime,
            )
            candidates = analysis.get("candidates", [])
            if not candidates:
                continue
            candidate = dict(candidates[0])
            candidate["holding_sessions"] = [2, FORWARD_SESSIONS]
            normalized = float(candidate["score"]) / max(1, float(candidate["max_score"]))
            if normalized < MIN_SCORE:
                continue
            features = point_in_time_features(past, market_ind.iloc[:market_i + 1])
            turnover = safe_float(past.iloc[-1].get("avg_turnover20"), 0.0) or 0.0
            pool.append({
                "symbol": symbol, "index": index, "candidate": candidate,
                "setup": candidate["setup"], "normalized_score": normalized,
                "risk_factor": float(candidate.get("risk_factor", 1.0)),
                "avg_turnover20": turnover, "features": features,
            })

        benchmark_close = {
            pd.Timestamp(row.date).strftime("%Y-%m-%d"): float(row.close)
            for row in market_ind.iloc[:market_i + FORWARD_SESSIONS + 1].itertuples()
        }
        for arm in arms:
            chosen = rank_candidates(pool, arm)
            arm_returns = []
            for rank, item in enumerate(chosen, 1):
                trade = simulate(histories[item["symbol"]], item["index"], item["candidate"])
                row = {
                    "arm": arm, "window": window_no, "asof_date": str(asof.date()),
                    "market_regime": regime, "rank": rank, "symbol": item["symbol"],
                    "setup": item["setup"], "normalized_score": round(item["normalized_score"], 4),
                    "rank_score": item.get("research_rank_score"),
                    **item["features"],
                    "entry_date": trade["entry_date"] if trade else None,
                    "exit_date": trade["exit_date"] if trade else None,
                    "entry_price": trade["actual_entry"] if trade else None,
                    "stop_loss": trade["initial_stop"] if trade else None,
                    "slippage_rate": item["candidate"]["risk_plan"].get("slippage_rate", 0.001),
                    "risk_factor": item["risk_factor"],
                    "net_return_pct": trade["net_return_pct"] if trade else None,
                    "outcome": trade["exit_reason"] if trade else "NO_EXECUTION",
                }
                if trade:
                    entry_close = benchmark_close.get(trade["entry_date"])
                    exit_close = benchmark_close.get(trade["exit_date"])
                    if entry_close and exit_close and entry_close > 0:
                        benchmark_return = (exit_close / entry_close - 1) * 100
                        row["benchmark_return_pct"] = round(benchmark_return, 3)
                        row["benchmark_excess_pct"] = round(
                            trade["net_return_pct"] - benchmark_return, 3
                        )
                    arm_returns.append(row)
                selected_rows.append(row)
            window_rows.append({
                "arm": arm, "window": window_no, "asof_date": str(asof.date()),
                "market_regime": regime, "candidate_pool": len(pool),
                "selected": len(chosen), "executed": len(arm_returns),
                "average_return_pct": round(float(np.mean([
                    x["net_return_pct"] for x in arm_returns
                ])), 3) if arm_returns else None,
            })
        if window_no % 10 == 0:
            print(f"Checkpoint {window_no}/{len(checkpoints)} {asof.date()} candidates={len(pool)}")

    selections = pd.DataFrame(selected_rows)
    windows = pd.DataFrame(window_rows)
    selections.to_csv(out / "selection_trades.csv", index=False)
    windows.to_csv(out / "window_summary.csv", index=False)

    report = {
        "universe": "Current provider HOSE/HNX list; historical delisted constituents unavailable",
        "listed_symbols": len(symbols), "symbols_with_history": len(histories),
        "data_errors": len(errors), "checkpoints": len(checkpoints),
        "checkpoint_step_sessions": CHECKPOINT_STEP,
        "forward_sessions": FORWARD_SESSIONS,
        "max_picks_per_checkpoint": MAX_PICKS,
        "minimum_normalized_score": MIN_SCORE,
        "arms": {},
        "method": {
            "lookback": "All available completed history through each as-of date (up to 1500 sessions); 250 sessions are only the existing SMA200/universe warm-up minimum.",
            "checkpoints": f"{len(checkpoints)} recent checkpoints, spaced {CHECKPOINT_STEP} trading sessions apart.",
            "execution": "Next-session execution with the existing conservative stop/target simulator, 5-session maximum holding cap and existing fees/tax/slippage.",
            "benchmark": "Matched VNINDEX close-to-close return from each simulated entry date to exit date; excess is trade net return minus index return.",
            "selection": "At most three candidates per checkpoint; no forced picks. Same candidate pool and simulator across all arms.",
            "portfolio": "Each arm separately simulated with 10m VND, 2% risk, 20% max position, five concurrent positions and 100-share lots.",
            "no_production_change": True,
        },
        "limitations": [
            "Research arms are exploratory; repeated comparisons are not independent holdout validation.",
            "Overlapping checkpoints are correlated. Portfolio simulation rejects duplicate open symbols and respects cash/position limits.",
            "Current listed universe is not survivorship-bias-free because historical delisted constituents are unavailable.",
            "The portfolio simulator reports realized drawdown only, not daily mark-to-market drawdown.",
            "A profitable small sample is not sufficient evidence to change production BUY/SELL rules.",
        ],
        "data_errors_sample": errors[:100],
    }
    for arm in arms:
        frame = selections[selections.arm == arm].copy()
        executed = frame[frame.net_return_pct.notna()]
        metrics = trade_metrics(executed.to_dict("records"))
        portfolio_rows = executed.to_dict("records")
        portfolio = simulate_cash_portfolio(
            portfolio_rows, initial_capital=INITIAL_CAPITAL, risk_per_trade=0.02,
            max_positions=5, max_position_pct=0.20, lot_size=100,
        )
        report["arms"][arm] = {
            "selected_slots": int(len(frame)),
            "no_execution": int((frame.outcome == "NO_EXECUTION").sum()),
            **metrics,
            "cash_portfolio": portfolio,
        }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
