"""30-window point-in-time top-3 market selection and diagnostic test.

At each checkpoint, use only completed daily data through that date, rank
eligible HOSE/HNX candidates, select at most three, enter next session and
simulate a 2-5 session swing trade. Research only; no orders.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_provider import fetch_exchange_symbols, fetch_stock_history
from strategy_v2 import _indicators, analyze_daily_setup, universe_check
from strategy_v2_backtest import clean, simulate
from portfolio_execution import simulate_cash_portfolio

WINDOWS = 30
LOOKBACK_SESSIONS = 250
FORWARD_SESSIONS = 5
HISTORY_SESSIONS = 1500
MAX_SYMBOLS = 0  # 0 = all listed symbols returned by provider
MAX_DAILY_PICKS = 3
MIN_NORMALIZED_SCORE = 0.65


def select_candidates(ranked: list[dict], *, max_picks: int = MAX_DAILY_PICKS,
                      setup: str | None = None) -> list[dict]:
    """Select already-ranked candidates, optionally restricting to one setup."""
    pool = ranked if setup is None else [
        item for item in ranked if item["candidate"].get("setup") == setup
    ]
    return pool[:max_picks]


def candidate_diagnostics(past: pd.DataFrame, market_past: pd.DataFrame,
                          candidate: dict, trade: dict | None) -> dict:
    """Capture point-in-time signal features for post-trade research only.

    These fields are deliberately excluded from rank_score so this diagnostic
    pass cannot silently change the strategy or fit thresholds to this sample.
    """
    r = past.iloc[-1]
    out = {}
    for field in ("rsi", "adx", "vol_ratio", "avg_turnover20", "atr"):
        value = r.get(field)
        out[field] = float(value) if pd.notna(value) else None
    close = float(r.close)
    atr = float(r.atr) if pd.notna(r.atr) else 0.0
    ema21 = float(r.ema21) if pd.notna(r.ema21) else None
    out["close_to_ema21_atr"] = (
        round((close - ema21) / atr, 4) if ema21 is not None and atr > 0 else None
    )
    out["atr_pct"] = round(atr / close * 100, 4) if close > 0 and atr > 0 else None

    for n in (20, 60):
        if len(past) > n and len(market_past) > n:
            stock_ret = float(close / float(past.iloc[-n-1].close) - 1)
            market_ret = float(
                market_past.iloc[-1].close / market_past.iloc[-n-1].close - 1
            )
            out[f"relative_strength_{n}s_pct"] = round(
                (stock_ret - market_ret) * 100, 4
            )
        else:
            out[f"relative_strength_{n}s_pct"] = None

    entry = trade.get("actual_entry") if trade else None
    stop = trade.get("initial_stop") if trade else None
    out["stop_distance_pct"] = (
        round((float(entry) - float(stop)) / float(entry) * 100, 4)
        if entry and stop and float(entry) > 0 else None
    )
    out["entry_gap_pct"] = (
        round((float(entry) / close - 1) * 100, 4)
        if entry and close > 0 else None
    )
    out["setup"] = candidate.get("setup")
    return out


def feature_outcome_diagnostics(frame: pd.DataFrame) -> dict:
    """Descriptive winner/loser feature medians; not a fitted scoring model."""
    fields = [
        "rsi", "adx", "vol_ratio", "close_to_ema21_atr", "atr_pct",
        "relative_strength_20s_pct", "relative_strength_60s_pct",
        "stop_distance_pct", "entry_gap_pct",
    ]
    result = {}
    if frame.empty or "net_return_pct" not in frame:
        return result
    for label, subset in (
        ("winners", frame[frame.net_return_pct > 0]),
        ("losers", frame[frame.net_return_pct <= 0]),
    ):
        result[label] = {
            "trades": int(len(subset)),
            "feature_medians": {
                field: round(float(pd.to_numeric(subset[field], errors="coerce").median()), 4)
                if field in subset and pd.to_numeric(subset[field], errors="coerce").notna().any()
                else None
                for field in fields
            },
        }
    return result

def main():
    out = Path("top3_walkforward_output")
    out.mkdir(exist_ok=True)
    listed = sorted(set(fetch_exchange_symbols()))
    symbols = listed if MAX_SYMBOLS == 0 else listed[:MAX_SYMBOLS]

    market = clean(fetch_stock_history("VNINDEX", length=HISTORY_SESSIONS, is_index=True))
    market_ind = _indicators(market)
    if len(market_ind) < LOOKBACK_SESSIONS + WINDOWS * FORWARD_SESSIONS:
        raise RuntimeError("VNINDEX không đủ dữ liệu để tạo 30 khung thời gian.")

    # 30 checkpoints, five trading sessions apart; every checkpoint has
    # at least 250 prior sessions and five forward sessions for evaluation.
    last_checkpoint = len(market_ind) - FORWARD_SESSIONS - 1
    first_checkpoint = last_checkpoint - (WINDOWS - 1) * FORWARD_SESSIONS
    checkpoints = list(range(first_checkpoint, last_checkpoint + 1, FORWARD_SESSIONS))
    if len(checkpoints) != WINDOWS:
        raise RuntimeError(f"Chỉ tạo được {len(checkpoints)} khung, cần {WINDOWS}.")

    histories = {}
    errors = []
    for n, symbol in enumerate(symbols, 1):
        try:
            raw = clean(fetch_stock_history(symbol, length=HISTORY_SESSIONS))
            if len(raw) < LOOKBACK_SESSIONS + FORWARD_SESSIONS + 1:
                errors.append({"symbol": symbol, "error": f"Chỉ có {len(raw)} phiên"})
                continue
            histories[symbol] = _indicators(raw)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:180]})
        if n % 25 == 0:
            print(f"Đã tải {n}/{len(symbols)} mã; có dữ liệu {len(histories)} mã")

    market_dates = pd.to_datetime(market_ind.date).to_numpy()
    window_rows, selection_rows = [], []
    swing_only_rows = []
    all_selected_returns = []

    for window_no, market_i in enumerate(checkpoints, 1):
        asof_date = pd.Timestamp(market_ind.iloc[market_i].date)
        market_row = market_ind.iloc[market_i]
        if pd.isna(market_row.sma50) or pd.isna(market_row.sma200):
            regime = "UNKNOWN"
        elif market_row.close < market_row.sma200:
            regime = "BEARISH"
        elif market_row.sma50 > market_row.sma200:
            regime = "BULLISH"
        else:
            regime = "NEUTRAL"

        ranked = []
        for symbol, full in histories.items():
            dates = pd.to_datetime(full.date).to_numpy()
            i = int(dates.searchsorted(asof_date.to_datetime64(), side="right")) - 1
            if i < LOOKBACK_SESSIONS - 1 or i + FORWARD_SESSIONS >= len(full):
                continue
            # Ensure the stock has a bar on the same as-of date (no stale data).
            if pd.Timestamp(full.iloc[i].date).normalize() != asof_date.normalize():
                continue
            past = full.iloc[:i + 1].copy()
            eligible, blockers = universe_check(past)
            if not eligible:
                continue
            result = analyze_daily_setup(
                symbol, past, None, indicators_precomputed=True,
                market_regime_override=regime
            )
            candidates = result.get("candidates", [])
            if not candidates:
                continue
            candidate = dict(candidates[0])
            candidate["holding_sessions"] = [2, FORWARD_SESSIONS]
            normalized_score = float(candidate["score"]) / max(1, float(candidate["max_score"]))
            if normalized_score < MIN_NORMALIZED_SCORE:
                continue
            # Rank quality first, then setup score and risk-adjusted exposure.
            rank_score = normalized_score * 100.0 + float(candidate.get("risk_factor", 1.0)) * 5.0
            ranked.append({
                "symbol": symbol, "index": i, "candidate": candidate,
                "normalized_score": normalized_score, "rank_score": rank_score,
                "avg_turnover20": float(past.iloc[-1].avg_turnover20)
                    if pd.notna(past.iloc[-1].avg_turnover20) else 0.0,
            })

        ranked.sort(key=lambda x: (
            x["rank_score"], x["avg_turnover20"]
        ), reverse=True)
        selected = select_candidates(ranked)
        # Parallel research arm: at most three Swing Breakout candidates only.
        # It does not alter the baseline ranking or production BUY logic.
        swing_selected = select_candidates(ranked, setup="SWING_BREAKOUT")
        returns = []
        selected_names = []

        for rank, item in enumerate(selected, 1):
            trade = simulate(histories[item["symbol"]], item["index"], item["candidate"])
            if trade is None:
                outcome = "NO_EXECUTION"
                net_return = None
            else:
                outcome = trade["exit_reason"]
                net_return = float(trade["net_return_pct"])
                returns.append(net_return)
                all_selected_returns.append(net_return)
            selected_names.append(item["symbol"])
            features = candidate_diagnostics(
                histories[item["symbol"]].iloc[:item["index"] + 1],
                market_ind.iloc[:market_i + 1],
                item["candidate"], trade,
            )
            selection_rows.append({
                "window": window_no, "asof_date": str(asof_date.date()),
                "rank": rank, "symbol": item["symbol"],
                "setup": item["candidate"]["setup"],
                "score": item["candidate"]["score"],
                "max_score": item["candidate"]["max_score"],
                "rank_score": round(item["rank_score"], 3),
                "market_regime": regime,
                "entry_date": trade["entry_date"] if trade else None,
                "exit_date": trade["exit_date"] if trade else None,
                "net_return_pct": net_return, "outcome": outcome,
                "entry_price": trade["actual_entry"] if trade else None,
                "stop_loss": trade["initial_stop"] if trade else None,
                "slippage_rate": item["candidate"]["risk_plan"].get("slippage_rate", 0.001),
                "risk_factor": float(item["candidate"].get("risk_factor", 1.0)),
                "normalized_score": round(item["normalized_score"], 4),
                **features,
            })

        for swing_rank, item in enumerate(swing_selected, 1):
            trade = simulate(histories[item["symbol"]], item["index"], item["candidate"])
            swing_only_rows.append({
                "window": window_no,
                "asof_date": str(asof_date.date()),
                "rank": swing_rank,
                "symbol": item["symbol"],
                "setup": item["candidate"]["setup"],
                "normalized_score": round(item["normalized_score"], 4),
                "net_return_pct": float(trade["net_return_pct"]) if trade else None,
                "outcome": trade["exit_reason"] if trade else "NO_EXECUTION",
                "entry_date": trade["entry_date"] if trade else None,
                "exit_date": trade["exit_date"] if trade else None,
            })

        forward_market_return = (
            float(market_ind.iloc[market_i + FORWARD_SESSIONS].close /
                  market_ind.iloc[market_i].close - 1) * 100.0
        )
        avg_return = sum(returns) / len(returns) if returns else None
        win_rate = (sum(x > 0 for x in returns) / len(returns) * 100.0) if returns else None
        window_rows.append({
            "window": window_no, "asof_date": str(asof_date.date()),
            "market_regime": regime, "eligible_candidates": len(ranked),
            "selected_count": len(selected), "selected_symbols": ", ".join(selected_names),
            "executed_trades": len(returns),
            "avg_selected_net_return_pct": round(avg_return, 3) if avg_return is not None else None,
            "win_rate_pct": round(win_rate, 2) if win_rate is not None else None,
            "vnindex_forward_5s_pct": round(forward_market_return, 3),
        })
        print(f"Window {window_no:02d}/{WINDOWS} | {asof_date.date()} | {regime} | "
              f"candidates={len(ranked)} | selected={len(selected)} | "
              f"executed={len(returns)} | avg={avg_return}")

    trades_df = pd.DataFrame(selection_rows)
    windows_df = pd.DataFrame(window_rows)
    trades_df.to_csv(out / "top3_selections.csv", index=False)
    windows_df.to_csv(out / "30_window_summary.csv", index=False)

    valid = [x for x in all_selected_returns]
    executable_rows = [r for r in selection_rows if r.get("net_return_pct") is not None]
    portfolio = simulate_cash_portfolio(
        executable_rows, initial_capital=10_000_000, risk_per_trade=0.02,
        max_positions=5, max_position_pct=0.20, lot_size=100,
    )
    pd.DataFrame(portfolio.get("accepted_trade_details", [])).to_csv(
        out / "portfolio_accepted_trades.csv", index=False
    )
    pd.DataFrame(portfolio.get("skipped_trade_details", [])).to_csv(
        out / "portfolio_skipped_trades.csv", index=False
    )
    executed_df = trades_df[trades_df["net_return_pct"].notna()].copy() if not trades_df.empty else pd.DataFrame()
    swing_df = pd.DataFrame(swing_only_rows)
    swing_executed_df = (
        swing_df[swing_df["net_return_pct"].notna()].copy()
        if not swing_df.empty else pd.DataFrame()
    )
    swing_df.to_csv(out / "swing_only_research.csv", index=False)

    def performance_by(frame, field):
        result = {}
        if frame.empty or field not in frame:
            return result
        for key, group in frame.groupby(field, dropna=False):
            values = pd.to_numeric(group["net_return_pct"], errors="coerce").dropna()
            gains = float(values[values > 0].sum())
            losses = abs(float(values[values < 0].sum()))
            result[str(key)] = {
                "trades": int(len(values)),
                "win_rate_pct": round(float((values > 0).mean() * 100), 2) if len(values) else None,
                "average_net_return_pct": round(float(values.mean()), 3) if len(values) else None,
                "median_net_return_pct": round(float(values.median()), 3) if len(values) else None,
                "profit_factor": round(gains / losses, 3) if losses > 0 else None,
            }
        return result

    benchmark_values = pd.to_numeric(windows_df["vnindex_forward_5s_pct"], errors="coerce").dropna()
    overall = {
        "windows_requested": WINDOWS,
        "windows_completed": len(window_rows),
        "listed_symbols_from_provider": len(symbols),
        "symbols_with_history": len(histories),
        "data_errors": len(errors),
        "max_picks_per_checkpoint": MAX_DAILY_PICKS,
        "minimum_normalized_score": MIN_NORMALIZED_SCORE,
        "selected_slots": len(selection_rows),
        "no_execution_selections": int((trades_df["outcome"] == "NO_EXECUTION").sum()) if not trades_df.empty else 0,
        "executed_trades": len(valid),
        "win_rate_pct": round(sum(x > 0 for x in valid) / len(valid) * 100, 2) if valid else None,
        "average_net_return_pct": round(sum(valid) / len(valid), 3) if valid else None,
        "median_net_return_pct": round(float(pd.Series(valid).median()), 3) if valid else None,
        "profit_factor": round(sum(x for x in valid if x > 0) /
                               abs(sum(x for x in valid if x < 0)), 3)
                         if any(x < 0 for x in valid) else None,
        "diagnostics": {
            "by_setup": performance_by(executed_df, "setup"),
            "swing_only_parallel_experiment": {
                "selected_slots": int(len(swing_df)),
                "no_execution_selections": int((swing_df["outcome"] == "NO_EXECUTION").sum()) if not swing_df.empty else 0,
                "executed_trades": int(len(swing_executed_df)),
                "performance": performance_by(swing_executed_df.assign(setup="SWING_BREAKOUT"), "setup"),
                "note": "Parallel research arm using the same point-in-time candidate pool, dates, execution simulator and costs; baseline ranking is unchanged. Small samples are descriptive, not proof of edge."
            },
            "winner_loser_feature_medians": feature_outcome_diagnostics(executed_df),
            "by_exit_reason": performance_by(executed_df, "outcome"),
            "by_market_regime": performance_by(executed_df, "market_regime"),
            "by_normalized_score": performance_by(
                executed_df.assign(score_band=pd.cut(
                    executed_df["normalized_score"],
                    bins=[0.65, 0.75, 0.85, 0.95, 1.000001],
                    labels=["65-74%", "75-84%", "85-94%", "95-100%"],
                    include_lowest=True
                )), "score_band"
            ) if not executed_df.empty else {},
            "selection_outcomes": trades_df["outcome"].value_counts(dropna=False).to_dict()
                if not trades_df.empty else {},
        },
        "benchmark_context": {
            "vnindex_forward_5_sessions_average_pct": round(float(benchmark_values.mean()), 3) if len(benchmark_values) else None,
            "vnindex_forward_5_sessions_median_pct": round(float(benchmark_values.median()), 3) if len(benchmark_values) else None,
            "note": "Descriptive checkpoint context only; not a matched benchmark because candidate holding periods vary from 2 to 5 sessions."
        },
        "cash_portfolio_simulation": portfolio,
        "method": {
            "lookback": "250 completed trading sessions for indicators/eligibility",
            "checkpoints": "30 dates spaced five trading sessions apart",
            "selection": f"At most {MAX_DAILY_PICKS} highest-ranked candidates per checkpoint; no forced picks",
            "quality_filter": f"Exclude candidates below normalized setup score {MIN_NORMALIZED_SCORE:.2f}",
            "execution": "Next-session execution; 2-5 session holding; existing conservative stop/target simulator",
            "costs": "Existing strategy V2 fees, taxes and slippage",
            "benchmark": "VNINDEX close-to-close return over the next five sessions, reported as context only",
        },
        "limitations": [
            "Current listed universe is used; historical delisted constituents are not reconstructed.",
            "Selection is point-in-time on each checkpoint, but the universe listing itself is current.",
            f"At most {MAX_DAILY_PICKS} candidates are selected per checkpoint; portfolio simulation reserves cash and prevents duplicate open tickers.",
            "Cash portfolio uses 10m VND, 2% risk per trade, 20% maximum allocation per position, five concurrent positions and 100-share board lots.",
            "Portfolio drawdown is realized-only; daily mark-to-market drawdown requires a daily equity curve and is not claimed here.",
            "A small number of selected/executed trades is not sufficient evidence of a persistent statistical edge.",
            "The parallel swing-only arm is a research comparison and does not change production signals or the baseline top-three selection.",
            "This is a research simulation, not a live portfolio or guaranteed return; no production BUY rules are changed.",
        ],
    }
    (out / "report.json").write_text(
        json.dumps({"overall": overall, "windows": window_rows, "data_errors_sample": errors[:100]},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    if not histories:
        raise RuntimeError("Không tải được dữ liệu lịch sử cổ phiếu.")


if __name__ == "__main__":
    main()
