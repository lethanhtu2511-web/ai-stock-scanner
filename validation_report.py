"""Separate executable BUY backtests from early-warning setup research."""
from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd


EXECUTABLE_STRATEGIES = {"SWING", "REVERSAL", "PULLBACK"}
EARLY_STRATEGIES = {"EARLY_BREAKOUT", "PREPARE"}


def _metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "trades": 0, "win_rate_pct": None, "avg_return_pct": None,
            "median_return_pct": None, "profit_factor": None,
            "best_return_pct": None, "worst_return_pct": None,
        }
    returns = pd.to_numeric(frame["net_return_pct"], errors="coerce").dropna()
    wins, losses = returns[returns > 0], returns[returns < 0]
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(returns)),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2),
        "avg_return_pct": round(float(returns.mean()), 3),
        "median_return_pct": round(float(returns.median()), 3),
        "profit_factor": round(float(wins.sum()) / gross_loss, 3) if gross_loss else None,
        "best_return_pct": round(float(returns.max()), 3),
        "worst_return_pct": round(float(returns.min()), 3),
    }


def simulate_portfolio(trades: pd.DataFrame, initial_capital: float = 10_000_000,
                       risk_per_trade: float = 0.02, max_positions: int = 5,
                       lot_size: int = 100) -> dict:
    """Long-only cash portfolio; 2% equity risk and 20% max allocation per slot."""
    if initial_capital <= 0 or not 0 < risk_per_trade <= 0.05 or max_positions < 1:
        raise ValueError("Tham số mô phỏng danh mục không hợp lệ.")
    if trades.empty:
        return {"status": "NO_EXECUTABLE_SIGNALS", "initial_capital": initial_capital,
                "ending_equity": initial_capital, "return_pct": 0.0,
                "accepted_trades": 0, "skipped_trades": 0, "max_positions": max_positions}

    rows = trades.sort_values(["entry_date", "symbol", "strategy"]).to_dict("records")
    open_positions = []
    equity = float(initial_capital)
    accepted = skipped = 0
    peak = equity
    max_drawdown = 0.0
    slot_cap = 1.0 / max_positions

    for row in rows:
        entry_day = pd.Timestamp(row["entry_date"])
        still_open = []
        for pos in open_positions:
            if pd.Timestamp(pos["exit_date"]) < entry_day:
                equity += pos["pnl"]
                peak = max(peak, equity)
                max_drawdown = min(max_drawdown, equity / peak - 1)
            else:
                still_open.append(pos)
        open_positions = still_open

        symbol = str(row["symbol"]).upper()
        if len(open_positions) >= max_positions or any(p["symbol"] == symbol for p in open_positions):
            skipped += 1
            continue

        try:
            entry = float(row.get("entry_price"))
            stop = float(row.get("stop_loss"))
            ret = float(row.get("net_return_pct")) / 100
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not all(math.isfinite(value) for value in (entry, stop, ret)) or entry <= 0 or stop <= 0 or stop >= entry:
            skipped += 1
            continue

        risk_per_share = entry - stop
        risk_budget = equity * risk_per_trade
        max_value = equity * slot_cap
        shares_by_risk = int(risk_budget / risk_per_share)
        shares_by_value = int(max_value / entry)
        shares = min(shares_by_risk, shares_by_value)
        shares = (shares // lot_size) * lot_size
        if shares < lot_size:
            skipped += 1
            continue

        invested = shares * entry
        pnl = invested * ret
        open_positions.append({
            "symbol": symbol, "exit_date": row["exit_date"], "pnl": pnl,
        })
        accepted += 1

    for pos in sorted(open_positions, key=lambda p: p["exit_date"]):
        equity += pos["pnl"]
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1)

    return {
        "status": "SIMULATED" if accepted else "NO_TRADES_FIT_POSITION_RULES",
        "initial_capital": round(initial_capital, 2),
        "ending_equity": round(equity, 2),
        "return_pct": round((equity / initial_capital - 1) * 100, 3),
        "accepted_trades": accepted,
        "skipped_trades": skipped,
        "max_positions": max_positions,
        "risk_per_trade_pct": round(risk_per_trade * 100, 2),
        "max_allocation_per_position_pct": round(slot_cap * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 3),
        "note": "Mô phỏng vốn tiền mặt, tối đa 5 vị thế, lô 100; chỉ dùng giao dịch có stop-loss. Lãi/lỗ đã bao gồm phí, thuế và trượt giá trong net_return_pct.",
    }


def build_validation_report(trades: pd.DataFrame) -> dict:
    if trades.empty:
        empty = trades.copy()
        return {
            "executable_trading": {
                **_metrics(empty), "status": "NO_CONFIRMED_BUY_SIGNALS",
                "portfolio_simulation": simulate_portfolio(empty),
            },
            "early_warning_research": {**_metrics(empty), "status": "NO_EARLY_SETUPS"},
            "interpretation": "Không có dữ liệu giao dịch để kết luận hiệu quả.",
        }

    normalized = trades.copy()
    normalized["strategy"] = normalized["strategy"].astype(str).str.upper()
    executable = normalized[normalized["strategy"].isin(EXECUTABLE_STRATEGIES)].copy()
    early = normalized[normalized["strategy"].isin(EARLY_STRATEGIES)].copy()
    actual_metrics = _metrics(executable)
    early_metrics = _metrics(early)
    actual_metrics["status"] = "MEASURED" if not executable.empty else "NO_CONFIRMED_BUY_SIGNALS"
    actual_metrics["portfolio_simulation"] = simulate_portfolio(executable)
    early_metrics["status"] = "RESEARCH_ONLY_NOT_BUY_SIGNALS" if not early.empty else "NO_EARLY_SETUPS"
    return {
        "executable_trading": actual_metrics,
        "early_warning_research": early_metrics,
        "interpretation": (
            "EARLY_BREAKOUT/PREPARE là cảnh báo sớm, không phải lệnh BUY xác nhận và không được gộp vào hiệu quả giao dịch thật. "
            "Chỉ SWING/REVERSAL/PULLBACK có BUY xác nhận mới được đưa vào mô phỏng danh mục."
        ),
    }
