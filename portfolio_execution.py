"""Cash- and risk-constrained portfolio simulation for point-in-time research picks.

Research only. No broker integration or order placement.
"""
from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import Iterable

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001


def simulate_cash_portfolio(
    trades: Iterable[dict],
    initial_capital: float = 10_000_000,
    risk_per_trade: float = 0.02,
    max_positions: int = 5,
    max_position_pct: float = 0.20,
    lot_size: int = 100,
) -> dict:
    """Simulate long-only cash positions with fees, taxes, slippage and board lots.

    A position consumes cash at entry and releases principal plus net P/L at exit.
    Risk sizing includes estimated round-trip costs. Equity used for sizing is
    cash plus entry cost of open positions; unrealized P/L is deliberately not
    guessed because this input contains trade-level outcomes, not daily marks.
    """
    if initial_capital <= 0 or not 0 < risk_per_trade <= 0.05:
        raise ValueError("Vốn hoặc tỷ lệ rủi ro không hợp lệ.")
    if max_positions < 1 or not 0 < max_position_pct <= 1 or lot_size < 1:
        raise ValueError("Giới hạn vị thế hoặc lô giao dịch không hợp lệ.")

    rows = [dict(x) for x in trades]
    rows = [x for x in rows if x.get("entry_date") and x.get("exit_date")]
    rows.sort(key=lambda x: (
        str(x.get("entry_date")), int(x.get("rank", 999)),
        -float(x.get("rank_score", 0) or 0), str(x.get("symbol", ""))
    ))

    cash = float(initial_capital)
    open_positions: list[dict] = []
    accepted, skipped = [], []
    reasons = defaultdict(int)
    peak_realized_equity = cash
    max_realized_drawdown = 0.0
    peak_open_risk = 0.0

    def close_positions_before(day: str) -> None:
        nonlocal cash, peak_realized_equity, max_realized_drawdown
        remaining = []
        for pos in open_positions:
            if str(pos["exit_date"]) < day:
                cash += pos["invested_cost"] + pos["net_pnl"]
                peak_realized_equity = max(peak_realized_equity, cash)
                if peak_realized_equity > 0:
                    max_realized_drawdown = min(
                        max_realized_drawdown, cash / peak_realized_equity - 1
                    )
            else:
                remaining.append(pos)
        open_positions[:] = remaining

    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        entry_date, exit_date = str(row["entry_date"]), str(row["exit_date"])
        close_positions_before(entry_date)

        def skip(reason: str) -> None:
            reasons[reason] += 1
            skipped.append({"symbol": symbol, "entry_date": entry_date, "reason": reason})

        if not symbol:
            skip("MISSING_SYMBOL")
            continue
        if exit_date < entry_date:
            skip("EXIT_BEFORE_ENTRY")
            continue
        if len(open_positions) >= max_positions:
            skip("MAX_CONCURRENT_POSITIONS")
            continue
        if any(p["symbol"] == symbol for p in open_positions):
            skip("DUPLICATE_OPEN_SYMBOL")
            continue

        try:
            entry = float(row["entry_price"])
            stop = float(row["stop_loss"])
            net_return = float(row["net_return_pct"]) / 100.0
            slip = float(row.get("slippage_rate", 0.001) or 0.001)
        except (KeyError, TypeError, ValueError):
            skip("MISSING_OR_INVALID_FIELDS")
            continue
        if (entry <= 0 or stop <= 0 or stop >= entry
                or not all(map(lambda v: v == v and abs(v) != float("inf"),
                               (entry, stop, net_return, slip)))
                or slip < 0 or slip > 0.02):
            skip("INVALID_PRICE_STOP_RETURN_OR_SLIPPAGE")
            continue

        buy_cost_per_share = entry * (1 + slip) * (1 + BUY_FEE)
        stop_proceeds_per_share = stop * (1 - slip) * (1 - SELL_FEE - SELL_TAX)
        risk_per_share = buy_cost_per_share - stop_proceeds_per_share
        if risk_per_share <= 0:
            skip("INVALID_EFFECTIVE_RISK")
            continue

        equity_for_sizing = cash + sum(p["invested_cost"] for p in open_positions)
        # Conviction scales risk down in neutral regimes and higher-risk setups.
        try:
            risk_multiplier = float(row["risk_factor"]) if "risk_factor" in row else 1.0
        except (TypeError, ValueError):
            skip("INVALID_RISK_FACTOR")
            continue
        if not (0 < risk_multiplier <= 1.0):
            skip("INVALID_RISK_FACTOR")
            continue
        risk_budget = equity_for_sizing * risk_per_trade * risk_multiplier
        allocation_cap = min(equity_for_sizing * max_position_pct, cash)
        shares_by_risk = floor(risk_budget / risk_per_share)
        shares_by_cash = floor(allocation_cap / buy_cost_per_share)
        quantity = min(shares_by_risk, shares_by_cash)
        quantity = (quantity // lot_size) * lot_size
        if quantity < lot_size:
            if shares_by_cash < lot_size:
                skip("INSUFFICIENT_CASH_FOR_BOARD_LOT_OR_ALLOCATION")
            else:
                skip("RISK_BUDGET_TOO_SMALL_FOR_BOARD_LOT")
            continue

        invested_cost = quantity * buy_cost_per_share
        net_pnl = invested_cost * net_return
        open_risk = quantity * risk_per_share
        pos = {
            "symbol": symbol, "setup": row.get("setup", ""),
            "entry_date": entry_date, "exit_date": exit_date,
            "quantity": quantity, "entry_price": round(entry, 2),
            "stop_loss": round(stop, 2),
            "invested_cost": invested_cost, "net_pnl": net_pnl,
            "net_return_pct": round(net_return * 100, 3),
            "risk_budget": round(risk_budget, 2),
            "risk_multiplier": round(risk_multiplier, 3),
            "estimated_initial_risk": round(open_risk, 2),
            "slippage_rate": slip,
        }
        cash -= invested_cost
        open_positions.append(pos)
        accepted.append(pos)
        peak_open_risk = max(
            peak_open_risk, sum(p["estimated_initial_risk"] for p in open_positions)
        )

    # Settle all remaining positions at their simulated exit dates.
    for pos in sorted(open_positions, key=lambda p: (p["exit_date"], p["entry_date"])):
        cash += pos["invested_cost"] + pos["net_pnl"]
        peak_realized_equity = max(peak_realized_equity, cash)
        if peak_realized_equity > 0:
            max_realized_drawdown = min(
                max_realized_drawdown, cash / peak_realized_equity - 1
            )

    total_pnl = cash - float(initial_capital)
    wins = [p for p in accepted if p["net_pnl"] > 0]
    losses = [p for p in accepted if p["net_pnl"] < 0]
    gross_profit = sum(p["net_pnl"] for p in wins)
    gross_loss = abs(sum(p["net_pnl"] for p in losses))
    return {
        "status": "SIMULATED" if accepted else "NO_TRADES_FIT_CASH_RISK_RULES",
        "initial_capital": round(float(initial_capital), 2),
        "ending_equity": round(cash, 2),
        "net_profit": round(total_pnl, 2),
        "return_pct": round(total_pnl / float(initial_capital) * 100, 3),
        "accepted_trades": len(accepted),
        "skipped_trades": len(skipped),
        "win_rate_pct": round(len(wins) / len(accepted) * 100, 2) if accepted else None,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "max_realized_drawdown_pct": round(max_realized_drawdown * 100, 3),
        "peak_open_initial_risk": round(peak_open_risk, 2),
        "max_positions": max_positions,
        "risk_per_trade_pct": round(risk_per_trade * 100, 2),
        "max_position_pct": round(max_position_pct * 100, 2),
        "lot_size": lot_size,
        "costs": {
            "buy_fee_pct": BUY_FEE * 100, "sell_fee_pct": SELL_FEE * 100,
            "sell_tax_pct": SELL_TAX * 100,
            "slippage": "per trade, supplied by strategy risk plan",
        },
        "skip_reasons": dict(reasons),
        "accepted_trade_details": accepted,
        "skipped_trade_details": skipped,
        "drawdown_note": (
            "Drawdown chỉ tính trên vốn đã hiện thực hóa khi vị thế đóng; "
            "chưa có chuỗi giá hằng ngày để tính drawdown thả nổi trong thời gian nắm giữ."
        ),
    }
