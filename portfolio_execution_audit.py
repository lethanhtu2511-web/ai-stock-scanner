"""Independent audit of the research cash portfolio simulator.

Drawdown is measured on realized equity = cash + cost basis of open positions.
This avoids treating cash deployed into open positions as an equity loss.
No broker integration or production trading changes.
"""
from __future__ import annotations
from collections import defaultdict
from math import floor

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001


def simulate_cash_portfolio_audited(
    trades, initial_capital=10_000_000, risk_per_trade=0.02,
    max_positions=5, max_position_pct=0.20, lot_size=100,
):
    if initial_capital <= 0 or not 0 < risk_per_trade <= 0.05:
        raise ValueError("Vốn hoặc tỷ lệ rủi ro không hợp lệ.")
    if max_positions < 1 or not 0 < max_position_pct <= 1 or lot_size < 1:
        raise ValueError("Giới hạn vị thế hoặc lô giao dịch không hợp lệ.")
    rows = [dict(x) for x in trades if x.get("entry_date") and x.get("exit_date")]
    rows.sort(key=lambda x: (str(x["entry_date"]), int(x.get("rank", 999)),
                             -float(x.get("rank_score", 0) or 0), str(x.get("symbol", ""))))
    cash = float(initial_capital)
    opened, accepted, skipped = [], [], []
    reasons = defaultdict(int)
    peak_equity, max_dd, peak_open_risk = cash, 0.0, 0.0

    def equity():
        return cash + sum(p["invested_cost"] for p in opened)

    def settle_before(day):
        nonlocal cash, peak_equity, max_dd
        due_dates = sorted({p["exit_date"] for p in opened if p["exit_date"] < day})
        for exit_day in due_dates:
            due = [p for p in opened if p["exit_date"] == exit_day]
            for p in due:
                cash += p["invested_cost"] + p["net_pnl"]
            opened[:] = [p for p in opened if p["exit_date"] != exit_day]
            current = equity()
            peak_equity = max(peak_equity, current)
            if peak_equity > 0:
                max_dd = min(max_dd, current / peak_equity - 1)

    for row in rows:
        symbol, entry_day, exit_day = str(row.get("symbol", "")).upper(), str(row["entry_date"]), str(row["exit_date"])
        settle_before(entry_day)
        def skip(reason):
            reasons[reason] += 1
            skipped.append({"symbol": symbol, "entry_date": entry_day, "reason": reason})
        if not symbol:
            skip("MISSING_SYMBOL"); continue
        if exit_day < entry_day:
            skip("EXIT_BEFORE_ENTRY"); continue
        if len(opened) >= max_positions:
            skip("MAX_CONCURRENT_POSITIONS"); continue
        if any(p["symbol"] == symbol for p in opened):
            skip("DUPLICATE_OPEN_SYMBOL"); continue
        try:
            entry, stop = float(row["entry_price"]), float(row["stop_loss"])
            ret = float(row["net_return_pct"]) / 100
            slip = float(row.get("slippage_rate", 0.001) or 0.001)
            multiplier = float(row.get("risk_factor", 1.0))
        except (KeyError, TypeError, ValueError):
            skip("MISSING_OR_INVALID_FIELDS"); continue
        vals = (entry, stop, ret, slip, multiplier)
        if (entry <= 0 or stop <= 0 or stop >= entry or
            any(v != v or abs(v) == float("inf") for v in vals) or
            not 0 <= slip <= 0.02 or not 0 < multiplier <= 1):
            skip("INVALID_PRICE_STOP_RETURN_SLIPPAGE_OR_RISK"); continue
        buy_ps = entry * (1 + slip) * (1 + BUY_FEE)
        stop_ps = stop * (1 - slip) * (1 - SELL_FEE - SELL_TAX)
        risk_ps = buy_ps - stop_ps
        if risk_ps <= 0:
            skip("INVALID_EFFECTIVE_RISK"); continue
        sizing_equity = equity()
        budget = sizing_equity * risk_per_trade * multiplier
        allocation = min(sizing_equity * max_position_pct, cash)
        by_risk, by_cash = floor(budget / risk_ps), floor(allocation / buy_ps)
        qty = min(by_risk, by_cash)
        qty = (qty // lot_size) * lot_size
        if qty < lot_size:
            skip("INSUFFICIENT_CASH_OR_RISK_FOR_BOARD_LOT"); continue
        cost = qty * buy_ps
        pnl = cost * ret
        risk = qty * risk_ps
        pos = {"symbol": symbol, "setup": row.get("setup", ""),
               "entry_date": entry_day, "exit_date": exit_day, "quantity": qty,
               "entry_price": round(entry, 2), "stop_loss": round(stop, 2),
               "invested_cost": cost, "net_pnl": pnl,
               "net_return_pct": round(ret * 100, 3), "risk_budget": round(budget, 2),
               "estimated_initial_risk": round(risk, 2)}
        cash -= cost
        opened.append(pos); accepted.append(pos)
        peak_open_risk = max(peak_open_risk, sum(p["estimated_initial_risk"] for p in opened))

    for exit_day in sorted({p["exit_date"] for p in opened}):
        due = [p for p in opened if p["exit_date"] == exit_day]
        for p in due:
            cash += p["invested_cost"] + p["net_pnl"]
        opened[:] = [p for p in opened if p["exit_date"] != exit_day]
        current = equity()
        peak_equity = max(peak_equity, current)
        if peak_equity > 0:
            max_dd = min(max_dd, current / peak_equity - 1)

    pnl_total = cash - initial_capital
    wins = [p for p in accepted if p["net_pnl"] > 0]
    losses = [p for p in accepted if p["net_pnl"] < 0]
    gp, gl = sum(p["net_pnl"] for p in wins), abs(sum(p["net_pnl"] for p in losses))
    return {
        "status": "SIMULATED" if accepted else "NO_TRADES_FIT_CASH_RISK_RULES",
        "initial_capital": round(initial_capital, 2), "ending_equity": round(cash, 2),
        "net_profit": round(pnl_total, 2), "return_pct": round(pnl_total / initial_capital * 100, 3),
        "accepted_trades": len(accepted), "skipped_trades": len(skipped),
        "win_rate_pct": round(len(wins) / len(accepted) * 100, 2) if accepted else None,
        "profit_factor": round(gp / gl, 3) if gl else None,
        "max_realized_drawdown_pct": round(max_dd * 100, 3),
        "peak_open_initial_risk": round(peak_open_risk, 2),
        "max_positions": max_positions, "risk_per_trade_pct": risk_per_trade * 100,
        "max_position_pct": max_position_pct * 100, "lot_size": lot_size,
        "cost_basis_equity_method": "cash + entry cost basis of open positions; no mark-to-market P/L",
        "skip_reasons": dict(reasons), "accepted_trade_details": accepted,
        "skipped_trade_details": skipped,
    }
