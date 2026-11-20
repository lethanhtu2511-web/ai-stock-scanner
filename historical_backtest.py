"""Walk-forward historical backtest for AI Stock Scanner.

Signals are calculated using data available through session t only.
Execution is simulated at session t+1 open with costs and slippage.
No future bars are used to create a signal.
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from data_provider import fetch_exchange_symbols, fetch_stock_history
from strategy_engine import analyze_stock


DEFAULT_COSTS = {
    "buy_fee": 0.0015,
    "sell_fee": 0.0015,
    "sell_tax": 0.0010,
    "slippage": 0.0020,
}
STRATEGY_HORIZONS = {"SWING": 5, "REVERSAL": 10, "PULLBACK": 7}


def _clean_history(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d.columns and "Date" in raw.columns:
        d["date"] = pd.to_datetime(raw["Date"], errors="coerce")
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError("Thiếu cột dữ liệu: " + ", ".join(missing))
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for col in required[1:]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=required).sort_values("date")
    d = d.drop_duplicates("date", keep="last").reset_index(drop=True)
    return d


def _net_return(entry: float, exit_price: float, costs: dict) -> float:
    buy_cost = entry * (1 + costs["slippage"]) * (1 + costs["buy_fee"])
    sell_proceeds = exit_price * (1 - costs["slippage"]) * (
        1 - costs["sell_fee"] - costs["sell_tax"]
    )
    return sell_proceeds / buy_cost - 1 if buy_cost > 0 else 0.0


def _simulate_trade(symbol: str, strategy: str, d: pd.DataFrame,
                    signal_idx: int, result: dict, costs: dict) -> dict | None:
    entry_idx = signal_idx + 1
    if entry_idx >= len(d):
        return None
    entry_bar = d.iloc[entry_idx]
    entry_raw = float(entry_bar["open"])
    if not math.isfinite(entry_raw) or entry_raw <= 0:
        return None

    entry = entry_raw * (1 + costs["slippage"])
    stop = float(result.get("stop_loss") or 0)
    target = float(result.get("target_price") or 0)
    horizon = STRATEGY_HORIZONS[strategy]
    last_idx = min(len(d) - 1, entry_idx + horizon - 1)
    exit_raw = float(d.iloc[last_idx]["close"])
    exit_idx = last_idx
    exit_reason = "HẾT KỲ NẮM GIỮ"

    for idx in range(entry_idx, last_idx + 1):
        bar = d.iloc[idx]
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        # Conservative gap handling and stop-first rule when both levels are hit.
        if stop > 0 and o <= stop:
            exit_raw, exit_idx, exit_reason = o, idx, "GAP DƯỚI STOP"
            break
        if target > 0 and o >= target:
            exit_raw, exit_idx, exit_reason = o, idx, "GAP TRÊN TARGET"
            break
        stop_hit = stop > 0 and l <= stop
        target_hit = target > 0 and h >= target
        if stop_hit:
            exit_raw, exit_idx, exit_reason = stop, idx, "STOP LOSS"
            break
        if target_hit:
            exit_raw, exit_idx, exit_reason = target, idx, "TAKE PROFIT"
            break

    exit_fill = exit_raw * (1 - costs["slippage"])
    net = _net_return(entry_raw, exit_raw, costs)
    signal_date = pd.Timestamp(d.iloc[signal_idx]["date"]).strftime("%Y-%m-%d")
    entry_date = pd.Timestamp(d.iloc[entry_idx]["date"]).strftime("%Y-%m-%d")
    exit_date = pd.Timestamp(d.iloc[exit_idx]["date"]).strftime("%Y-%m-%d")
    return {
        "symbol": symbol,
        "strategy": strategy,
        "signal_date": signal_date,
        "signal_close": round(float(d.iloc[signal_idx]["close"]), 3),
        "entry_date": entry_date,
        "entry_price": round(entry, 3),
        "stop_loss": round(stop, 3) if stop else None,
        "target_price": round(target, 3) if target else None,
        "exit_date": exit_date,
        "exit_price": round(exit_fill, 3),
        "exit_reason": exit_reason,
        "holding_sessions": exit_idx - entry_idx + 1,
        "net_return_pct": round(net * 100, 3),
        "win": bool(net > 0),
    }


def _summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0, "win_rate_pct": 0.0, "avg_return_pct": 0.0,
            "median_return_pct": 0.0, "profit_factor": None,
            "max_drawdown_pct": 0.0, "avg_holding_sessions": 0.0,
            "total_return_pct": 0.0,
        }
    returns = trades["net_return_pct"].astype(float) / 100
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    return {
        "trades": int(len(trades)),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2),
        "avg_return_pct": round(float(returns.mean() * 100), 3),
        "median_return_pct": round(float(returns.median() * 100), 3),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
        "max_drawdown_pct": round(float(drawdown.min() * 100), 3),
        "avg_holding_sessions": round(float(trades["holding_sessions"].mean()), 2),
        "total_return_pct": round(float((equity.iloc[-1] - 1) * 100), 3),
    }


def run_historical_backtest(
    symbols: Iterable[str] | None = None,
    lookback_sessions: int = 120,
    max_symbols: int = 50,
    strategy_filter: str = "ALL",
    costs: dict | None = None,
    progress_callback=None,
) -> dict:
    """Run a chronological walk-forward test on the latest available D1 history."""
    costs_used = {**DEFAULT_COSTS, **(costs or {})}
    for key, value in costs_used.items():
        costs_used[key] = float(value)
        if costs_used[key] < 0:
            raise ValueError(f"Chi phí {key} không được âm.")

    universe = list(symbols) if symbols is not None else fetch_exchange_symbols()
    universe = list(dict.fromkeys(str(s).strip().upper() for s in universe if str(s).strip()))
    if not universe:
        raise ValueError("Không có danh sách mã để backtest.")

    histories, errors = {}, []
    ranked = []
    for i, symbol in enumerate(universe):
        try:
            raw = fetch_stock_history(symbol, length=600)
            d = _clean_history(raw)
            if len(d) < 220:
                raise ValueError(f"Chỉ có {len(d)} phiên, cần ít nhất 220.")
            avg_turnover = float((d["close"] * d["volume"]).tail(20).mean())
            histories[symbol] = d
            ranked.append((symbol, avg_turnover))
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
        if progress_callback:
            progress_callback(i + 1, len(universe), f"Tải dữ liệu {symbol}")

    ranked.sort(key=lambda item: item[1], reverse=True)
    selected = [symbol for symbol, _ in ranked[:max(1, int(max_symbols))]]
    all_trades = []
    strategy_keys = list(STRATEGY_HORIZONS) if strategy_filter == "ALL" else [strategy_filter]
    invalid_strategy = [s for s in strategy_keys if s not in STRATEGY_HORIZONS]
    if invalid_strategy:
        raise ValueError("Chiến lược không hợp lệ: " + ", ".join(invalid_strategy))

    for symbol_index, symbol in enumerate(selected):
        d = histories[symbol].reset_index(drop=True)
        start_idx = max(200, len(d) - max(1, int(lookback_sessions)) - max(STRATEGY_HORIZONS.values()))
        for signal_idx in range(start_idx, len(d) - 1):
            # Strict point-in-time slice: no bar after signal_idx enters the model.
            point_in_time = d.iloc[:signal_idx + 1].copy()
            for strategy in strategy_keys:
                try:
                    analysis = analyze_stock(symbol, point_in_time)
                    item = analysis["strategies"][strategy]
                    if item["signal"] == "MUA" and item["confirmed"]:
                        trade = _simulate_trade(
                            symbol, strategy, d, signal_idx, analysis, costs_used
                        )
                        if trade:
                            all_trades.append(trade)
                except Exception:
                    continue
        if progress_callback:
            progress_callback(
                len(universe) + symbol_index + 1,
                len(universe) + len(selected),
                f"Backtest {symbol}",
            )

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(
            ["entry_date", "symbol", "strategy"]
        ).reset_index(drop=True)

    by_strategy = []
    if not trades_df.empty:
        for strategy, group in trades_df.groupby("strategy"):
            by_strategy.append({"strategy": strategy, **_summarize(group)})
    return {
        "summary": _summarize(trades_df),
        "by_strategy": by_strategy,
        "trades": trades_df,
        "symbols_requested": len(universe),
        "symbols_tested": len(selected),
        "data_errors": errors,
        "selected_symbols": selected,
        "costs": costs_used,
        "lookback_sessions": int(lookback_sessions),
        "method_note": (
            "Tín hiệu được tính bằng dữ liệu đến hết phiên T; mô phỏng mua tại giá mở cửa T+1. "
            "Tính phí mua 0,15%, phí bán 0,15%, thuế bán 0,10% và trượt giá 0,20% mỗi chiều. "
            "Nếu cùng một nến chạm cả stop và target, giả định stop khớp trước để thận trọng. "
            "Đây là backtest D1 mô phỏng, chưa phản ánh độ sâu sổ lệnh, giới hạn giá, thanh khoản "
            "khớp lệnh thực tế, cổ phiếu hủy niêm yết và corporate actions đã điều chỉnh."
        ),
    }
