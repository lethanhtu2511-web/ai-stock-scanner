import unittest
from unittest.mock import patch

import pandas as pd

import historical_backtest as hb


class HistoricalBacktestTests(unittest.TestCase):
    def test_portfolio_caps_concurrent_positions_and_sizes_equally(self):
        trades = pd.DataFrame([
            {'entry_date': '2025-01-02', 'exit_date': '2025-01-06', 'net_return_pct': 10.0},
            {'entry_date': '2025-01-03', 'exit_date': '2025-01-07', 'net_return_pct': -5.0},
            {'entry_date': '2025-01-04', 'exit_date': '2025-01-08', 'net_return_pct': 4.0},
            {'entry_date': '2025-01-05', 'exit_date': '2025-01-09', 'net_return_pct': 2.0},
            {'entry_date': '2025-01-06', 'exit_date': '2025-01-10', 'net_return_pct': 3.0},
            {'entry_date': '2025-01-06', 'exit_date': '2025-01-11', 'net_return_pct': 8.0},
        ])
        result = hb._simulate_portfolio(trades, initial_capital=10_000_000, max_positions=5)
        self.assertEqual(result['accepted_trades'], 5)
        self.assertEqual(result['skipped_capacity'], 1)
        self.assertEqual(result['max_concurrent_positions'], 5)
        self.assertGreater(result['ending_equity'], 0)
        self.assertEqual(result['initial_capital'], 10_000_000)

    def test_costs_reduce_gross_return(self):
        gross = 10 / 100
        net = hb._net_return(100, 110, hb.DEFAULT_COSTS)
        self.assertLess(net, gross)
        self.assertGreater(net, 0)

    def test_stop_is_prioritized_when_stop_and_target_hit_same_bar(self):
        dates = pd.bdate_range("2025-01-01", periods=8)
        frame = pd.DataFrame({
            "date": dates,
            "open": [100, 100, 100, 100, 100, 100, 100, 100],
            "high": [101, 106, 101, 101, 101, 101, 101, 101],
            "low": [99, 94, 99, 99, 99, 99, 99, 99],
            "close": [100] * 8,
            "volume": [100000] * 8,
        })
        trade = hb._simulate_trade(
            "TST", "SWING", frame, 0,
            {"stop_loss": 95, "target_price": 105},
            hb.DEFAULT_COSTS,
        )
        self.assertEqual(trade["exit_reason"], "STOP LOSS")
        self.assertEqual(trade["holding_sessions"], 1)

    def test_early_breakout_is_backtested_separately(self):
        dates = pd.bdate_range("2025-01-01", periods=240)
        close = [100 + i * 0.05 for i in range(240)]
        frame = pd.DataFrame({
            "date": dates,
            "open": close,
            "high": [x * 1.01 for x in close],
            "low": [x * 0.99 for x in close],
            "close": close,
            "volume": [1_000_000] * 240,
        })
        with patch.object(hb, "fetch_stock_history", return_value=frame), \
             patch.object(hb, "evaluate_early_signal", return_value={
                 "early_signal": "EARLY_BREAKOUT"
             }):
            result = hb.run_historical_backtest(
                symbols=["TST"],
                lookback_sessions=20,
                max_symbols=1,
                strategy_filter="EARLY",
            )
        self.assertGreater(result["summary"]["trades"], 0)
        self.assertTrue(
            set(result["trades"]["strategy"]) == {"EARLY_BREAKOUT"}
        )
        self.assertTrue(
            (result["trades"]["holding_sessions"] <= 5).all()
        )

    def test_walk_forward_uses_only_past_bars_and_full_horizon(self):
        dates = pd.bdate_range("2025-01-01", periods=240)
        close = [100 + i * 0.05 for i in range(240)]
        frame = pd.DataFrame({
            "date": dates,
            "open": close,
            "high": [x * 1.01 for x in close],
            "low": [x * 0.99 for x in close],
            "close": close,
            "volume": [1_000_000] * 240,
        })
        seen_lengths = []

        def fake_analyze(symbol, history, market_df=None, foreign_flow=None):
            seen_lengths.append(len(history))
            return {
                "strategies": {
                    key: {
                        "signal": "BUY", "confirmed": True,
                        "stop_loss": None, "target_price": None,
                    }
                    for key in hb.STRATEGY_HORIZONS
                }
            }

        with patch.object(hb, "fetch_exchange_symbols", return_value=["TST"]), \
             patch.object(hb, "fetch_stock_history", return_value=frame), \
             patch.object(hb, "analyze_historical_signal", side_effect=fake_analyze):
            result = hb.run_historical_backtest(
                lookback_sessions=20,
                max_symbols=1,
                strategy_filter="SWING",
            )

        self.assertTrue(seen_lengths)
        self.assertTrue(all(length <= 240 for length in seen_lengths))
        self.assertGreater(result["summary"]["trades"], 0)
        for trade in result["trades"].to_dict("records"):
            self.assertLess(trade["signal_date"], trade["entry_date"])
            self.assertLessEqual(trade["holding_sessions"], 5)


if __name__ == "__main__":
    unittest.main()
