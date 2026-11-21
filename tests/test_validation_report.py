import unittest

import pandas as pd

from validation_report import build_validation_report, simulate_portfolio


class ValidationReportTests(unittest.TestCase):
    def test_early_alerts_are_not_counted_as_executable_buys(self):
        trades = pd.DataFrame([
            {"symbol": "AAA", "strategy": "PREPARE", "entry_date": "2026-01-02",
             "entry_price": 10, "stop_loss": None, "exit_date": "2026-01-08",
             "net_return_pct": -3.0},
            {"symbol": "BBB", "strategy": "EARLY_BREAKOUT", "entry_date": "2026-01-02",
             "entry_price": 20, "stop_loss": None, "exit_date": "2026-01-08",
             "net_return_pct": 2.0},
        ])
        report = build_validation_report(trades)
        self.assertEqual(report["executable_trading"]["trades"], 0)
        self.assertEqual(report["executable_trading"]["status"], "NO_CONFIRMED_BUY_SIGNALS")
        self.assertEqual(report["early_warning_research"]["trades"], 2)

    def test_portfolio_simulation_requires_stop_and_respects_lot_size(self):
        trades = pd.DataFrame([
            {"symbol": "AAA", "strategy": "SWING", "entry_date": "2026-01-02",
             "entry_price": 10, "stop_loss": 9, "exit_date": "2026-01-06",
             "net_return_pct": 5.0},
            {"symbol": "BBB", "strategy": "PULLBACK", "entry_date": "2026-01-02",
             "entry_price": 20, "stop_loss": 19, "exit_date": "2026-01-06",
             "net_return_pct": -2.0},
            {"symbol": "CCC", "strategy": "REVERSAL", "entry_date": "2026-01-02",
             "entry_price": 30, "stop_loss": None, "exit_date": "2026-01-06",
             "net_return_pct": 4.0},
        ])
        result = simulate_portfolio(trades, initial_capital=10_000_000)
        self.assertEqual(result["accepted_trades"], 2)
        self.assertEqual(result["skipped_trades"], 1)
        self.assertEqual(result["max_positions"], 5)

    def test_empty_confirmed_set_is_not_reported_as_profitable(self):
        report = build_validation_report(pd.DataFrame())
        self.assertEqual(report["executable_trading"]["status"], "NO_CONFIRMED_BUY_SIGNALS")
        self.assertIsNone(report["executable_trading"]["win_rate_pct"])


    def test_portfolio_reports_why_board_lot_was_skipped(self):
        trades = pd.DataFrame([{
            "symbol": "HIGH", "strategy": "SWING",
            "entry_date": "2026-01-02", "entry_price": 100_000,
            "stop_loss": 90_000, "exit_date": "2026-01-06",
            "net_return_pct": 5.0,
        }])
        result = simulate_portfolio(trades, initial_capital=10_000_000)
        self.assertEqual(result["accepted_trades"], 0)
        self.assertEqual(result["skipped_trades"], 1)
        self.assertEqual(
            result["skip_reasons"]["INSUFFICIENT_ALLOCATION_FOR_BOARD_LOT"], 1
        )


if __name__ == "__main__":
    unittest.main()
