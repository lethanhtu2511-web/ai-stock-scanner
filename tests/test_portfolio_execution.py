import unittest

from portfolio_execution import simulate_cash_portfolio


class CashPortfolioTests(unittest.TestCase):
    def trade(self, symbol="AAA", entry_date="2026-01-02", exit_date="2026-01-06",
              entry=10000, stop=9500, ret=5.0, rank=1):
        return {
            "symbol": symbol, "setup": "PULLBACK", "entry_date": entry_date,
            "exit_date": exit_date, "entry_price": entry, "stop_loss": stop,
            "net_return_pct": ret, "rank": rank, "slippage_rate": 0.001,
        }

    def test_sizes_in_board_lots_and_respects_allocation(self):
        result = simulate_cash_portfolio([self.trade()], initial_capital=10_000_000)
        self.assertEqual(result["accepted_trades"], 1)
        pos = result["accepted_trade_details"][0]
        self.assertEqual(pos["quantity"] % 100, 0)
        self.assertLessEqual(pos["invested_cost"], 2_000_000)
        self.assertGreater(result["ending_equity"], 10_000_000)

    def test_rejects_trade_when_one_board_lot_exceeds_allocation(self):
        result = simulate_cash_portfolio(
            [self.trade(entry=25000, stop=24000)],
            initial_capital=10_000_000,
        )
        self.assertEqual(result["accepted_trades"], 0)
        self.assertEqual(result["skip_reasons"]["INSUFFICIENT_CASH_FOR_BOARD_LOT_OR_ALLOCATION"], 1)

    def test_cash_is_reserved_and_concurrent_positions_are_limited(self):
        trades = [
            self.trade("AAA", "2026-01-02", "2026-01-10", rank=1),
            self.trade("BBB", "2026-01-02", "2026-01-10", rank=2),
            self.trade("CCC", "2026-01-02", "2026-01-10", rank=3),
            self.trade("DDD", "2026-01-02", "2026-01-10", rank=4),
            self.trade("EEE", "2026-01-02", "2026-01-10", rank=5),
            self.trade("FFF", "2026-01-02", "2026-01-10", rank=6),
        ]
        result = simulate_cash_portfolio(trades, initial_capital=10_000_000, max_positions=5)
        self.assertLessEqual(result["accepted_trades"], 5)
        self.assertGreaterEqual(result["skipped_trades"], 1)

    def test_duplicate_symbol_cannot_be_opened_twice(self):
        trades = [
            self.trade("AAA", "2026-01-02", "2026-01-10", rank=1),
            self.trade("AAA", "2026-01-03", "2026-01-08", rank=2),
        ]
        result = simulate_cash_portfolio(trades, initial_capital=10_000_000)
        self.assertEqual(result["accepted_trades"], 1)
        self.assertEqual(result["skip_reasons"]["DUPLICATE_OPEN_SYMBOL"], 1)

    def test_invalid_stop_is_rejected(self):
        result = simulate_cash_portfolio(
            [self.trade(stop=11000)], initial_capital=10_000_000
        )
        self.assertEqual(result["accepted_trades"], 0)
        self.assertEqual(result["skip_reasons"]["INVALID_PRICE_STOP_RETURN_OR_SLIPPAGE"], 1)


if __name__ == "__main__":
    unittest.main()
