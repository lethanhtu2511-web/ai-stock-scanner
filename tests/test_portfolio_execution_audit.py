import unittest
from portfolio_execution_audit import simulate_cash_portfolio_audited


class PortfolioAuditTests(unittest.TestCase):
    def trade(self, symbol, entry, exit_, ret=-1.0, rank=1):
        return {"symbol": symbol, "entry_date": entry, "exit_date": exit_,
                "entry_price": 10000, "stop_loss": 9500,
                "net_return_pct": ret, "slippage_rate": 0.001,
                "risk_factor": 1.0, "rank": rank}

    def test_open_cost_basis_is_not_counted_as_drawdown(self):
        rows = [self.trade("AAA", "2026-01-02", "2026-01-06", -1)]
        result = simulate_cash_portfolio_audited(rows)
        self.assertEqual(result["max_realized_drawdown_pct"], 0.0)
        self.assertEqual(result["accepted_trades"], 1)

    def test_same_symbol_cannot_be_reopened_before_exit(self):
        rows = [self.trade("AAA", "2026-01-02", "2026-01-06"),
                self.trade("AAA", "2026-01-03", "2026-01-07", rank=2)]
        result = simulate_cash_portfolio_audited(rows)
        self.assertEqual(result["accepted_trades"], 1)
        self.assertEqual(result["skip_reasons"].get("DUPLICATE_OPEN_SYMBOL"), 1)

    def test_losses_are_reflected_after_realized_exit(self):
        rows = [self.trade("AAA", "2026-01-02", "2026-01-03", -5)]
        result = simulate_cash_portfolio_audited(rows)
        self.assertLess(result["max_realized_drawdown_pct"], 0)
        self.assertAlmostEqual(result["ending_equity"], 10_000_000 + result["net_profit"], places=2)


if __name__ == "__main__":
    unittest.main()
