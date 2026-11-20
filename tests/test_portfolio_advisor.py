import unittest

from portfolio_advisor import advise_position


class PortfolioAdvisorTests(unittest.TestCase):
    def setUp(self):
        self.position = {"symbol": "ABC", "quantity": 1000, "avg_cost": 20000}
        self.settings = {"fee_sell": 0.0015, "sell_tax": 0.001}

    def test_stop_loss_recommends_full_exit(self):
        rows = [{"stop_loss": 19.0, "target_price": 24.0, "signal": "BUY"}]
        result = advise_position(self.position, {"price": 18_900, "price_status": "REALTIME"}, rows, self.settings)
        self.assertEqual(result["action"], "SELL_ALL")
        self.assertEqual(result["quantity"], 1000)

    def test_target_recommends_partial_profit(self):
        rows = [{"stop_loss": 18.0, "target_price": 22.0, "signal": "BUY"}]
        result = advise_position(self.position, {"price": 22_100, "price_status": "REALTIME"}, rows, self.settings)
        self.assertEqual(result["action"], "SELL_PART")
        self.assertEqual(result["quantity"], 500)

    def test_bearish_consensus_recommends_partial_reduction(self):
        rows = [
            {"signal": "SKIP", "score": 40, "market_regime": "BEARISH"},
            {"signal": "FILTERED", "score": 30, "market_regime": "BEARISH"},
            {"signal": "WATCH", "score": 60, "market_regime": "BEARISH"},
        ]
        result = advise_position(self.position, {"price": 19_500, "price_status": "REALTIME"}, rows, self.settings)
        self.assertEqual(result["action"], "SELL_PART")
        self.assertEqual(result["quantity"], 500)

    def test_stale_price_never_recommends_sell(self):
        result = advise_position(self.position, {"price": None, "price_status": "D1_STALE"}, [], self.settings)
        self.assertEqual(result["action"], "REVIEW")
        self.assertEqual(result["quantity"], 0)

    def test_one_lot_cannot_be_partially_sold(self):
        position = {"symbol": "ABC", "quantity": 100, "avg_cost": 20_000}
        rows = [{"stop_loss": 18.0, "target_price": 22.0, "signal": "BUY"}]
        result = advise_position(position, {"price": 22_100, "price_status": "REALTIME"}, rows, self.settings)
        self.assertEqual(result["action"], "SELL_ALL")
        self.assertEqual(result["quantity"], 100)


if __name__ == "__main__":
    unittest.main()
