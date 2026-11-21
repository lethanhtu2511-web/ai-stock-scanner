import unittest

from scanner import _cap_buy_recommendations
from strategy_engine import _make_result


class SignalQualityGateTests(unittest.TestCase):
    def test_buy_requires_four_confirmations(self):
        result = _make_result(90, ["a", "b", "c"], [], [], [])
        self.assertEqual(result["signal"], "CHỜ XÁC NHẬN")

    def test_buy_requires_no_blockers(self):
        result = _make_result(95, ["a", "b", "c", "d"], [], ["risk"], [])
        self.assertEqual(result["signal"], "CHỜ XÁC NHẬN")

    def test_market_buy_recommendations_capped_at_three(self):
        rows = [
            {"symbol": f"T{i}", "signal": "BUY", "score": 90-i,
             "reward_risk": 2.2, "relative_strength": 5, "avg_turnover_20d": 8}
            for i in range(5)
        ]
        _cap_buy_recommendations(rows, max_buy=3)
        self.assertEqual(sum(row["signal"] == "BUY" for row in rows), 3)
        self.assertEqual(sum(row["signal"] == "WAIT" for row in rows), 2)
        self.assertTrue(all(row.get("recommendation_rank") for row in rows[:3]))

    def test_zero_cap_means_no_buy_recommendations(self):
        rows = [{"signal": "BUY", "score": 90}, {"signal": "BUY", "score": 80}]
        _cap_buy_recommendations(rows, max_buy=0)
        self.assertFalse(any(row["signal"] == "BUY" for row in rows))


if __name__ == "__main__":
    unittest.main()
