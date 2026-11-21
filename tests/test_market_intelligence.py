import unittest
import pandas as pd

from market_intelligence import calculate_market_breadth, calculate_sector_strength
from fundamental_analysis import assess_fundamentals


class MarketIntelligenceTests(unittest.TestCase):
    def test_breadth_reports_sample_and_bullish(self):
        dates = pd.bdate_range("2026-01-01", periods=80)
        frame = pd.DataFrame({"date": dates, "close": range(100, 180)})
        result = calculate_market_breadth({"AAA": frame, "BBB": frame})
        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(result["regime"], "BROAD_BULLISH")
        self.assertEqual(result["sample_size"], 2)

    def test_breadth_missing_data_is_unavailable(self):
        self.assertEqual(calculate_market_breadth({})["status"], "UNAVAILABLE")

    def test_sector_strength_requires_explicit_mapping(self):
        dates = pd.bdate_range("2026-01-01", periods=30)
        up = pd.DataFrame({"date": dates, "close": range(100, 130)})
        down = pd.DataFrame({"date": dates, "close": range(130, 100, -1)})
        result = calculate_sector_strength({"AAA": up, "BBB": down}, {"AAA": "Bank", "BBB": "Steel"})
        self.assertEqual(result["sectors"][0]["sector"], "Bank")
        self.assertEqual(result["status"], "AVAILABLE")


class FundamentalAnalysisTests(unittest.TestCase):
    def test_missing_data_never_gets_score(self):
        result = assess_fundamentals({})
        self.assertIsNone(result["score"])
        self.assertIn(result["status"], ("UNAVAILABLE", "INCOMPLETE"))

    def test_complete_positive_metrics_get_score(self):
        result = assess_fundamentals({
            "revenue_growth_yoy": 10, "net_profit_growth_yoy": 20,
            "roe": 18, "debt_to_equity": 0.5,
            "operating_cash_flow": 1000000, "period_end": "2026-09-30",
        }, as_of=pd.Timestamp("2026-11-21").date())
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["status"], "AVAILABLE")


if __name__ == "__main__":
    unittest.main()
