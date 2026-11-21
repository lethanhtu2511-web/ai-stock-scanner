import unittest
import pandas as pd
import numpy as np

from pattern_research import extract_point_in_time_features, label_forward_outcome, analyze_patterns


def sample_history(n=260):
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = np.linspace(10, 20, n)
    return pd.DataFrame({
        "date": dates, "open": close * 0.999, "high": close * 1.02,
        "low": close * 0.98, "close": close, "volume": np.full(n, 1_000_000),
    })


class PatternResearchTests(unittest.TestCase):
    def test_features_are_point_in_time(self):
        d = sample_history()
        before = extract_point_in_time_features(d, 220)
        d.loc[221:, ["open","high","low","close","volume"]] *= 10
        after = extract_point_in_time_features(d, 220)
        self.assertEqual(before["ret_5d_pct"], after["ret_5d_pct"])
        self.assertEqual(before["rsi14"], after["rsi14"])

    def test_outcome_starts_next_session(self):
        d = sample_history()
        result = label_forward_outcome(d, 220, 5)
        self.assertEqual(result["entry_date"], d.iloc[221]["date"].strftime("%Y-%m-%d"))
        self.assertEqual(result["exit_date"], d.iloc[225]["date"].strftime("%Y-%m-%d"))
        self.assertGreater(result["net_return_pct"], 0)

    def test_report_separates_strategies_and_requires_sample(self):
        d = sample_history()
        rows = []
        for strategy in ("SWING","REVERSAL","PULLBACK"):
            for i in range(30):
                rows.append({
                    "strategy": strategy, "net_return_pct": 2.0, "mfe_pct": 4.0,
                    "mae_pct": -1.0, "quality_win": False,
                    "above_ema20": True, "ema20_above_ema50": True,
                    "volume_expansion": False, "foreign_net_buy_5d": False,
                    "close_location": 0.8, "market_regime": "BULLISH",
                })
        report = analyze_patterns(pd.DataFrame(rows))
        self.assertEqual(report["by_strategy"]["SWING"]["samples"], 30)
        self.assertTrue(report["confluence_patterns"])


if __name__ == "__main__":
    unittest.main()
