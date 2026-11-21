import unittest
import pandas as pd
import numpy as np

from pattern_research import extract_point_in_time_features, label_forward_outcome, analyze_patterns, walk_forward_validation


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


    def test_walk_forward_uses_later_holdout_dates(self):
        dates = pd.bdate_range("2025-01-01", periods=100)
        rows = []
        for strategy in ("SWING", "REVERSAL", "PULLBACK"):
            for i, date in enumerate(dates):
                rows.append({
                    "strategy": strategy, "date": date.strftime("%Y-%m-%d"),
                    "exit_date": (date + pd.offsets.BDay(5)).strftime("%Y-%m-%d"),
                    "net_return_pct": 2.0 if i % 2 else -1.0,
                    "mfe_pct": 4.0, "mae_pct": -2.0, "quality_win": False,
                    "above_ema20": i % 2 == 0, "ema20_above_ema50": True,
                    "volume_expansion": i % 3 == 0, "foreign_net_buy_5d": i % 4 == 0,
                    "market_regime": "BULLISH", "rsi14": float(40 + i % 50),
                    "ret_1d_pct": float(i % 5), "ret_5d_pct": float(i % 7),
                    "ret_10d_pct": float(i % 9), "ret_20d_pct": float(i % 11),
                    "dist_ema20_pct": float(i % 6), "dist_ema50_pct": float(i % 8),
                    "dist_ma200_pct": float(i % 10), "macd_hist_pct": float(i % 4),
                    "volume_ratio20": float(0.5 + i % 5), "turnover20_bn": float(5 + i % 6),
                    "atr14_pct": float(1 + i % 3), "body_pct": float(-1 + i % 4),
                    "close_location": float(0.5 + (i % 5) / 10),
                    "foreign_net5_ratio_pct": float(-2 + i % 5),
                })
        result = walk_forward_validation(pd.DataFrame(rows))
        self.assertEqual(result["status"], "HOLDOUT_EVALUATED")
        self.assertGreater(result["holdout_observations"], 0)
        self.assertGreaterEqual(result["holdout_date_range"][0], result["discovery_cutoff"])

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
