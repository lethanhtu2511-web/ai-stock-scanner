import unittest

import pandas as pd

from candlestick_patterns import detect_candlestick_patterns


def sample_frame(last):
    rows = []
    for i in range(30):
        price = 100.0
        rows.append({
            "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1_000_000,
        })
    rows[-1].update(last)
    return pd.DataFrame(rows)


class CandlestickPatternTests(unittest.TestCase):
    def test_hammer_at_support_is_context_confirmed(self):
        df = sample_frame({
            "open": 100.5, "high": 101.0, "low": 97.0,
            "close": 100.5, "volume": 1_200_000,
        })
        result = detect_candlestick_patterns(df)
        self.assertIn("Hammer", result["patterns"])
        self.assertIn("Hammer", result["confirmed_bullish"])

    def test_shooting_star_is_detected(self):
        df = sample_frame({
            "open": 100.5, "high": 104.5, "low": 100.0,
            "close": 100.4, "volume": 1_200_000,
        })
        result = detect_candlestick_patterns(df)
        self.assertIn("Shooting Star", result["patterns"])

    def test_doji_is_neutral_not_bullish(self):
        df = sample_frame({
            "open": 100.0, "high": 102.0, "low": 98.0,
            "close": 100.05, "volume": 1_000_000,
        })
        result = detect_candlestick_patterns(df)
        self.assertIn("Doji", result["neutral"])
        self.assertNotIn("Doji", result["bullish"])

    def test_insufficient_data_returns_safe_result(self):
        result = detect_candlestick_patterns(pd.DataFrame())
        self.assertEqual(result["context"], "INSUFFICIENT_DATA")
        self.assertEqual(result["patterns"], [])


if __name__ == "__main__":
    unittest.main()
