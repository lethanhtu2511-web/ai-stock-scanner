import unittest
import numpy as np
import pandas as pd

from early_signal_engine import evaluate_early_signal


def make_history(n=90):
    idx = np.arange(n, dtype=float)
    close = 20 + idx * 0.08 + np.sin(idx / 4) * 0.08
    open_ = close - 0.03
    high = close + 0.12
    low = close - 0.12
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=n, freq="B"),
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


class EarlySignalTests(unittest.TestCase):
    def test_returns_expected_schema(self):
        result = evaluate_early_signal(make_history())
        self.assertIn(result["early_signal"], {"EARLY_BREAKOUT", "PREPARE", "NONE", "UNAVAILABLE"})
        self.assertGreaterEqual(result["early_score"], 0)
        self.assertLessEqual(result["early_score"], 100)
        self.assertIn("early_reasons", result)
        self.assertIn("early_risks", result)

    def test_short_history_is_unavailable(self):
        result = evaluate_early_signal(make_history(40))
        self.assertEqual(result["early_signal"], "UNAVAILABLE")
        self.assertEqual(result["early_score"], 0)

    def test_unverified_quote_is_never_counted_as_realtime(self):
        quote = {"price": 30.0, "quote_is_fresh": False}
        result = evaluate_early_signal(make_history(), quote=quote)
        self.assertFalse(result["early_quote_verified"])
        self.assertTrue(any("Chưa có quote realtime" in item for item in result["early_risks"]))

    def test_verified_quote_flag_requires_source_validation(self):
        quote = {"price": 30.0, "quote_is_fresh": True}
        result = evaluate_early_signal(make_history(), quote=quote)
        self.assertTrue(result["early_quote_verified"])


if __name__ == "__main__":
    unittest.main()
