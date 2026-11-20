import unittest
from signal_evaluator import evaluate_signal_history


def row(day, close, signal="WAIT", early="NONE", scanned_at=None):
    return {
        "symbol": "AAA",
        "strategy": "SWING",
        "data_date": day,
        "d1_close": close,
        "price": close,
        "signal": signal,
        "early_signal": early,
        "scanned_at": scanned_at or day,
    }


class SignalEvaluatorTests(unittest.TestCase):
    def test_deduplicates_repeated_scans_and_measures_forward_close(self):
        rows = [
            row("02/01/2026", 10, "BUY", "EARLY_BREAKOUT", "2026-01-02T09:00:00"),
            row("02/01/2026", 10.2, "BUY", "EARLY_BREAKOUT", "2026-01-02T10:00:00"),
            row("05/01/2026", 10.5),
            row("06/01/2026", 10.4),
            row("07/01/2026", 11.0),
            row("08/01/2026", 10.8),
            row("09/01/2026", 11.2),
        ]
        result = evaluate_signal_history(rows)
        self.assertEqual(result["predictions_recorded"], 2)
        buy_5d = next(
            item for item in result["metrics"]
            if item["signal_type"] == "BUY" and item["horizon"] == 5
        )
        self.assertEqual(buy_5d["sample_size"], 1)
        self.assertGreater(buy_5d["average_return_pct"], 0)

    def test_early_signal_is_tracked_separately(self):
        rows = [
            row("02/01/2026", 10, "WAIT", "PREPARE"),
            row("05/01/2026", 10.5),
        ]
        result = evaluate_signal_history(rows)
        self.assertTrue(any(item["signal_type"] == "EARLY" for item in result["metrics"]))

    def test_missing_or_invalid_prices_are_ignored(self):
        result = evaluate_signal_history([{
            "symbol": "AAA", "strategy": "SWING",
            "data_date": "02/01/2026", "d1_close": 0,
            "signal": "BUY",
        }])
        self.assertEqual(result["predictions_recorded"], 0)


if __name__ == "__main__":
    unittest.main()
