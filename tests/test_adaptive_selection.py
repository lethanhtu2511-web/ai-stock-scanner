import unittest

from adaptive_selection_backtest import make_checkpoints, rank_candidates


class AdaptiveSelectionTests(unittest.TestCase):
    def test_checkpoints_are_recent_and_spaced(self):
        points = make_checkpoints(1000, windows=5, step=2, forward=5, minimum_history=250)
        self.assertEqual(len(points), 5)
        self.assertEqual([b - a for a, b in zip(points, points[1:])], [2, 2, 2, 2])
        self.assertEqual(points[-1], 994)

    def test_checkpoints_reject_insufficient_history(self):
        with self.assertRaises(ValueError):
            make_checkpoints(260, windows=10, step=2, forward=5, minimum_history=250)

    def candidate(self, symbol, score, rs60, adx=25, rsi=60, gap=0.5, turnover=1e10):
        return {
            "symbol": symbol,
            "normalized_score": score,
            "risk_factor": 1.0,
            "avg_turnover20": turnover,
            "features": {
                "rs60_pct": rs60,
                "adx": adx,
                "rsi": rsi,
                "close_to_ema21_atr": gap,
            },
        }

    def test_rs60_arm_excludes_negative_relative_strength(self):
        pool = [
            self.candidate("AAA", 0.9, -1.0),
            self.candidate("BBB", 0.8, 2.0),
        ]
        selected = rank_candidates(pool, "RS60_POSITIVE")
        self.assertEqual([x["symbol"] for x in selected], ["BBB"])

    def test_quality_gate_applies_all_predeclared_conditions(self):
        pool = [
            self.candidate("AAA", 0.9, 2.0, adx=25, rsi=60, gap=0.5),
            self.candidate("BBB", 0.9, -1.0, adx=25, rsi=60, gap=0.5),
            self.candidate("CCC", 0.9, 2.0, adx=45, rsi=60, gap=0.5),
            self.candidate("DDD", 0.7, 2.0, adx=25, rsi=60, gap=0.5),
        ]
        selected = rank_candidates(pool, "QUALITY_GATE")
        self.assertEqual([x["symbol"] for x in selected], ["AAA"])

    def test_cross_sectional_blend_is_deterministic(self):
        pool = [
            self.candidate("AAA", 0.9, 1.0),
            self.candidate("BBB", 0.8, 5.0),
            self.candidate("CCC", 0.7, -1.0),
        ]
        first = [x["symbol"] for x in rank_candidates(pool, "CROSS_SECTIONAL_BLEND")]
        second = [x["symbol"] for x in rank_candidates(pool, "CROSS_SECTIONAL_BLEND")]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


if __name__ == "__main__":
    unittest.main()
