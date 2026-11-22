import unittest
from independent_oos_swing_backtest import rank_candidates


class IndependentSwingSelectionTests(unittest.TestCase):
    def candidate(self, symbol, score, rs20, rs60, regime="BULLISH", turnover=20_000_000_000):
        return {"symbol": symbol, "normalized_score": score, "avg_turnover20": turnover,
                "market_regime": regime,
                "features": {"rs20_pct": rs20, "rs60_pct": rs60}}

    def test_rs_arm_requires_positive_both_horizons(self):
        pool = [self.candidate("A", .9, 2, -1), self.candidate("B", .8, 1, 3)]
        self.assertEqual([x["symbol"] for x in rank_candidates(pool, "RS20_RS60")], ["B"])

    def test_strict_arm_requires_bullish_regime_and_liquidity(self):
        pool = [self.candidate("A", .9, 2, 3, regime="NEUTRAL"),
                self.candidate("B", .8, 2, 3, turnover=15_000_000_000),
                self.candidate("C", .7, 2, 3)]
        self.assertEqual([x["symbol"] for x in rank_candidates(pool, "RS20_RS60_BULL_LIQUID")], ["C"])

    def test_no_candidate_is_forced(self):
        pool = [self.candidate("A", .9, -2, -3)]
        self.assertEqual(rank_candidates(pool, "RS20_RS60"), [])


if __name__ == "__main__":
    unittest.main()
