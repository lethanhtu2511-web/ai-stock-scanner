import unittest

from adaptive_strategy import adaptive_assessment, classify_market_regime


class AdaptiveStrategyTests(unittest.TestCase):
    def test_adaptive_profile_never_emits_executable_buy(self):
        result = adaptive_assessment(
            {"score": 88, "reasons": ["trend", "volume", "RSI"], "blockers": []},
            liquidity_ok=True,
            risk_plan={"reward_risk": 2.5},
            atr_pct=3.0,
            market_regime="BULLISH",
            strategy="SWING",
        )
        self.assertFalse(result["executable_buy"])
        self.assertIn(result["status"], {"HIGH_CONVICTION_SETUP", "SETUP_WATCH"})

    def test_hard_risk_blocks_adaptive_setup(self):
        result = adaptive_assessment(
            {"score": 90, "reasons": ["trend", "volume", "RSI"], "blockers": []},
            liquidity_ok=True,
            risk_plan={"reward_risk": 1.5},
            atr_pct=3.0,
        )
        self.assertEqual(result["status"], "RISK_BLOCKED")
        self.assertTrue(result["hard_risks"])

    def test_soft_confirmation_is_penalized_not_hard_blocked(self):
        result = adaptive_assessment(
            {"score": 80, "reasons": ["trend", "volume", "RSI"],
             "blockers": ["MACD chưa xác nhận xu hướng tăng"]},
            liquidity_ok=True,
            risk_plan={"reward_risk": 2.2},
            atr_pct=2.5,
        )
        self.assertEqual(result["status"], "HIGH_CONVICTION_SETUP")
        self.assertEqual(result["score"], 74)
        self.assertFalse(result["hard_risks"])

    def test_market_regime_requires_enough_history(self):
        import pandas as pd
        self.assertEqual(
            classify_market_regime(pd.DataFrame({"close": [1, 2, 3]})),
            "UNKNOWN",
        )


if __name__ == "__main__":
    unittest.main()
