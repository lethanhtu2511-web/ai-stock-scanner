import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from strategy_v2_backtest import simulate

from strategy_v2 import (
    analyze_daily_setup,
    market_regime,
    position_size,
    universe_check,
)


def make_bars(n=320, start=12000.0, drift=0.0008, volume=900000):
    rng = np.random.default_rng(17)
    returns = drift + rng.normal(0, 0.006, n)
    close = start * np.cumprod(1 + returns)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.988
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.full(n, volume, dtype=float),
    })


class StrategyV2Tests(unittest.TestCase):
    def test_market_regime_bullish(self):
        market = make_bars(drift=0.001)
        self.assertIn(market_regime(market), {"BULLISH", "NEUTRAL"})

    def test_universe_rejects_short_history(self):
        ok, blockers = universe_check(make_bars(n=200))
        self.assertFalse(ok)
        self.assertTrue(any("250" in item for item in blockers))

    def test_universe_rejects_controlled_stock(self):
        ok, blockers = universe_check(make_bars(), controlled=True)
        self.assertFalse(ok)
        self.assertTrue(any("kiểm soát" in item for item in blockers))

    def test_position_size_obeys_risk_and_position_caps(self):
        result = position_size(
            capital=100_000_000, entry=20_000, stop=19_000,
            avg_turnover=20_000_000_000,
        )
        self.assertTrue(result["valid"])
        self.assertLessEqual(result["effective_risk_pct"], 0.75)
        self.assertLessEqual(result["position_value"], 10_000_000)

    def test_invalid_stop_never_sizes_position(self):
        result = position_size(capital=10_000_000, entry=20_000, stop=21_000)
        self.assertFalse(result["valid"])
        self.assertEqual(result["quantity"], 0)

    def test_bearish_market_blocks_new_candidates(self):
        stock = make_bars(drift=0.0007)
        market = make_bars(start=20000, drift=-0.0015)
        result = analyze_daily_setup("TST", stock, market)
        self.assertEqual(result["market_regime"], "BEARISH")
        self.assertEqual(result["candidates"], [])

    def test_suspended_stock_is_excluded(self):
        stock = make_bars()
        market = make_bars(start=20000, drift=0.001)
        result = analyze_daily_setup("TST", stock, market, suspended=True)
        self.assertFalse(result["universe_eligible"])
        self.assertEqual(result["candidates"], [])

    def test_trailing_stop_does_not_activate_before_first_target(self):
        d = make_bars(n=333)
        d.loc[325:329, "low"] = 95.0
        d.loc[329, ["open", "close", "high", "low"]] = [100.0, 100.0, 101.0, 99.0]
        d.loc[330, ["open", "close", "high", "low"]] = [101.0, 101.0, 102.0, 100.0]
        # The next session dips below the would-be Chandelier trail (99),
        # but remains above the original protective stop (94.6).
        d.loc[331, ["open", "close", "high", "low"]] = [100.0, 100.0, 102.0, 98.5]
        d.loc[332, ["open", "close", "high", "low"]] = [100.0, 100.0, 101.0, 99.0]
        candidate = {
            "setup": "PULLBACK", "entry_type": "NEXT_OPEN", "gap_limit_atr": 1.0,
            "atr": 2.0, "holding_sessions": [3, 3], "score": 5,
            "risk_plan": {"slippage_rate": 0.001, "stop_loss": 90.0},
        }
        # Force a deterministic previous-close Chandelier level of 99.
        indicators = pd.DataFrame({
            "high22_prev": np.full(len(d), 105.0),
            "atr": np.full(len(d), 2.0),
            "ema21": np.full(len(d), 95.0),
        })
        with patch("strategy_v2_backtest._indicators", return_value=indicators):
            result = simulate(d, 329, candidate)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["initial_stop"], 94.6, places=1)
        self.assertEqual(result["exit_reason"], "TIME_EXIT")

    def test_reversal_stop_entry_is_not_cancelled_by_same_day_close(self):
        d = make_bars(n=331)
        d.loc[329, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
        # The next session crosses the stop-entry trigger intraday, then closes
        # below the confirmation low. The fill must not be erased retrospectively.
        d.loc[330, ["open", "high", "low", "close"]] = [100.0, 102.0, 98.0, 98.5]
        candidate = {
            "setup": "REVERSAL", "entry_type": "STOP_ABOVE_CONFIRMATION_HIGH",
            "entry_trigger": 101.0, "confirmation_low": 99.0,
            "cancel_if_close_below_confirmation_low": 99.0,
            "atr": 1.0, "holding_sessions": [2, 2], "score": 4,
            "risk_plan": {"slippage_rate": 0.001, "stop_loss": 98.5},
        }
        result = simulate(d, 329, candidate)
        self.assertIsNotNone(result)
        self.assertEqual(result["actual_entry"], 101.0)
        self.assertEqual(result["exit_reason"], "STOP")

    def test_backtest_rebuilds_pullback_stop_from_actual_fill(self):
        d = make_bars(n=331)
        d.loc[325:329, "low"] = 95.0
        d.loc[329, ["open", "close", "high", "low"]] = [100.0, 100.0, 101.0, 99.0]
        d.loc[330, ["open", "close", "high", "low"]] = [101.0, 101.0, 102.0, 100.0]
        candidate = {
            "setup": "PULLBACK", "entry_type": "NEXT_OPEN", "gap_limit_atr": 1.0,
            "atr": 2.0, "holding_sessions": [3, 3], "score": 5,
            "risk_plan": {"slippage_rate": 0.001, "stop_loss": 90.0},
        }
        result = simulate(d, 329, candidate)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["initial_stop"], 94.6, places=1)
        self.assertGreater(result["actual_entry"], result["initial_stop"])


if __name__ == "__main__":
    unittest.main()
