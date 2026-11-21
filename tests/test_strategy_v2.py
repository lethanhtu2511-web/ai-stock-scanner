import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from strategy_v2_backtest import simulate, point_in_time_features, swing_filter_diagnostics
from top5_walkforward_backtest import candidate_diagnostics, select_candidates

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
    def test_select_candidates_can_run_swing_only_without_changing_baseline(self):
        ranked = [
            {"symbol": "AAA", "rank_score": 99, "candidate": {"setup": "PULLBACK"}},
            {"symbol": "BBB", "rank_score": 98, "candidate": {"setup": "SWING_BREAKOUT"}},
            {"symbol": "CCC", "rank_score": 97, "candidate": {"setup": "SWING_BREAKOUT"}},
            {"symbol": "DDD", "rank_score": 96, "candidate": {"setup": "REVERSAL"}},
            {"symbol": "EEE", "rank_score": 95, "candidate": {"setup": "SWING_BREAKOUT"}},
            {"symbol": "FFF", "rank_score": 94, "candidate": {"setup": "SWING_BREAKOUT"}},
        ]
        baseline = select_candidates(ranked)
        swing_only = select_candidates(ranked, setup="SWING_BREAKOUT")
        self.assertEqual([x["symbol"] for x in baseline], ["AAA", "BBB", "CCC"])
        self.assertEqual([x["symbol"] for x in swing_only], ["BBB", "CCC", "EEE"])
        self.assertTrue(all(x["candidate"]["setup"] == "SWING_BREAKOUT" for x in swing_only))
        self.assertLessEqual(len(swing_only), 3)

    def test_point_in_time_swing_features_use_only_past_rows(self):
        from strategy_v2 import _indicators
        stock = _indicators(make_bars(n=330))
        market = _indicators(make_bars(n=330, start=20000, drift=0.0002))
        trade = {"actual_entry": float(stock.iloc[310].close)*1.01,
                 "initial_stop": float(stock.iloc[310].close)*0.95}
        first = point_in_time_features(stock.iloc[:311], 310, market.iloc[:311], 310, trade)
        changed = stock.copy()
        changed.loc[311:, ["open", "high", "low", "close", "volume"]] *= 4
        second = point_in_time_features(changed.iloc[:311], 310, market.iloc[:311], 310, trade)
        self.assertEqual(first, second)
        self.assertIn("relative_strength_20s_pct", first)
        self.assertIn("signal_stop_distance_pct", first)

    def test_swing_filter_diagnostics_are_research_only_and_count_trades(self):
        frame = pd.DataFrame([
            {"setup":"SWING_BREAKOUT","net_return_pct":3.0,"oos_window":1,
             "relative_strength_20s_pct":2.0,"relative_strength_60s_pct":1.0,
             "signal_vol_ratio":1.8,"signal_adx":25.0,
             "signal_entry_gap_pct":0.2,"signal_stop_distance_pct":5.0},
            {"setup":"SWING_BREAKOUT","net_return_pct":-2.0,"oos_window":2,
             "relative_strength_20s_pct":-1.0,"relative_strength_60s_pct":-2.0,
             "signal_vol_ratio":1.2,"signal_adx":15.0,
             "signal_entry_gap_pct":0.8,"signal_stop_distance_pct":9.0},
            {"setup":"PULLBACK","net_return_pct":5.0,"oos_window":1,
             "relative_strength_20s_pct":5.0,"relative_strength_60s_pct":5.0,
             "signal_vol_ratio":2.0,"signal_adx":30.0,
             "signal_entry_gap_pct":0.1,"signal_stop_distance_pct":4.0},
        ])
        result = swing_filter_diagnostics(frame)
        self.assertEqual(result["BASELINE"]["trades"], 2)
        self.assertEqual(result["RS20_POSITIVE"]["trades"], 1)
        self.assertEqual(result["RS20_POSITIVE"]["average_net_return_pct"], 3.0)
        self.assertEqual(result["RS20_POSITIVE"]["by_oos_window"]["1"]["trades"], 1)

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

    def test_candidate_diagnostics_use_only_supplied_point_in_time_rows(self):
        from strategy_v2 import _indicators

        stock = _indicators(make_bars(n=320))
        market = _indicators(make_bars(n=320, start=20000, drift=0.0004))
        past_stock = stock.iloc[:310].copy()
        past_market = market.iloc[:310].copy()
        candidate = {"setup": "SWING_BREAKOUT"}
        trade = {"actual_entry": 15000.0, "initial_stop": 14500.0}
        result = candidate_diagnostics(past_stock, past_market, candidate, trade)

        self.assertEqual(result["setup"], "SWING_BREAKOUT")
        self.assertIsNotNone(result["relative_strength_20s_pct"])
        self.assertAlmostEqual(result["stop_distance_pct"], (500 / 15000) * 100, places=3)
        # Appending future bars must not change a snapshot made from the past slice.
        changed_future = stock.copy()
        changed_future.loc[310:, ["open", "high", "low", "close", "volume"]] *= 3
        repeated = candidate_diagnostics(
            changed_future.iloc[:310], past_market, candidate, trade
        )
        self.assertEqual(result, repeated)

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
