import unittest

import pandas as pd

from strategy_engine import _relative_strength
from scanner import _classify_market_regime


class MarketContextTests(unittest.TestCase):
    def test_relative_strength_uses_common_trading_dates(self):
        dates = pd.bdate_range("2026-01-01", periods=30)
        stock = pd.DataFrame({
            "date": dates,
            "close": [100 + i for i in range(30)],
        })
        # Deliberately omit one market session; calculation must align dates.
        market_dates = dates.delete(5)
        market = pd.DataFrame({
            "Date": market_dates,
            "Close": [100 + i * 0.5 for i in range(29)],
        })
        value = _relative_strength(stock, market)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)

    def test_relative_strength_missing_market_is_unavailable(self):
        self.assertIsNone(_relative_strength(
            pd.DataFrame({"close": range(1, 30)}), None
        ))

    def test_relative_strength_requires_21_common_sessions(self):
        dates = pd.bdate_range("2026-01-01", periods=20)
        stock = pd.DataFrame({"date": dates, "close": range(100, 120)})
        market = pd.DataFrame({"date": dates, "close": range(100, 120)})
        self.assertIsNone(_relative_strength(stock, market))

    def test_market_regime_accepts_provider_column_case(self):
        closes = [100 + i for i in range(60)]
        frame = pd.DataFrame({"close": closes})
        self.assertEqual(_classify_market_regime(frame), "BULLISH")


if __name__ == "__main__":
    unittest.main()
