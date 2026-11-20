import unittest

import pandas as pd

from scanner import _classify_market_regime


class MarketRegimeTests(unittest.TestCase):
    def test_missing_market_data_is_unknown(self):
        self.assertEqual(_classify_market_regime(None), "UNKNOWN")

    def test_bullish_market_regime(self):
        closes = [100 + i for i in range(60)]
        frame = pd.DataFrame({"Close": closes})
        self.assertEqual(_classify_market_regime(frame), "BULLISH")

    def test_bearish_market_regime(self):
        closes = [200 - i for i in range(60)]
        frame = pd.DataFrame({"Close": closes})
        self.assertEqual(_classify_market_regime(frame), "BEARISH")

    def test_insufficient_market_history_is_unknown(self):
        frame = pd.DataFrame({"Close": list(range(1, 30))})
        self.assertEqual(_classify_market_regime(frame), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
