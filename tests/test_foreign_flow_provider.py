import unittest
import pandas as pd

from foreign_flow_provider import normalize_foreign_flow, summarize_foreign_flow


class ForeignFlowTests(unittest.TestCase):
    def test_normalize_foreign_flow_and_net_value(self):
        raw = pd.DataFrame([
            {"time": "2026-11-17", "buy_val": 120, "sell_val": 80, "buy_vol": 12, "sell_vol": 8},
            {"time": "2026-11-18", "buy_val": 50, "sell_val": 90, "buy_vol": 5, "sell_vol": 9},
        ])
        result = normalize_foreign_flow(raw)
        self.assertEqual(len(result), 2)
        self.assertEqual(float(result.iloc[0]["net_val"]), 40)
        self.assertEqual(float(result.iloc[1]["net_vol"]), -4)

    def test_asof_filter_prevents_future_foreign_flow_leakage(self):
        raw = pd.DataFrame([
            {"time": "2026-11-17", "buy_val": 100, "sell_val": 50, "buy_vol": 10, "sell_vol": 5},
            {"time": "2026-11-18", "buy_val": 20, "sell_val": 90, "buy_vol": 2, "sell_vol": 9},
            {"time": "2026-11-19", "buy_val": 10, "sell_val": 200, "buy_vol": 1, "sell_vol": 20},
        ])
        summary = summarize_foreign_flow(raw, as_of="2026-11-18", market_turnover_5d=1000)
        self.assertEqual(summary["foreign_data_date"], "2026-11-18")
        self.assertEqual(summary["foreign_net_value"], -70)
        self.assertEqual(summary["foreign_net_5d"],  -30)
        self.assertEqual(summary["foreign_net_ratio"], -30 / 1000)

    def test_missing_flow_is_unavailable_not_zero_net(self):
        summary = summarize_foreign_flow(pd.DataFrame())
        self.assertEqual(summary["foreign_flow_status"], "UNAVAILABLE")
        self.assertIsNone(summary["foreign_net_value"])
        self.assertEqual(summary["foreign_score_adjustment"], 0)


if __name__ == "__main__":
    unittest.main()
