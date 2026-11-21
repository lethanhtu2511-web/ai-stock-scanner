import os
import tempfile
import unittest

from portfolio import (
    add_transaction,
    get_positions,
    get_portfolio_summary,
    get_cash_summary,
    get_settings,
    save_settings,
)


class PortfolioLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "portfolio.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_buy_sell_fees_tax_average_cost_and_realized_pnl(self):
        buy = add_transaction("MUA", "FPT", 100, 20_000, db_path=self.db_path)
        self.assertAlmostEqual(buy["fee"], 3_000)
        self.assertAlmostEqual(buy["tax"], 0)

        sell = add_transaction("BÁN", "FPT", 50, 22_000, db_path=self.db_path)
        self.assertAlmostEqual(sell["fee"], 1_650)
        self.assertAlmostEqual(sell["tax"], 1_100)

        positions = get_positions(self.db_path)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["quantity"], 50)
        self.assertAlmostEqual(positions[0]["avg_cost"], 20_030)

        summary = get_portfolio_summary(self.db_path)
        self.assertAlmostEqual(summary["realized_pnl"], 95_750)
        self.assertAlmostEqual(summary["total_cost"], 1_001_500)

        cash = get_cash_summary(self.db_path)
        self.assertAlmostEqual(cash["net_buys"], 2_003_000)
        self.assertAlmostEqual(cash["net_sells"], 1_097_250)

    def test_new_install_has_unconfigured_capital(self):
        self.assertEqual(get_settings(self.db_path)["capital"], 0)

    def test_existing_user_capital_of_10m_is_preserved(self):
        save_settings({"capital": 10_000_000}, self.db_path)
        self.assertEqual(get_settings(self.db_path)["capital"], 10_000_000)

    def test_cannot_sell_more_than_held(self):
        add_transaction("MUA", "FPT", 100, 20_000, db_path=self.db_path)
        with self.assertRaisesRegex(ValueError, "chỉ đang có 100"):
            add_transaction("BÁN", "FPT", 200, 21_000, db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
