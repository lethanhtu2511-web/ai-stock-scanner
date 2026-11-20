import tempfile
import unittest
from pathlib import Path

from scan_store import get_latest_scan, save_scan_results


class AdaptiveScanPersistenceTests(unittest.TestCase):
    def test_adaptive_payload_survives_sqlite_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            db_path = str(Path(folder) / "scanner.sqlite3")
            payload = {
                "profile": "ADAPTIVE_RESEARCH_V1",
                "score": 76,
                "status": "SETUP_WATCH",
                "executable_buy": False,
                "hard_risks": [],
                "soft_confirmations_missing": ["MACD chưa xác nhận"],
            }
            count = save_scan_results([{
                "symbol": "FPT",
                "strategy": "SWING",
                "signal": "WAIT",
                "score": 64,
                "adaptive_research": payload,
            }], db_path=db_path)
            self.assertEqual(count, 1)
            rows = get_latest_scan(db_path=db_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["adaptive_research"]["score"], 76)
            self.assertFalse(rows[0]["adaptive_research"]["executable_buy"])


if __name__ == "__main__":
    unittest.main()
