import os
import sqlite3
import tempfile
import unittest

from scan_lock import scan_lock


class ScanLockTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "scanner.sqlite3")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_lock_is_exclusive_between_process_contexts(self):
        with scan_lock(self.db_path) as first:
            self.assertTrue(first)
            with scan_lock(self.db_path) as second:
                self.assertFalse(second)

        with scan_lock(self.db_path) as third:
            self.assertTrue(third)

    def test_expired_lock_can_be_reclaimed(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE runtime_locks (
                    lock_name TEXT PRIMARY KEY,
                    owner_token TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            conn.execute(
                "INSERT INTO runtime_locks VALUES (?, ?, ?)",
                ("market_scan", "dead-process", 1),
            )

        with scan_lock(self.db_path) as acquired:
            self.assertTrue(acquired)


if __name__ == "__main__":
    unittest.main()
