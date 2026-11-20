import os
import sqlite3
import tempfile
import unittest

from backup_manager import backup_database, restore_database


class DatabaseBackupTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = self.tempdir.name
        self.db = os.path.join(self.root, "scanner.sqlite3")
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE transactions (symbol TEXT, quantity INTEGER)")
            conn.execute("INSERT INTO transactions VALUES ('FPT', 100)")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_backup_is_integral_and_contains_data(self):
        backup = backup_database(self.db, keep=3)
        self.assertTrue(os.path.isfile(backup))
        with sqlite3.connect(backup) as conn:
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(conn.execute("SELECT * FROM transactions").fetchone(), ("FPT", 100))

    def test_restore_requires_explicit_confirmation_and_restores_data(self):
        backup = backup_database(self.db)
        with self.assertRaises(PermissionError):
            restore_database(backup, self.db)

        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM transactions")

        restore_database(backup, self.db, confirm=True)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT * FROM transactions").fetchone(), ("FPT", 100))

    def test_restore_rejects_corrupt_backup(self):
        corrupt = os.path.join(self.root, "bad.sqlite3")
        with open(corrupt, "wb") as handle:
            handle.write(b"not a sqlite database")
        with self.assertRaises(sqlite3.DatabaseError):
            restore_database(corrupt, self.db, confirm=True)


if __name__ == "__main__":
    unittest.main()
