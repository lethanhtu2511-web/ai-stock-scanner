import os
import sqlite3
import time
import uuid
from contextlib import contextmanager


LOCK_NAME = "market_scan"
LOCK_LEASE_SECONDS = 2 * 60 * 60


@contextmanager
def scan_lock(db_path, lease_seconds=LOCK_LEASE_SECONDS):
    """
    Khóa quét liên tiến trình bằng SQLite.

    BEGIN IMMEDIATE đảm bảo hai tiến trình không thể cùng giành khóa.
    Lease 2 giờ tự giải phóng khóa mồ côi nếu tiến trình bị dừng đột ngột.
    """
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    token = uuid.uuid4().hex
    now = time.time()
    acquired = False

    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runtime_locks (
                lock_name TEXT PRIMARY KEY,
                owner_token TEXT NOT NULL,
                expires_at REAL NOT NULL
            )
        """)
        conn.commit()

        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT owner_token, expires_at FROM runtime_locks WHERE lock_name = ?",
            (LOCK_NAME,),
        ).fetchone()

        if row is None or float(row[1]) <= now:
            conn.execute(
                """
                INSERT INTO runtime_locks(lock_name, owner_token, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner_token = excluded.owner_token,
                    expires_at = excluded.expires_at
                """,
                (LOCK_NAME, token, now + max(60, int(lease_seconds))),
            )
            conn.commit()
            acquired = True
        else:
            conn.rollback()

    finally:
        conn.close()

    try:
        yield acquired
    finally:
        if acquired:
            release = sqlite3.connect(db_path, timeout=30)
            try:
                release.execute("PRAGMA busy_timeout = 30000")
                release.execute(
                    "DELETE FROM runtime_locks WHERE lock_name = ? AND owner_token = ?",
                    (LOCK_NAME, token),
                )
                release.commit()
            finally:
                release.close()
