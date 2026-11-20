"""Cross-process SQLite lease lock for market scans."""
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

LOCK_NAME = "market_scan"
LOCK_LEASE_SECONDS = 180
LOCK_HEARTBEAT_SECONDS = 30


def _connect(db_path):
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def scan_lock(db_path, lease_seconds=LOCK_LEASE_SECONDS):
    """
    Atomic SQLite lock shared by Streamlit sessions and worker processes.
    A heartbeat renews the lease while a scan is genuinely running; if the
    process dies, the lock becomes reclaimable after the short lease expires.
    """
    token = uuid.uuid4().hex
    lease = max(90, int(lease_seconds))
    now = time.time()
    acquired = False

    conn = _connect(db_path)
    try:
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
        # Reclaim legacy two-hour locks created before heartbeat support.
        legacy_lease = row is not None and (float(row[1]) - now) > lease + 30
        if row is None or float(row[1]) <= now or legacy_lease:
            conn.execute("""
                INSERT INTO runtime_locks(lock_name, owner_token, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(lock_name) DO UPDATE SET
                    owner_token = excluded.owner_token,
                    expires_at = excluded.expires_at
            """, (LOCK_NAME, token, now + lease))
            conn.commit()
            acquired = True
        else:
            conn.rollback()
    finally:
        conn.close()

    stop_heartbeat = threading.Event()

    def renew():
        while not stop_heartbeat.wait(LOCK_HEARTBEAT_SECONDS):
            renewal = None
            try:
                renewal = _connect(db_path)
                renewal.execute(
                    "UPDATE runtime_locks SET expires_at = ? "
                    "WHERE lock_name = ? AND owner_token = ?",
                    (time.time() + lease, LOCK_NAME, token),
                )
                renewal.commit()
            except Exception:
                # A transient DB issue should not crash the scan. The lease
                # remains valid until its previous expiry.
                pass
            finally:
                if renewal is not None:
                    renewal.close()

    heartbeat_thread = None
    if acquired:
        heartbeat_thread = threading.Thread(
            target=renew, name="scan-lock-heartbeat", daemon=True
        )
        heartbeat_thread.start()

    try:
        yield acquired
    finally:
        if acquired:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2)
            release = _connect(db_path)
            try:
                release.execute(
                    "DELETE FROM runtime_locks WHERE lock_name = ? AND owner_token = ?",
                    (LOCK_NAME, token),
                )
                release.commit()
            finally:
                release.close()
