
import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def connect(db_path="data/stock_scanner.sqlite3"):
    folder = os.path.dirname(db_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_scan_db(db_path="data/stock_scanner.sqlite3"):
    with connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal TEXT NOT NULL,
                score REAL,
                price REAL,
                stop_loss REAL,
                target_price REAL,
                reason TEXT DEFAULT ''
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scan_symbol_time
            ON scan_results(symbol, scanned_at)
        """)


def save_scan_results(results, db_path="data/stock_scanner.sqlite3"):
    if not results:
        return 0

    init_scan_db(db_path)

    scanned_at = datetime.now(VN_TZ).isoformat(
        timespec="seconds"
    )

    rows = []

    for item in results:
        rows.append((
            scanned_at,
            str(item.get("symbol", "")).upper(),
            str(item.get("strategy", "DEFAULT")),
            str(item.get("signal", "WAIT")),
            float(item.get("score", 0) or 0),
            float(item.get("price", 0) or 0),
            float(item.get("stop_loss", 0) or 0),
            float(item.get("target_price", 0) or 0),
            str(item.get("reason", ""))[:1000],
        ))

    with connect(db_path) as conn:
        conn.executemany("""
            INSERT INTO scan_results (
                scanned_at, symbol, strategy, signal,
                score, price, stop_loss, target_price, reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    return len(rows)


def get_latest_scan(db_path="data/stock_scanner.sqlite3"):
    init_scan_db(db_path)

    with connect(db_path) as conn:
        rows = conn.execute("""
            SELECT s.*
            FROM scan_results s
            INNER JOIN (
                SELECT symbol, strategy, MAX(id) AS latest_id
                FROM scan_results
                GROUP BY symbol, strategy
            ) latest ON s.id = latest.latest_id
            ORDER BY s.score DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_scan_history(
    symbol=None,
    limit=100,
    db_path="data/stock_scanner.sqlite3"
):
    init_scan_db(db_path)

    limit = max(1, min(int(limit), 2000))

    with connect(db_path) as conn:
        if symbol:
            rows = conn.execute("""
                SELECT *
                FROM scan_results
                WHERE symbol = ?
                ORDER BY id DESC
                LIMIT ?
            """, (str(symbol).upper(), limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT *
                FROM scan_results
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

    return [dict(row) for row in rows]


def clear_old_scan_results(
    days=30,
    db_path="data/stock_scanner.sqlite3"
):
    init_scan_db(db_path)

    days = max(1, int(days))

    with connect(db_path) as conn:
        cursor = conn.execute("""
            DELETE FROM scan_results
            WHERE scanned_at < datetime('now', ?)
        """, (f"-{days} days",))

        return cursor.rowcount
