
import os
import sqlite3
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

DEFAULT_DB_PATH = "data/stock_scanner.sqlite3"

SIGNAL_PRIORITY = {
    "BUY": 5,
    "WAIT": 4,
    "WATCH": 3,
    "SKIP": 2,
    "FILTERED": 1,
}

STRATEGY_PRIORITY = {
    "SWING": 1,
    "REVERSAL": 2,
    "PULLBACK": 3,
}


def connect(db_path=DEFAULT_DB_PATH):
    folder = os.path.dirname(db_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        timeout=30,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    return conn


def init_scan_db(db_path=DEFAULT_DB_PATH):
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

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scan_strategy_symbol
            ON scan_results(strategy, symbol, id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scan_time
            ON scan_results(scanned_at)
        """)


def _safe_number(value):
    if value is None:
        return None

    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def save_scan_results(
    results,
    db_path=DEFAULT_DB_PATH
):
    if not results:
        return 0

    init_scan_db(db_path)

    scanned_at = datetime.now(VN_TZ).isoformat(
        timespec="seconds"
    )

    rows = []

    for item in results:
        symbol = str(
            item.get("symbol", "")
        ).strip().upper()

        strategy = str(
            item.get("strategy", "CORE")
        ).strip().upper()

        signal = str(
            item.get("signal", "WAIT")
        ).strip().upper()

        if not symbol:
            continue

        rows.append((
            scanned_at,
            symbol,
            strategy,
            signal,
            _safe_number(item.get("score")),
            _safe_number(item.get("price")),
            _safe_number(item.get("stop_loss")),
            _safe_number(item.get("target_price")),
            str(item.get("reason", ""))[:1000],
        ))

    if not rows:
        return 0

    with connect(db_path) as conn:
        conn.executemany("""
            INSERT INTO scan_results (
                scanned_at,
                symbol,
                strategy,
                signal,
                score,
                price,
                stop_loss,
                target_price,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    return len(rows)


def get_latest_scan(db_path=DEFAULT_DB_PATH):
    """
    Lấy kết quả mới nhất của từng cặp mã + chiến lược.
    Mỗi mã có thể xuất hiện tối đa 3 lần:
    SWING, REVERSAL, PULLBACK.
    """

    init_scan_db(db_path)

    with connect(db_path) as conn:
        rows = conn.execute("""
            SELECT s.*
            FROM scan_results AS s
            INNER JOIN (
                SELECT
                    symbol,
                    strategy,
                    MAX(id) AS latest_id
                FROM scan_results
                GROUP BY symbol, strategy
            ) AS latest
            ON s.id = latest.latest_id
            ORDER BY
                CASE s.signal
                    WHEN 'BUY' THEN 5
                    WHEN 'WAIT' THEN 4
                    WHEN 'WATCH' THEN 3
                    WHEN 'SKIP' THEN 2
                    WHEN 'FILTERED' THEN 1
                    ELSE 0
                END DESC,
                COALESCE(s.score, 0) DESC,
                CASE s.strategy
                    WHEN 'SWING' THEN 1
                    WHEN 'REVERSAL' THEN 2
                    WHEN 'PULLBACK' THEN 3
                    ELSE 4
                END ASC,
                s.symbol ASC
        """).fetchall()

    return [dict(row) for row in rows]


def get_scan_history(
    symbol=None,
    limit=100,
    db_path=DEFAULT_DB_PATH
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
            """, (
                str(symbol).strip().upper(),
                limit
            )).fetchall()

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
    db_path=DEFAULT_DB_PATH
):
    init_scan_db(db_path)

    days = max(1, int(days))

    cutoff = (
        datetime.now(VN_TZ) - timedelta(days=days)
    ).isoformat(timespec="seconds")

    with connect(db_path) as conn:
        cursor = conn.execute("""
            DELETE FROM scan_results
            WHERE scanned_at < ?
        """, (cutoff,))

        return cursor.rowcount


def get_scan_summary(db_path=DEFAULT_DB_PATH):
    init_scan_db(db_path)

    with connect(db_path) as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT symbol) AS total_symbols,
                MAX(scanned_at) AS latest_scan_at
            FROM scan_results
        """).fetchone()

    return {
        "total_rows": row["total_rows"],
        "total_symbols": row["total_symbols"],
        "latest_scan_at": row["latest_scan_at"],
    }
