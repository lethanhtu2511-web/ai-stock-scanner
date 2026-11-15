
import os
import sqlite3
import math
import json

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

EXTRA_COLUMNS = {
    "data_date": "TEXT",
    "reward_risk": "REAL",
    "stop_distance_pct": "REAL",
    "target_distance_pct": "REAL",
    "atr_pct": "REAL",
    "relative_strength": "REAL",
    "avg_turnover_20d": "REAL",
    "median_turnover_20d": "REAL",
    "market_context": "TEXT",
    "blockers": "TEXT",
    "reasons": "TEXT",
    "penalties": "TEXT",

    # Giá và trạng thái dữ liệu.
    "d1_close": "REAL",
    "realtime_price": "REAL",
    "kbs_quote_price": "REAL",
    "realtime_volume": "REAL",
    "quote_source_time": "TEXT",
    "quote_fetched_at": "TEXT",
    "quote_age_seconds": "REAL",
    "quote_source": "TEXT",
    "price_data_status": "TEXT",
    "price_data_message": "TEXT",
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


def _safe_text(value, limit=2000):
    if value is None:
        return ""

    return str(value)[:limit]


def _safe_json(value):
    if value is None:
        value = []

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        return "[]"


def _decode_json(value):
    if not value:
        return []

    try:
        result = json.loads(value)

        return result if isinstance(result, list) else []

    except (TypeError, ValueError):
        return []


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

        # Tự bổ sung cột còn thiếu.
        # Không xóa hoặc ghi đè dữ liệu cũ.
        existing_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(scan_results)"
            ).fetchall()
        }

        for column, column_type in EXTRA_COLUMNS.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE scan_results "
                    f"ADD COLUMN {column} {column_type}"
                )

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

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scan_signal_score
            ON scan_results(signal, score)
        """)


def save_scan_results(
    results,
    db_path=DEFAULT_DB_PATH,
):
    if not results:
        return 0

    init_scan_db(db_path)

    scanned_at = datetime.now(
        VN_TZ
    ).isoformat(timespec="seconds")

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
            _safe_text(item.get("reason"), 2000),

            _safe_text(item.get("data_date"), 30),
            _safe_number(item.get("reward_risk")),
            _safe_number(item.get("stop_distance_pct")),
            _safe_number(item.get("target_distance_pct")),
            _safe_number(item.get("atr_pct")),
            _safe_number(item.get("relative_strength")),
            _safe_number(item.get("avg_turnover_20d")),
            _safe_number(item.get("median_turnover_20d")),
            _safe_text(item.get("market_context"), 200),

            _safe_json(item.get("blockers")),
            _safe_json(item.get("reasons")),
            _safe_json(item.get("penalties")),

            _safe_number(item.get("d1_close")),
            _safe_number(item.get("realtime_price")),
            _safe_number(item.get("kbs_quote_price")),
            _safe_number(item.get("realtime_volume")),
            _safe_text(item.get("quote_source_time"), 80),
            _safe_text(item.get("quote_fetched_at"), 50),
            _safe_number(item.get("quote_age_seconds")),
            _safe_text(item.get("quote_source"), 100),
            _safe_text(item.get("price_data_status"), 30),
            _safe_text(item.get("price_data_message"), 500),
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
                reason,
                data_date,
                reward_risk,
                stop_distance_pct,
                target_distance_pct,
                atr_pct,
                relative_strength,
                avg_turnover_20d,
                median_turnover_20d,
                market_context,
                blockers,
                reasons,
                penalties,
                d1_close,
                realtime_price,
                kbs_quote_price,
                realtime_volume,
                quote_source_time,
                quote_fetched_at,
                quote_age_seconds,
                quote_source,
                price_data_status,
                price_data_message
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, rows)

    return len(rows)


def _format_row(row):
    item = dict(row)

    item["blockers"] = _decode_json(
        item.get("blockers")
    )

    item["reasons"] = _decode_json(
        item.get("reasons")
    )

    item["penalties"] = _decode_json(
        item.get("penalties")
    )

    return item


def get_latest_scan(db_path=DEFAULT_DB_PATH):
    """Lấy kết quả mới nhất của từng cặp mã + chiến lược."""

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

    return [
        _format_row(row)
        for row in rows
    ]


def get_scan_history(
    symbol=None,
    limit=100,
    db_path=DEFAULT_DB_PATH,
):
    init_scan_db(db_path)

    limit = max(
        1,
        min(int(limit), 2000),
    )

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
                limit,
            )).fetchall()

        else:
            rows = conn.execute("""
                SELECT *
                FROM scan_results
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

    return [
        _format_row(row)
        for row in rows
    ]


def clear_old_scan_results(
    days=30,
    db_path=DEFAULT_DB_PATH,
):
    init_scan_db(db_path)

    days = max(1, int(days))

    cutoff = (
        datetime.now(VN_TZ)
        - timedelta(days=days)
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
