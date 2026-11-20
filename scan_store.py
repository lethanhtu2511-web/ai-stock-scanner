
import os
import json
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_DB_PATH = os.getenv(
    "DATABASE_PATH",
    "data/stock_scanner.sqlite3",
)

JSON_FIELDS = {
    "blockers",
    "reasons",
    "penalties",
    "early_reasons",
    "early_risks",
}


# =====================================================
# DATABASE
# =====================================================

def connect(db_path=DEFAULT_DB_PATH):
    folder = os.path.dirname(db_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")

    return conn


def init_scan_db(db_path=DEFAULT_DB_PATH):
    with connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal TEXT,
                score REAL,
                price REAL,
                stop_loss REAL,
                target_price REAL,
                reward_risk REAL,
                reason TEXT,
                data_date TEXT,
                stop_distance_pct REAL,
                target_distance_pct REAL,
                atr_pct REAL,
                relative_strength REAL,
                avg_turnover_20d REAL,
                median_turnover_20d REAL,
                market_context TEXT,
                blockers TEXT,
                reasons TEXT,
                penalties TEXT,
                d1_close REAL,
                realtime_price REAL,
                kbs_quote_price REAL,
                realtime_volume REAL,
                quote_source_time TEXT,
                quote_fetched_at TEXT,
                quote_age_seconds REAL,
                quote_source TEXT,
                price_data_status TEXT,
                price_data_message TEXT,
                live_change_pct REAL,
                market_regime TEXT,
                early_signal TEXT,
                early_score REAL,
                early_reasons TEXT,
                early_risks TEXT,
                early_volume_ratio REAL,
                early_resistance REAL,
                early_quote_verified INTEGER
            )
        """)

        # Nâng cấp an toàn cho SQLite cũ.
        existing_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(scan_results)"
            ).fetchall()
        }

        if "live_change_pct" not in existing_columns:
            conn.execute(
                "ALTER TABLE scan_results ADD COLUMN live_change_pct REAL"
            )

        if "market_regime" not in existing_columns:
            conn.execute(
                "ALTER TABLE scan_results ADD COLUMN market_regime TEXT"
            )

        early_columns = {
            "early_signal": "TEXT",
            "early_score": "REAL",
            "early_reasons": "TEXT",
            "early_risks": "TEXT",
            "early_volume_ratio": "REAL",
            "early_resistance": "REAL",
            "early_quote_verified": "INTEGER",
        }
        for column, sql_type in early_columns.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE scan_results ADD COLUMN {column} {sql_type}")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_scan_results_scanned_at
            ON scan_results(scanned_at)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_scan_results_symbol
            ON scan_results(symbol)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_scan_results_strategy
            ON scan_results(strategy)
        """)


# =====================================================
# SERIALIZATION
# =====================================================

def _serialize_value(field, value):
    if value is None:
        return None

    if field in JSON_FIELDS:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            return "[]"

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return value


def _format_row(row):
    result = dict(row)

    for field in JSON_FIELDS:
        value = result.get(field)

        if value is None:
            result[field] = []
            continue

        if isinstance(value, list):
            continue

        try:
            parsed = json.loads(value)
            result[field] = (
                parsed if isinstance(parsed, list) else []
            )
        except (TypeError, ValueError):
            result[field] = []

    return result


# =====================================================
# SAVE SCAN RESULTS
# =====================================================

def save_scan_results(rows, db_path=DEFAULT_DB_PATH):
    """
    Lưu toàn bộ kết quả của một lượt quét.

    Tất cả bản ghi trong cùng một lượt quét sử dụng
    chung scanned_at để get_latest_scan lấy đúng một batch.
    """

    init_scan_db(db_path)

    if not rows:
        return 0

    scanned_at = datetime.now(VN_TZ).isoformat(
        timespec="seconds"
    )

    columns = [
        "scanned_at",
        "symbol",
        "strategy",
        "signal",
        "score",
        "price",
        "stop_loss",
        "target_price",
        "reward_risk",
        "reason",
        "data_date",
        "stop_distance_pct",
        "target_distance_pct",
        "atr_pct",
        "relative_strength",
        "avg_turnover_20d",
        "median_turnover_20d",
        "market_context",
        "blockers",
        "reasons",
        "penalties",
        "d1_close",
        "realtime_price",
        "kbs_quote_price",
        "realtime_volume",
        "quote_source_time",
        "quote_fetched_at",
        "quote_age_seconds",
        "quote_source",
        "price_data_status",
        "price_data_message",
        "live_change_pct",
        "market_regime",
        "early_signal",
        "early_score",
        "early_reasons",
        "early_risks",
        "early_volume_ratio",
        "early_resistance",
        "early_quote_verified",
    ]

    placeholders = ", ".join(["?"] * len(columns))

    sql = f"""
        INSERT INTO scan_results (
            {", ".join(columns)}
        )
        VALUES ({placeholders})
    """

    values = []

    for row in rows:
        symbol = str(
            row.get("symbol", "")
        ).strip().upper()

        strategy = str(
            row.get("strategy", "")
        ).strip().upper()

        if not symbol or not strategy:
            continue

        record = {
            "scanned_at": scanned_at,
            **row,
        }

        values.append(tuple(
            _serialize_value(
                column,
                record.get(column),
            )
            for column in columns
        ))

    if not values:
        return 0

    with connect(db_path) as conn:
        conn.executemany(sql, values)

    # Mirror completed scan batches to Supabase so the Windows worker and
    # Streamlit Cloud UI share one result store. Service key stays server-side.
    try:
        from portfolio import _get_supabase
        client = _get_supabase()
        if client is not None:
            remote_rows = []
            for row in rows:
                symbol = str(row.get("symbol", "")).strip().upper()
                strategy = str(row.get("strategy", "")).strip().upper()
                if not symbol or not strategy:
                    continue
                payload = {
                    key: _serialize_value(key, value)
                    for key, value in row.items()
                }
                remote_rows.append({
                    "scanned_at": scanned_at,
                    "symbol": symbol,
                    "strategy": strategy,
                    "signal": row.get("signal"),
                    "score": row.get("score"),
                    "payload": payload,
                })
            for offset in range(0, len(remote_rows), 500):
                client.table("scan_results").insert(
                    remote_rows[offset:offset + 500]
                ).execute()
    except Exception:
        # Keep local results; caller logs and can diagnose missing cloud sync.
        import logging
        logging.getLogger("scan_store").exception("Không đồng bộ được scan batch lên Supabase.")

    return len(values)


# =====================================================
# LATEST SCAN
# =====================================================

def get_latest_scan(db_path=DEFAULT_DB_PATH):
    """
    Chỉ lấy toàn bộ kết quả thuộc đúng lượt quét mới nhất.
    Không trộn dữ liệu từ nhiều lượt quét khác nhau.
    """

    init_scan_db(db_path)

    # Prefer shared Supabase results so phone/cloud UI can see PC worker scans.
    try:
        from portfolio import _get_supabase
        client = _get_supabase()
        if client is not None:
            remote = (
                client.table("scan_results")
                .select("scanned_at,symbol,strategy,signal,score,payload")
                .order("scanned_at", desc=True)
                .limit(5000)
                .execute().data or []
            )
            if remote:
                latest_batch = remote[0]["scanned_at"]
                selected = [r for r in remote if r["scanned_at"] == latest_batch]
                output = []
                for item in selected:
                    payload = item.get("payload") or {}
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except (TypeError, ValueError):
                            payload = {}
                    payload.setdefault("symbol", item.get("symbol"))
                    payload.setdefault("strategy", item.get("strategy"))
                    payload.setdefault("signal", item.get("signal"))
                    payload.setdefault("score", item.get("score"))
                    payload["scanned_at"] = latest_batch
                    for field in JSON_FIELDS:
                        value = payload.get(field)
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                                payload[field] = parsed if isinstance(parsed, list) else []
                            except (TypeError, ValueError):
                                payload[field] = []
                        elif value is None:
                            payload[field] = []
                    output.append(payload)
                return output
    except Exception:
        import logging
        logging.getLogger("scan_store").exception("Không đọc được scan batch từ Supabase; dùng SQLite.")

    with connect(db_path) as conn:
        latest_batch = conn.execute("SELECT MAX(scanned_at) FROM scan_results").fetchone()[0]
        if not latest_batch:
            return []
        rows = conn.execute("""
            SELECT * FROM scan_results WHERE scanned_at = ?
            ORDER BY CASE signal WHEN 'BUY' THEN 5 WHEN 'WAIT' THEN 4 WHEN 'WATCH' THEN 3 WHEN 'SKIP' THEN 2 WHEN 'FILTERED' THEN 1 ELSE 0 END DESC,
            COALESCE(score, 0) DESC, symbol ASC
        """, (latest_batch,)).fetchall()
    return [_format_row(row) for row in rows]


# =====================================================
# SCAN HISTORY
# =====================================================

def get_scan_history(
    limit=150,
    db_path=DEFAULT_DB_PATH,
):
    init_scan_db(db_path)
    try:
        limit = max(1, min(int(limit), 5000))
    except (TypeError, ValueError):
        limit = 150
    try:
        from portfolio import _get_supabase
        client = _get_supabase()
        if client is not None:
            remote = (
                client.table("scan_results")
                .select("scanned_at,symbol,strategy,signal,score,payload")
                .order("scanned_at", desc=True)
                .limit(limit)
                .execute().data or []
            )
            if remote:
                output = []
                for item in remote:
                    payload = item.get("payload") or {}
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except (TypeError, ValueError):
                            payload = {}
                    payload.setdefault("symbol", item.get("symbol"))
                    payload.setdefault("strategy", item.get("strategy"))
                    payload.setdefault("signal", item.get("signal"))
                    payload.setdefault("score", item.get("score"))
                    payload["scanned_at"] = item.get("scanned_at")
                    for field in JSON_FIELDS:
                        value = payload.get(field)
                        if isinstance(value, str):
                            try:
                                parsed = json.loads(value)
                                payload[field] = parsed if isinstance(parsed, list) else []
                            except (TypeError, ValueError):
                                payload[field] = []
                        elif value is None:
                            payload[field] = []
                    output.append(payload)
                return output
    except Exception:
        import logging
        logging.getLogger("scan_store").exception("Không đọc được scan history từ Supabase; dùng SQLite.")

    with connect(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM scan_results
            ORDER BY scanned_at DESC,
                CASE signal WHEN 'BUY' THEN 5 WHEN 'WAIT' THEN 4 WHEN 'WATCH' THEN 3 WHEN 'SKIP' THEN 2 WHEN 'FILTERED' THEN 1 ELSE 0 END DESC,
                COALESCE(score, 0) DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [_format_row(row) for row in rows]

# =====================================================
# SCAN SUMMARY
# =====================================================

def get_scan_summary(db_path=DEFAULT_DB_PATH):
    init_scan_db(db_path)

    try:
        from portfolio import _get_supabase
        client = _get_supabase()
        if client is not None:
            remote = (
                client.table("scan_results")
                .select("scanned_at,signal")
                .order("scanned_at", desc=True)
                .limit(5000)
                .execute().data or []
            )
            if remote:
                latest = remote[0]["scanned_at"]
                batch = [r for r in remote if r["scanned_at"] == latest]
                signals = [r.get("signal") for r in batch]
                return {
                    "latest_scan_at": latest,
                    "total_rows": len(remote),
                    "latest_batch_rows": len(batch),
                    "buy_count": signals.count("BUY"),
                    "wait_count": signals.count("WAIT"),
                    "watch_count": signals.count("WATCH"),
                }
    except Exception:
        import logging
        logging.getLogger("scan_store").exception("Không đọc được summary Supabase; dùng SQLite.")

    with connect(db_path) as conn:
        row = conn.execute("""
            SELECT
                MAX(scanned_at) AS latest_scan_at,
                COUNT(*) AS total_rows
            FROM scan_results
        """).fetchone()

        latest = row["latest_scan_at"]

        if not latest:
            return {
                "latest_scan_at": None,
                "total_rows": 0,
                "latest_batch_rows": 0,
                "buy_count": 0,
                "wait_count": 0,
                "watch_count": 0,
            }

        batch = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN signal = 'BUY' THEN 1 ELSE 0 END) AS buys,
                SUM(CASE WHEN signal = 'WAIT' THEN 1 ELSE 0 END) AS waits,
                SUM(CASE WHEN signal = 'WATCH' THEN 1 ELSE 0 END) AS watches
            FROM scan_results
            WHERE scanned_at = ?
        """, (latest,)).fetchone()

    return {
        "latest_scan_at": latest,
        "total_rows": row["total_rows"] or 0,
        "latest_batch_rows": batch["total"] or 0,
        "buy_count": batch["buys"] or 0,
        "wait_count": batch["waits"] or 0,
        "watch_count": batch["watches"] or 0,
    }
