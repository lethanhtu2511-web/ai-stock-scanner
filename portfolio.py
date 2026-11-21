"""Portfolio storage with Supabase persistence and SQLite fallback.

When SUPABASE_URL and a server-side SUPABASE_SECRET_KEY are configured,
portfolio transactions/settings are stored in Supabase. Existing SQLite
transactions are imported idempotently using legacy_id and are not deleted.
Without Supabase credentials, local SQLite behavior is retained.
"""
import os
import math
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_DB_PATH = "data/stock_scanner.sqlite3"

DEFAULTS = {
    "capital": 10_000_000,
    "risk_pct": 2.0,
    "fee_buy": 0.0015,
    "fee_sell": 0.0015,
    "sell_tax": 0.001,
}

_supabase_client = None
_supabase_client_key = None


def _config_value(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _get_supabase():
    global _supabase_client, _supabase_client_key
    url = _config_value("SUPABASE_URL")
    key = (
        _config_value("SUPABASE_SECRET_KEY")
        or _config_value("SUPABASE_SERVICE_ROLE_KEY")
    )
    if not url or not key:
        return None
    cache_key = (url, key)
    if _supabase_client is None or _supabase_client_key != cache_key:
        try:
            from supabase import create_client
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu thư viện supabase. Hãy cài requirements.txt mới nhất."
            ) from exc
        _supabase_client = create_client(url, key)
        _supabase_client_key = cache_key
    return _supabase_client


def _connect(db_path=DEFAULT_DB_PATH):
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, isolation_level="DEFERRED")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _init_sqlite(db_path=DEFAULT_DB_PATH):
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                side TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                fee REAL NOT NULL,
                tax REAL NOT NULL,
                note TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """)
        for key, value in DEFAULTS.items():
            conn.execute(
                "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
                (key, value),
            )


def _migrate_sqlite_to_supabase(client, db_path):
    """Idempotently copy legacy local data; keep SQLite untouched."""
    if not os.path.exists(db_path):
        return
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT id, created_at, side, symbol, quantity, price, fee, tax, note "
                "FROM transactions ORDER BY id"
            ).fetchall()
            settings_rows = conn.execute(
                "SELECT key, value FROM settings"
            ).fetchall()
        for row in rows:
            payload = {
                "legacy_id": str(row["id"]),
                "created_at": row["created_at"],
                "side": row["side"],
                "symbol": row["symbol"],
                "quantity": int(row["quantity"]),
                "price": float(row["price"]),
                "fee": float(row["fee"]),
                "tax": float(row["tax"]),
                "note": row["note"] or "",
            }
            client.table("transactions").upsert(
                payload, on_conflict="legacy_id"
            ).execute()
        # Import legacy settings only when the key is absent remotely.
        # A fresh Streamlit instance may contain only local defaults; those
        # must never overwrite settings already saved by the user in Supabase.
        remote_settings = client.table("settings").select("key").execute().data or []
        existing_keys = {item["key"] for item in remote_settings}
        for row in settings_rows:
            if row["key"] in DEFAULTS and row["key"] not in existing_keys:
                client.table("settings").insert(
                    {"key": row["key"], "value": float(row["value"])}
                ).execute()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Không thể đọc dữ liệu SQLite cũ: {exc}") from exc


def init_db(db_path=DEFAULT_DB_PATH):
    _init_sqlite(db_path)
    client = _get_supabase()
    if client is not None:
        _migrate_sqlite_to_supabase(client, db_path)


def _validate_settings(values):
    result = {}
    for key, value in values.items():
        if key not in DEFAULTS:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Giá trị cài đặt {key} không hợp lệ.")
        if not math.isfinite(number):
            raise ValueError(f"Giá trị cài đặt {key} không hợp lệ.")
        if key == "capital":
            if number < 0:
                raise ValueError("Vốn không được âm.")
        elif key == "risk_pct":
            if not 0 < number <= 10:
                raise ValueError("Rủi ro mỗi lệnh phải lớn hơn 0 và không quá 10%.")
        elif not 0 <= number <= 0.1:
            raise ValueError(f"Tỷ lệ {key} phải nằm trong khoảng 0% đến 10%.")
        result[key] = number
    return result


def _remote_transactions(client):
    response = (
        client.table("transactions")
        .select("*")
        .order("created_at")
        .order("id")
        .execute()
    )
    return response.data or []


def get_storage_backend():
    """Identify whether portfolio records use durable remote storage or local SQLite."""
    return "SUPABASE" if _get_supabase() is not None else "SQLITE"


def get_settings(db_path=DEFAULT_DB_PATH):
    init_db(db_path)
    client = _get_supabase()
    if client is not None:
        rows = client.table("settings").select("key,value").execute().data or []
        result = DEFAULTS.copy()
        result.update({
            row["key"]: float(row["value"])
            for row in rows if row["key"] in DEFAULTS
        })
        return result
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    result = DEFAULTS.copy()
    result.update({row["key"]: row["value"] for row in rows if row["key"] in DEFAULTS})
    return result


def save_settings(values, db_path=DEFAULT_DB_PATH):
    init_db(db_path)
    if not isinstance(values, dict):
        raise ValueError("Cài đặt phải ở dạng dữ liệu hợp lệ.")
    validated = _validate_settings(values)
    client = _get_supabase()
    if client is not None:
        for key, value in validated.items():
            client.table("settings").upsert(
                {"key": key, "value": value, "updated_at": datetime.now(VN_TZ).isoformat()},
                on_conflict="key",
            ).execute()
        return
    with _connect(db_path) as conn:
        for key, value in validated.items():
            conn.execute("""
                INSERT INTO settings(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (key, value))


def _positions_from_rows(rows):
    positions = {}
    for row in rows:
        symbol = row["symbol"]
        state = positions.setdefault(symbol, {"symbol": symbol, "quantity": 0, "total_cost": 0.0})
        quantity = int(row["quantity"])
        price = float(row["price"])
        fee = float(row["fee"])
        if row["side"] == "MUA":
            state["quantity"] += quantity
            state["total_cost"] += quantity * price + fee
        elif row["side"] == "BÁN":
            if quantity > state["quantity"]:
                raise ValueError(f"Lịch sử {symbol} có lệnh bán vượt số lượng mua.")
            if state["quantity"] <= 0:
                raise ValueError(f"Lịch sử giao dịch {symbol} không hợp lệ.")
            average_cost = state["total_cost"] / state["quantity"]
            state["total_cost"] -= average_cost * quantity
            state["quantity"] -= quantity
            if state["quantity"] == 0:
                state["total_cost"] = 0.0
    return positions


def _read_positions(db_path=DEFAULT_DB_PATH):
    client = _get_supabase()
    rows = _remote_transactions(client) if client is not None else None
    if rows is not None:
        return _positions_from_rows(rows)
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM transactions ORDER BY id").fetchall()
    return _positions_from_rows(rows)


def get_positions(db_path=DEFAULT_DB_PATH):
    init_db(db_path)
    positions = _read_positions(db_path)
    result = []
    for position in positions.values():
        if position["quantity"] <= 0:
            continue
        item = position.copy()
        item["avg_cost"] = item["total_cost"] / item["quantity"]
        result.append(item)
    return result


def _validate_trade_input(side, symbol, quantity, price):
    side = str(side).upper().strip()
    symbol = str(symbol).upper().strip()
    if side not in ("MUA", "BÁN"):
        raise ValueError("Loại giao dịch không hợp lệ.")
    if not symbol.isalnum() or not 3 <= len(symbol) <= 10:
        raise ValueError("Mã cổ phiếu không hợp lệ.")
    try:
        quantity_float = float(quantity)
        price = float(price)
    except (TypeError, ValueError):
        raise ValueError("Khối lượng và giá phải là số hợp lệ.")
    if not math.isfinite(quantity_float):
        raise ValueError("Khối lượng không hợp lệ.")
    if not quantity_float.is_integer() or quantity_float <= 0:
        raise ValueError("Khối lượng phải là số nguyên dương.")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Giá phải là số lớn hơn 0.")
    return side, symbol, int(quantity_float), price


def _build_preview(side, quantity, price, settings):
    gross = quantity * price
    fee_rate = float(settings["fee_buy"] if side == "MUA" else settings["fee_sell"])
    tax_rate = float(settings["sell_tax"]) if side == "BÁN" else 0.0
    fee = gross * fee_rate
    tax = gross * tax_rate
    cash_flow = gross + fee if side == "MUA" else gross - fee - tax
    return {
        "gross": gross, "fee": fee, "tax": tax, "cash_flow": cash_flow,
        "fee_rate": fee_rate, "tax_rate": tax_rate,
    }


def calculate_trade_preview(side, symbol, quantity, price, settings=None, db_path=DEFAULT_DB_PATH):
    side, symbol, quantity, price = _validate_trade_input(side, symbol, quantity, price)
    current_settings = get_settings(db_path) if settings is None else {**DEFAULTS, **_validate_settings(settings)}
    preview = _build_preview(side, quantity, price, current_settings)
    if side == "BÁN":
        held = _read_positions(db_path).get(symbol, {}).get("quantity", 0)
        if quantity > held:
            raise ValueError(f"Bạn chỉ đang có {held:,} cổ phiếu {symbol}.")
    return preview


def add_transaction(side, symbol, quantity, price, note="", db_path=DEFAULT_DB_PATH):
    init_db(db_path)
    side, symbol, quantity, price = _validate_trade_input(side, symbol, quantity, price)
    settings = get_settings(db_path)
    preview = _build_preview(side, quantity, price, settings)
    created_at = datetime.now(VN_TZ).isoformat(timespec="microseconds")
    client = _get_supabase()
    if client is not None:
        if side == "BÁN":
            held = _read_positions(db_path).get(symbol, {}).get("quantity", 0)
            if quantity > held:
                raise ValueError(f"Bạn chỉ đang có {held:,} cổ phiếu {symbol}.")
        client.table("transactions").insert({
            "created_at": created_at, "side": side, "symbol": symbol,
            "quantity": quantity, "price": price, "fee": preview["fee"],
            "tax": preview["tax"], "note": str(note).strip()[:1000],
        }).execute()
        return preview
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if side == "BÁN":
            rows = conn.execute("SELECT * FROM transactions ORDER BY id").fetchall()
            held = _positions_from_rows(rows).get(symbol, {}).get("quantity", 0)
            if quantity > held:
                raise ValueError(f"Bạn chỉ đang có {held:,} cổ phiếu {symbol}.")
        conn.execute("""
            INSERT INTO transactions (created_at, side, symbol, quantity, price, fee, tax, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (created_at, side, symbol, quantity, price, preview["fee"], preview["tax"], str(note).strip()[:1000]))
    return preview


def get_transactions(db_path=DEFAULT_DB_PATH):
    init_db(db_path)
    client = _get_supabase()
    if client is not None:
        rows = list(reversed(_remote_transactions(client)))
    else:
        with _connect(db_path) as conn:
            rows = [dict(row) for row in conn.execute("""
                SELECT created_at AS 'Thời gian', side AS 'Loại', symbol AS 'Mã',
                       quantity AS 'Khối lượng', price AS 'Giá khớp', fee AS 'Phí',
                       tax AS 'Thuế', note AS 'Ghi chú'
                FROM transactions ORDER BY id DESC
            """).fetchall()]
        return rows
    return [{
        "Thời gian": row["created_at"], "Loại": row["side"], "Mã": row["symbol"],
        "Khối lượng": row["quantity"], "Giá khớp": row["price"],
        "Phí": row["fee"], "Thuế": row["tax"], "Ghi chú": row.get("note", ""),
    } for row in rows]


def get_portfolio_summary(db_path=DEFAULT_DB_PATH):
    init_db(db_path)
    client = _get_supabase()
    if client is not None:
        rows = _remote_transactions(client)
    else:
        with _connect(db_path) as conn:
            rows = [dict(row) for row in conn.execute("SELECT * FROM transactions ORDER BY id").fetchall()]
    positions = {}
    realized_pnl = 0.0
    for row in rows:
        symbol = row["symbol"]
        quantity = int(row["quantity"])
        gross = quantity * float(row["price"])
        fee = float(row["fee"])
        tax = float(row["tax"])
        state = positions.setdefault(symbol, {"quantity": 0, "cost": 0.0})
        if row["side"] == "MUA":
            state["quantity"] += quantity
            state["cost"] += gross + fee
        elif row["side"] == "BÁN":
            if quantity > state["quantity"]:
                raise ValueError(f"Lịch sử {symbol} có lệnh bán vượt số lượng mua.")
            if state["quantity"] <= 0:
                raise ValueError(f"Lịch sử giao dịch {symbol} không hợp lệ.")
            average_cost = state["cost"] / state["quantity"]
            realized_pnl += gross - fee - tax - average_cost * quantity
            state["quantity"] -= quantity
            state["cost"] -= average_cost * quantity
            if state["quantity"] == 0:
                state["cost"] = 0.0
    return {
        "realized_pnl": realized_pnl,
        "total_cost": sum(item["cost"] for item in positions.values()),
        "open_quantity": sum(item["quantity"] for item in positions.values()),
    }


def get_cash_summary(db_path=DEFAULT_DB_PATH):
    init_db(db_path)
    client = _get_supabase()
    if client is not None:
        rows = _remote_transactions(client)
    else:
        with _connect(db_path) as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT side, quantity, price, fee, tax FROM transactions"
            ).fetchall()]
    net_buys = sum(
        row["quantity"] * row["price"] + row["fee"]
        for row in rows if row["side"] == "MUA"
    )
    net_sells = sum(
        row["quantity"] * row["price"] - row["fee"] - row["tax"]
        for row in rows if row["side"] == "BÁN"
    )
    return {"net_buys": net_buys, "net_sells": net_sells}
