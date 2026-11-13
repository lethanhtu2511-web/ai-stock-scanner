
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
    "fee_buy": 0.001,
    "fee_sell": 0.001,
    "sell_tax": 0.001,
}


def _connect(db_path=DEFAULT_DB_PATH):
    folder = os.path.dirname(db_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(
        db_path,
        timeout=30,
        isolation_level="DEFERRED"
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")

    return conn


def init_db(db_path=DEFAULT_DB_PATH):
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
                """
                INSERT OR IGNORE INTO settings(key, value)
                VALUES (?, ?)
                """,
                (key, value)
            )


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

        else:
            if not 0 <= number <= 0.1:
                raise ValueError(
                    f"Tỷ lệ {key} phải nằm trong khoảng 0% đến 10%."
                )

        result[key] = number

    return result


def get_settings(db_path=DEFAULT_DB_PATH):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT key, value FROM settings"
        ).fetchall()

    result = DEFAULTS.copy()
    result.update({
        row["key"]: row["value"]
        for row in rows
        if row["key"] in DEFAULTS
    })

    return result


def save_settings(values, db_path=DEFAULT_DB_PATH):
    init_db(db_path)

    if not isinstance(values, dict):
        raise ValueError("Cài đặt phải ở dạng dữ liệu hợp lệ.")

    validated = _validate_settings(values)

    with _connect(db_path) as conn:
        for key, value in validated.items():
            conn.execute("""
                INSERT INTO settings(key, value)
                VALUES (?, ?)
                ON CONFLICT(key)
                DO UPDATE SET value = excluded.value
            """, (key, value))


def _positions_from_rows(rows):
    positions = {}

    for row in rows:
        symbol = row["symbol"]

        state = positions.setdefault(symbol, {
            "symbol": symbol,
            "quantity": 0,
            "total_cost": 0.0,
        })

        quantity = int(row["quantity"])
        price = float(row["price"])
        fee = float(row["fee"])

        if row["side"] == "MUA":
            state["quantity"] += quantity
            state["total_cost"] += quantity * price + fee

        elif row["side"] == "BÁN":
            if quantity > state["quantity"]:
                raise ValueError(
                    f"Lịch sử {symbol} có lệnh bán vượt số lượng mua."
                )

            if state["quantity"] <= 0:
                raise ValueError(
                    f"Lịch sử giao dịch {symbol} không hợp lệ."
                )

            average_cost = (
                state["total_cost"] / state["quantity"]
            )

            state["total_cost"] -= average_cost * quantity
            state["quantity"] -= quantity

            if state["quantity"] == 0:
                state["total_cost"] = 0.0

    return positions


def _read_positions(db_path=DEFAULT_DB_PATH):
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()

    return _positions_from_rows(rows)


def get_positions(db_path=DEFAULT_DB_PATH):
    init_db(db_path)

    positions = _read_positions(db_path)
    result = []

    for position in positions.values():
        if position["quantity"] <= 0:
            continue

        item = position.copy()
        item["avg_cost"] = (
            item["total_cost"] / item["quantity"]
        )

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

    fee_rate = float(
        settings["fee_buy"]
        if side == "MUA"
        else settings["fee_sell"]
    )

    tax_rate = (
        float(settings["sell_tax"])
        if side == "BÁN"
        else 0.0
    )

    fee = gross * fee_rate
    tax = gross * tax_rate

    cash_flow = (
        gross + fee
        if side == "MUA"
        else gross - fee - tax
    )

    return {
        "gross": gross,
        "fee": fee,
        "tax": tax,
        "cash_flow": cash_flow,
        "fee_rate": fee_rate,
        "tax_rate": tax_rate,
    }


def calculate_trade_preview(
    side,
    symbol,
    quantity,
    price,
    settings=None,
    db_path=DEFAULT_DB_PATH
):
    side, symbol, quantity, price = _validate_trade_input(
        side, symbol, quantity, price
    )

    current_settings = (
        get_settings(db_path)
        if settings is None
        else {
            **DEFAULTS,
            **_validate_settings(settings)
        }
    )

    preview = _build_preview(
        side,
        quantity,
        price,
        current_settings
    )

    if side == "BÁN":
        positions = _read_positions(db_path)
        held = positions.get(symbol, {}).get("quantity", 0)

        if quantity > held:
            raise ValueError(
                f"Bạn chỉ đang có {held:,} cổ phiếu {symbol}."
            )

    return preview


def add_transaction(
    side,
    symbol,
    quantity,
    price,
    note="",
    db_path=DEFAULT_DB_PATH
):
    init_db(db_path)

    side, symbol, quantity, price = _validate_trade_input(
        side, symbol, quantity, price
    )

    settings = get_settings(db_path)

    preview = _build_preview(
        side,
        quantity,
        price,
        settings
    )

    created_at = datetime.now(VN_TZ).isoformat(
        timespec="seconds"
    )

    # Khóa ghi trước khi kiểm tra số lượng để tránh
    # hai thao tác bán đồng thời làm bán vượt số đang nắm giữ.
    with _connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")

        if side == "BÁN":
            rows = conn.execute(
                "SELECT * FROM transactions ORDER BY id"
            ).fetchall()

            positions = _positions_from_rows(rows)
            held = positions.get(symbol, {}).get("quantity", 0)

            if quantity > held:
                raise ValueError(
                    f"Bạn chỉ đang có {held:,} cổ phiếu {symbol}."
                )

        conn.execute("""
            INSERT INTO transactions (
                created_at,
                side,
                symbol,
                quantity,
                price,
                fee,
                tax,
                note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            created_at,
            side,
            symbol,
            quantity,
            price,
            preview["fee"],
            preview["tax"],
            str(note).strip()[:1000]
        ))

    return preview


def get_transactions(db_path=DEFAULT_DB_PATH):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT
                created_at AS 'Thời gian',
                side AS 'Loại',
                symbol AS 'Mã',
                quantity AS 'Khối lượng',
                price AS 'Giá khớp',
                fee AS 'Phí',
                tax AS 'Thuế',
                note AS 'Ghi chú'
            FROM transactions
            ORDER BY id DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_portfolio_summary(db_path=DEFAULT_DB_PATH):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()

    positions = {}
    realized_pnl = 0.0

    for row in rows:
        symbol = row["symbol"]
        quantity = int(row["quantity"])
        gross = quantity * float(row["price"])
        fee = float(row["fee"])
        tax = float(row["tax"])

        state = positions.setdefault(symbol, {
            "quantity": 0,
            "cost": 0.0
        })

        if row["side"] == "MUA":
            state["quantity"] += quantity
            state["cost"] += gross + fee

        elif row["side"] == "BÁN":
            if quantity > state["quantity"]:
                raise ValueError(
                    f"Lịch sử {symbol} có lệnh bán vượt số lượng mua."
                )

            if state["quantity"] <= 0:
                raise ValueError(
                    f"Lịch sử giao dịch {symbol} không hợp lệ."
                )

            average_cost = state["cost"] / state["quantity"]

            realized_pnl += (
                gross - fee - tax - average_cost * quantity
            )

            state["quantity"] -= quantity
            state["cost"] -= average_cost * quantity

            if state["quantity"] == 0:
                state["cost"] = 0.0

    return {
        "realized_pnl": realized_pnl,
        "total_cost": sum(
            item["cost"] for item in positions.values()
        ),
        "open_quantity": sum(
            item["quantity"] for item in positions.values()
        ),
    }


def get_cash_summary(db_path=DEFAULT_DB_PATH):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT side, quantity, price, fee, tax
            FROM transactions
        """).fetchall()

    net_buys = sum(
        row["quantity"] * row["price"] + row["fee"]
        for row in rows
        if row["side"] == "MUA"
    )

    net_sells = sum(
        row["quantity"] * row["price"] - row["fee"] - row["tax"]
        for row in rows
        if row["side"] == "BÁN"
    )

    return {
        "net_buys": net_buys,
        "net_sells": net_sells,
    }
