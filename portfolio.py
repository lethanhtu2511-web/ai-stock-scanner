
import os
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

DEFAULTS = {
    "capital": 10_000_000,
    "risk_pct": 2.0,
    "fee_buy": 0.001,
    "fee_sell": 0.001,
    "sell_tax": 0.001,
}


def _connect(db_path):
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path="data/stock_scanner.sqlite3"):
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
                (key, value)
            )


def get_settings(db_path="data/stock_scanner.sqlite3"):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT key, value FROM settings"
        ).fetchall()

    result = DEFAULTS.copy()
    result.update({row["key"]: row["value"] for row in rows})
    return result


def save_settings(values, db_path="data/stock_scanner.sqlite3"):
    init_db(db_path)

    with _connect(db_path) as conn:
        for key, value in values.items():
            if key in DEFAULTS:
                conn.execute("""
                    INSERT INTO settings(key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key)
                    DO UPDATE SET value = excluded.value
                """, (key, float(value)))


def get_positions(db_path="data/stock_scanner.sqlite3"):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()

    positions = {}

    for row in rows:
        symbol = row["symbol"]
        position = positions.setdefault(symbol, {
            "symbol": symbol,
            "quantity": 0,
            "total_cost": 0.0,
        })

        quantity = int(row["quantity"])
        price = float(row["price"])
        fee = float(row["fee"])

        if row["side"] == "MUA":
            position["quantity"] += quantity
            position["total_cost"] += quantity * price + fee

        elif row["side"] == "BÁN":
            if quantity > position["quantity"]:
                raise ValueError(
                    f"Lịch sử {symbol} có lệnh bán vượt số lượng mua."
                )

            average_cost = (
                position["total_cost"] / position["quantity"]
                if position["quantity"] else 0
            )

            position["total_cost"] -= average_cost * quantity
            position["quantity"] -= quantity

    result = []

    for position in positions.values():
        if position["quantity"] > 0:
            position["avg_cost"] = (
                position["total_cost"] / position["quantity"]
            )
            result.append(position)

    return result


def calculate_trade_preview(
    side,
    symbol,
    quantity,
    price,
    settings=None,
    db_path="data/stock_scanner.sqlite3"
):
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

    if not quantity_float.is_integer() or quantity_float <= 0:
        raise ValueError("Khối lượng phải là số nguyên dương.")

    quantity = int(quantity_float)

    if price <= 0:
        raise ValueError("Giá phải lớn hơn 0.")

    current_settings = settings or get_settings(db_path)

    gross = quantity * price

    fee_rate = (
        current_settings["fee_buy"]
        if side == "MUA"
        else current_settings["fee_sell"]
    )

    fee = gross * fee_rate
    tax = (
        gross * current_settings["sell_tax"]
        if side == "BÁN"
        else 0.0
    )

    if side == "BÁN":
        held = {
            position["symbol"]: position["quantity"]
            for position in get_positions(db_path)
        }

        if quantity > held.get(symbol, 0):
            raise ValueError(
                f"Bạn chỉ đang có {held.get(symbol, 0):,} cổ phiếu {symbol}."
            )

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
        "tax_rate": (
            current_settings["sell_tax"]
            if side == "BÁN" else 0.0
        ),
    }


def add_transaction(
    side,
    symbol,
    quantity,
    price,
    note="",
    db_path="data/stock_scanner.sqlite3"
):
    init_db(db_path)

    side = str(side).upper().strip()
    symbol = str(symbol).upper().strip()

    settings = get_settings(db_path)

    preview = calculate_trade_preview(
        side, symbol, quantity, price, settings, db_path
    )

    created_at = datetime.now(VN_TZ).isoformat(timespec="seconds")

    with _connect(db_path) as conn:
        conn.execute("""
            INSERT INTO transactions (
                created_at, side, symbol, quantity,
                price, fee, tax, note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            created_at,
            side,
            symbol,
            int(quantity),
            float(price),
            preview["fee"],
            preview["tax"],
            str(note)
        ))

    return preview


def get_transactions(db_path="data/stock_scanner.sqlite3"):
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


def get_portfolio_summary(db_path="data/stock_scanner.sqlite3"):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY id"
        ).fetchall()

    realized_pnl = 0.0
    positions = {}

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

            average_cost = (
                state["cost"] / state["quantity"]
                if state["quantity"] else 0
            )

            realized_pnl += (
                gross - fee - tax - average_cost * quantity
            )

            state["quantity"] -= quantity
            state["cost"] -= average_cost * quantity

    return {
        "realized_pnl": realized_pnl,
        "total_cost": sum(item["cost"] for item in positions.values()),
        "open_quantity": sum(
            item["quantity"] for item in positions.values()
        ),
    }


def get_cash_summary(db_path="data/stock_scanner.sqlite3"):
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute("""
            SELECT side, quantity, price, fee, tax
            FROM transactions
        """).fetchall()

    net_buys = sum(
        row["quantity"] * row["price"] + row["fee"]
        for row in rows if row["side"] == "MUA"
    )

    net_sells = sum(
        row["quantity"] * row["price"] - row["fee"] - row["tax"]
        for row in rows if row["side"] == "BÁN"
    )

    return {
        "net_buys": net_buys,
        "net_sells": net_sells,
    }
