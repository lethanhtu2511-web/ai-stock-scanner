
import re
import time
import pandas as pd

from vn_market_data import get_ohlcv
from vnstock import Listing


HISTORY_CACHE_SECONDS = 900
SYMBOL_CACHE_SECONDS = 21600

_HISTORY_CACHE = {}

_SYMBOL_CACHE = None
_SYMBOL_CACHE_AT = 0.0


def normalize_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    return symbol


def _normalize_history(rows):
    if rows is None:
        raise ValueError("Nguồn dữ liệu không trả về lịch sử giá.")

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Nguồn dữ liệu không trả về lịch sử giá.")

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    df = df.rename(columns={
        "time": "date",
        "trading_date": "date",
        "tradingdate": "date",
        "vol": "volume"
    })

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Thiếu cột OHLCV: {missing}")

    df = df[required].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    for col in required[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df.dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    valid = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    df = df.loc[valid]

    return df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }).reset_index(drop=True)


def fetch_stock_history(symbol, length=260):
    symbol = normalize_symbol(symbol)
    length = max(220, int(length))

    key = (symbol, length)
    now = time.monotonic()

    cached = _HISTORY_CACHE.get(key)

    if cached and now - cached[0] < HISTORY_CACHE_SECONDS:
        return cached[1].copy()

    rows = get_ohlcv(
        symbol,
        lookback_days=600
    )

    df = (
        _normalize_history(rows)
        .tail(length)
        .reset_index(drop=True)
    )

    if len(df) < 200:
        raise ValueError(
            f"{symbol}: chỉ có {len(df)} phiên, cần tối thiểu 200 phiên."
        )

    df.attrs["source"] = "DNSE/Vietcap (fallback)"

    _HISTORY_CACHE[key] = (
        time.monotonic(),
        df.copy()
    )

    if len(_HISTORY_CACHE) > 500:
        oldest = min(
            _HISTORY_CACHE,
            key=lambda k: _HISTORY_CACHE[k][0]
        )
        del _HISTORY_CACHE[oldest]

    return df


def fetch_stock_quote(symbol):
    symbol = normalize_symbol(symbol)

    df = fetch_stock_history(
        symbol,
        length=260
    )

    row = df.iloc[-1]

    return {
        "symbol": symbol,
        "last_price": float(row["Close"]),
        "quote_date": pd.Timestamp(
            row["Date"]
        ).strftime("%d/%m/%Y"),
        "source": "Giá đóng cửa nến ngày"
    }


def fetch_exchange_symbols():
    global _SYMBOL_CACHE
    global _SYMBOL_CACHE_AT

    now = time.monotonic()

    if (
        _SYMBOL_CACHE is not None
        and now - _SYMBOL_CACHE_AT < SYMBOL_CACHE_SECONDS
    ):
        return list(_SYMBOL_CACHE)

    # Dùng VCI thay cho KBS vì VCI hỗ trợ
    # lấy danh sách cổ phiếu theo sàn ở cấp Cộng đồng.
    raw = Listing(
        source="VCI"
    ).symbols_by_exchange()

    if raw is None:
        raise RuntimeError(
            "Nguồn VCI không trả về danh sách cổ phiếu."
        )

    df = pd.DataFrame(raw).copy()

    if df.empty:
        raise RuntimeError(
            "Danh sách cổ phiếu từ VCI đang rỗng."
        )

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    if "symbol" not in df.columns:
        raise RuntimeError(
            "Không tìm thấy cột symbol. "
            f"Các cột nhận được: {list(df.columns)}"
        )

    exchange_col = next(
        (
            c for c in (
                "exchange",
                "com_group_code",
                "floor"
            )
            if c in df.columns
        ),
        None
    )

    if exchange_col is None:
        raise RuntimeError(
            "Không tìm thấy cột sàn giao dịch. "
            f"Các cột nhận được: {list(df.columns)}"
        )

    exchange = (
        df[exchange_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Chỉ lấy HOSE và HNX, loại UPCoM.
    df = df[
        exchange.isin([
            "HOSE",
            "HSX",
            "HNX"
        ])
    ].copy()

    # Nếu nguồn có cột loại chứng khoán,
    # chỉ lấy cổ phiếu, không lấy chứng quyền/quỹ/trái phiếu.
    if "type" in df.columns:
        stock_type = (
            df["type"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        df = df[
            stock_type.isin([
                "STOCK",
                "EQUITY",
                "CP"
            ])
        ]

    symbols = sorted({
        normalize_symbol(s)
        for s in df["symbol"].dropna().tolist()
        if re.fullmatch(
            r"[A-Z0-9]{3,10}",
            str(s).strip().upper()
        )
    })

    if not symbols:
        raise RuntimeError(
            "Không lọc được mã HOSE/HNX. "
            "Dừng quét để tránh báo cáo thiếu dữ liệu."
        )

    _SYMBOL_CACHE = symbols
    _SYMBOL_CACHE_AT = now

    return list(_SYMBOL_CACHE)
