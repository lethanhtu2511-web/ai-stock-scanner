
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
        raise ValueError(
            "Nguồn dữ liệu không trả về lịch sử giá."
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "Nguồn dữ liệu không trả về lịch sử giá."
        )

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
        raise ValueError(
            f"Thiếu cột OHLCV: {missing}"
        )

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

    if (
        cached
        and now - cached[0] < HISTORY_CACHE_SECONDS
    ):
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
            f"{symbol}: chỉ có {len(df)} phiên, "
            "cần tối thiểu 200 phiên."
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


def _extract_symbols(result):
    if result is None:
        return []

    if isinstance(result, pd.DataFrame):
        if "symbol" not in result.columns:
            raise RuntimeError(
                "Danh sách trả về không có cột symbol."
            )

        values = result["symbol"].tolist()

    elif isinstance(result, pd.Series):
        values = result.tolist()

    else:
        values = list(result)

    symbols = []

    for value in values:
        try:
            symbols.append(
                normalize_symbol(value)
            )
        except (ValueError, TypeError):
            continue

    return symbols


def fetch_exchange_symbols():
    global _SYMBOL_CACHE
    global _SYMBOL_CACHE_AT

    now = time.monotonic()

    if (
        _SYMBOL_CACHE is not None
        and now - _SYMBOL_CACHE_AT < SYMBOL_CACHE_SECONDS
    ):
        return list(_SYMBOL_CACHE)

    # Sử dụng KBS và lấy riêng từng sàn.
    # Không gọi symbols_by_exchange().
    listing = Listing(source="KBS")

    try:
        hose_raw = listing.symbols_by_group(
            group="HOSE"
        )

        hnx_raw = listing.symbols_by_group(
            group="HNX"
        )

    except Exception as exc:
        raise RuntimeError(
            "Không lấy được danh sách mã HOSE/HNX "
            "qua symbols_by_group. Chi tiết: "
            f"{exc}"
        ) from exc

    hose_symbols = _extract_symbols(hose_raw)
    hnx_symbols = _extract_symbols(hnx_raw)

    if not hose_symbols:
        raise RuntimeError(
            "Danh sách HOSE trả về rỗng. "
            "Không tiếp tục quét để tránh thiếu dữ liệu."
        )

    if not hnx_symbols:
        raise RuntimeError(
            "Danh sách HNX trả về rỗng. "
            "Không tiếp tục quét để tránh thiếu dữ liệu."
        )

    # Gộp hai sàn, loại mã trùng.
    # Không truy vấn nhóm UPCOM.
    symbols = sorted(
        set(hose_symbols + hnx_symbols)
    )

    if not symbols:
        raise RuntimeError(
            "Không tìm thấy mã cổ phiếu HOSE/HNX."
        )

    _SYMBOL_CACHE = symbols
    _SYMBOL_CACHE_AT = time.monotonic()

    return list(_SYMBOL_CACHE)
