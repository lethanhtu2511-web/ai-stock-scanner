import re
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd

from vn_market_data import get_ohlcv
from vnstock import Listing


HISTORY_CACHE_SECONDS = 900
SYMBOL_CACHE_SECONDS = 21600

_HISTORY_CACHE = {}
_SYMBOL_CACHE = None
_SYMBOL_CACHE_AT = 0.0

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Chỉ sử dụng nến ngày sau khi đã qua thời điểm này.
DAILY_CANDLE_READY_TIME = dt_time(15, 30)


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


def _remove_incomplete_today_candle(df):
    """
    Trước 15:30, loại nến có ngày bằng ngày hiện tại.
    Sau 15:30, giữ lại nếu nguồn dữ liệu đã trả về.
    """

    if df.empty:
        return df

    now = datetime.now(VN_TZ)
    today = pd.Timestamp(now.date())

    if now.time() < DAILY_CANDLE_READY_TIME:

        dates = pd.to_datetime(
            df["Date"],
            errors="coerce"
        ).dt.normalize()

        df = df.loc[
            dates < today
        ].copy()

    return df.reset_index(drop=True)


def fetch_stock_history(symbol, length=260):
    symbol = normalize_symbol(symbol)
    length = max(220, int(length))

    # Cache ngắn để hạn chế dữ liệu cũ.
    key = (symbol, length)
    now = time.monotonic()

    cached = _HISTORY_CACHE.get(key)

    if (
        cached
        and now - cached[0] < HISTORY_CACHE_SECONDS
    ):
        return cached[1].copy()

    try:
        rows = get_ohlcv(
            symbol,
            lookback_days=600
        )

    except Exception as exc:
        raise RuntimeError(
            f"{symbol}: không lấy được dữ liệu nến ngày: {exc}"
        ) from exc

    df = _normalize_history(rows)

    # Không dùng nến đang hình thành trong phiên.
    df = _remove_incomplete_today_candle(df)

    df = (
        df.tail(length)
        .reset_index(drop=True)
    )

    if len(df) < 200:
        raise ValueError(
            f"{symbol}: chỉ có {len(df)} phiên hoàn tất, "
            "cần tối thiểu 200 phiên."
        )

    df.attrs["source"] = (
        "DNSE/Vietcap - dữ liệu lịch sử nến ngày"
    )

    df.attrs["last_date"] = (
        pd.Timestamp(df.iloc[-1]["Date"]).strftime("%d/%m/%Y")
    )

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

        columns = {
            str(c).strip().lower(): c
            for c in result.columns
        }

        if "symbol" not in columns:
            raise RuntimeError(
                "Danh sách trả về không có cột symbol."
            )

        values = result[
            columns["symbol"]
        ].tolist()

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

    # Lấy riêng HOSE và HNX.
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
            "Không lấy được danh sách mã HOSE/HNX: "
            f"{exc}"
        ) from exc

    hose_symbols = _extract_symbols(hose_raw)
    hnx_symbols = _extract_symbols(hnx_raw)

    if not hose_symbols:
        raise RuntimeError(
            "Danh sách HOSE trả về rỗng."
        )

    if not hnx_symbols:
        raise RuntimeError(
            "Danh sách HNX trả về rỗng."
        )

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
