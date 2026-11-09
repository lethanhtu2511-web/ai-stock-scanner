import re
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd

from vn_market_data import get_ohlcv
from vnstock import Listing, Market


HISTORY_CACHE_SECONDS = 900
SYMBOL_CACHE_SECONDS = 21600

_HISTORY_CACHE = {}
_SYMBOL_CACHE = None
_SYMBOL_CACHE_AT = 0.0

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


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
        "date", "open", "high",
        "low", "close", "volume"
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
            f"{symbol}: chỉ có {len(df)} phiên, "
            "cần tối thiểu 200 phiên."
        )

    df.attrs["source"] = "DNSE/Vietcap - lịch sử nến ngày"

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


def get_market_status():
    """
    Nhận diện khung giờ giao dịch theo giờ Việt Nam.

    Lưu ý:
    - Không gọi Reference().market.status() vì tài khoản Khách
      không có quyền sử dụng chức năng đó.
    - Không tự nhận diện ngày nghỉ lễ.
    - Bảng giá thực tế vẫn phải được truy vấn thành công
      trước khi bật chế độ giá trong phiên.
    """

    now = datetime.now(VN_TZ)
    current_time = now.time()

    is_weekday = now.weekday() < 5

    morning_session = (
        dt_time(9, 0) <= current_time <= dt_time(11, 30)
    )

    afternoon_session = (
        dt_time(13, 0) <= current_time <= dt_time(15, 0)
    )

    in_trading_hours = (
        is_weekday
        and (morning_session or afternoon_session)
    )

    if not is_weekday:
        status = "Ngoài phiên - cuối tuần"

    elif current_time < dt_time(9, 0):
        status = "Ngoài phiên - chưa mở cửa"

    elif dt_time(11, 30) < current_time < dt_time(13, 0):
        status = "Ngoài phiên - nghỉ trưa"

    elif current_time > dt_time(15, 0):
        status = "Ngoài phiên - đã đóng cửa"

    elif in_trading_hours:
        status = "Trong khung giờ giao dịch dự kiến"

    else:
        status = "Ngoài phiên"

    return {
        "is_open": in_trading_hours,
        "status": status,
        "checked_at": now.strftime("%H:%M:%S %d/%m/%Y"),
        "source": "Đồng hồ Việt Nam - khung giờ giao dịch dự kiến",
        "holiday_check": False
    }


def fetch_market_quotes(symbols):
    """
    Lấy bảng giá nhiều mã bằng Market().quote().
    close_price là giá khớp gần nhất do nguồn bảng giá trả về.
    """

    clean_symbols = sorted({
        normalize_symbol(s)
        for s in symbols
    })

    if not clean_symbols:
        raise ValueError(
            "Danh sách mã cần lấy bảng giá đang rỗng."
        )

    try:
        raw = Market().quote(
            clean_symbols,
            source="kbs"
        )

    except Exception as exc:
        raise RuntimeError(
            "Không lấy được bảng giá VNStock KBS: "
            f"{exc}"
        ) from exc

    if raw is None:
        raise RuntimeError(
            "VNStock không trả về bảng giá."
        )

    quotes_df = pd.DataFrame(raw)

    if quotes_df.empty:
        raise RuntimeError(
            "Bảng giá VNStock trả về rỗng."
        )

    quotes_df.columns = [
        str(c).strip().lower()
        for c in quotes_df.columns
    ]

    required = [
        "symbol",
        "close_price"
    ]

    missing = [
        c for c in required
        if c not in quotes_df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Bảng giá thiếu cột bắt buộc: {missing}"
        )

    quotes_df["symbol"] = (
        quotes_df["symbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    quotes_df["close_price"] = pd.to_numeric(
        quotes_df["close_price"],
        errors="coerce"
    )

    if "reference_price" in quotes_df.columns:
        quotes_df["reference_price"] = pd.to_numeric(
            quotes_df["reference_price"],
            errors="coerce"
        )

    quotes_df = quotes_df.dropna(
        subset=["close_price"]
    )

    quotes_df = quotes_df[
        quotes_df["close_price"] > 0
    ]

    quotes = {}

    for row in quotes_df.to_dict("records"):
        symbol = row["symbol"]

        if symbol not in clean_symbols:
            continue

        reference_price = row.get("reference_price")

        quotes[symbol] = {
            "symbol": symbol,
            "last_price": float(row["close_price"]),
            "reference_price": (
                float(reference_price)
                if pd.notna(reference_price)
                else None
            ),
            "exchange": row.get("exchange", ""),
            "source": "VNStock KBS - bảng giá nhiều mã"
        }

    if not quotes:
        raise RuntimeError(
            "Không có mã nào nhận được giá hợp lệ."
        )

    return quotes


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
