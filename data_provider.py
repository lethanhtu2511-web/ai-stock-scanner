import re
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd

from vn_market_data import get_ohlcv
from vnstock import Listing, Market


HISTORY_CACHE_SECONDS = 900
HOURLY_CACHE_SECONDS = 300
SYMBOL_CACHE_SECONDS = 21600

_HISTORY_CACHE = {}
_HOURLY_CACHE = {}
_SYMBOL_CACHE = None
_SYMBOL_CACHE_AT = 0.0

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

DAILY_CANDLE_READY_TIME = dt_time(15, 30)

# Chỉ dùng các nguồn được hỗ trợ ở cấp Cộng đồng.
# KBS thử trước; nếu lỗi thì mới thử VCI.
HOURLY_SOURCES = ("kbs", "vci")


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

    df.columns = [str(c).strip().lower() for c in df.columns]

    df = df.rename(columns={
        "time": "date",
        "trading_date": "date",
        "tradingdate": "date",
        "vol": "volume"
    })

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Thiếu cột OHLCV: {missing}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
    if df.empty:
        return df

    now = datetime.now(VN_TZ)
    today = pd.Timestamp(now.date())

    if now.time() < DAILY_CANDLE_READY_TIME:
        dates = pd.to_datetime(
            df["Date"], errors="coerce"
        ).dt.normalize()

        df = df.loc[dates < today].copy()

    return df.reset_index(drop=True)


def fetch_stock_history(symbol, length=260):
    symbol = normalize_symbol(symbol)
    length = max(220, int(length))

    key = (symbol, length)
    now = time.monotonic()
    cached = _HISTORY_CACHE.get(key)

    if cached and now - cached[0] < HISTORY_CACHE_SECONDS:
        return cached[1].copy()

    try:
        rows = get_ohlcv(symbol, lookback_days=600)
    except Exception as exc:
        raise RuntimeError(
            f"{symbol}: không lấy được dữ liệu nến ngày: {exc}"
        ) from exc

    df = _normalize_history(rows)
    df = _remove_incomplete_today_candle(df)
    df = df.tail(length).reset_index(drop=True)

    if len(df) < 200:
        raise ValueError(
            f"{symbol}: chỉ có {len(df)} phiên hoàn tất, "
            "cần tối thiểu 200 phiên."
        )

    df.attrs["source"] = "DNSE/Vietcap - dữ liệu lịch sử nến ngày"
    df.attrs["last_date"] = pd.Timestamp(
        df.iloc[-1]["Date"]
    ).strftime("%d/%m/%Y")

    _HISTORY_CACHE[key] = (time.monotonic(), df.copy())

    if len(_HISTORY_CACHE) > 500:
        oldest = min(
            _HISTORY_CACHE,
            key=lambda k: _HISTORY_CACHE[k][0]
        )
        del _HISTORY_CACHE[oldest]

    return df


def _normalize_hourly(raw, symbol, source):
    if raw is None:
        raise ValueError("Nguồn không trả về dữ liệu.")

    df = pd.DataFrame(raw).copy()

    if df.empty:
        raise ValueError("Nguồn không trả về nến H1.")

    df.columns = [str(c).strip().lower() for c in df.columns]

    df = df.rename(columns={
        "time": "date",
        "trading_date": "date",
        "vol": "volume"
    })

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Thiếu cột H1: {missing}")

    df = df[required].copy()

    # Chuẩn hóa thời gian về múi giờ Việt Nam.
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")

    if parsed_dates.dt.tz is None:
        parsed_dates = parsed_dates.dt.tz_localize(
            "Asia/Ho_Chi_Minh"
        )
    else:
        parsed_dates = parsed_dates.dt.tz_convert(
            "Asia/Ho_Chi_Minh"
        )

    df["date"] = parsed_dates

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    # OHLCV cổ phiếu Vnstock trả giá theo nghìn đồng.
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * 1000

    # Loại nến H1 chưa hoàn tất.
    # Giữ các nến đã hoàn tất trong chính phiên hôm nay.
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    current_hour = now.floor("h")

    df = df.loc[df["date"] < current_hour].copy()

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

    df = df.loc[valid].copy()

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            f"Chỉ có {len(df)} nến H1 hoàn tất, cần tối thiểu 20."
        )

    # Kiểm tra đơn vị và mức giá so với D1.
    daily = fetch_stock_history(symbol, length=220)

    daily_close = float(daily.iloc[-1]["Close"])
    hourly_close = float(df.iloc[-1]["Close"])

    if daily_close <= 0 or hourly_close <= 0:
        raise ValueError("Giá H1 hoặc D1 không hợp lệ.")

    difference = abs(hourly_close / daily_close - 1)

    if difference > 0.15:
        raise ValueError(
            "Giá H1 lệch quá 15% so với D1; "
            "không sử dụng dữ liệu này."
        )

    df.attrs["source"] = source.upper()
    df.attrs["last_date"] = pd.Timestamp(
        df.iloc[-1]["Date"]
    ).strftime("%d/%m/%Y %H:%M")

    return df


def fetch_stock_hourly(symbol, lookback="14d"):
    """
    Lấy tối thiểu 20 nến H1 hoàn tất.

    Thử KBS trước, VCI sau.
    Không thử MAS/VND vì các nguồn đó không thuộc
    nhóm nguồn được phép ở cấp Cộng đồng.
    """

    symbol = normalize_symbol(symbol)

    key = (symbol, str(lookback))
    now_mono = time.monotonic()
    cached = _HOURLY_CACHE.get(key)

    if cached and now_mono - cached[0] < HOURLY_CACHE_SECONDS:
        return cached[1].copy()

    errors = []

    for source in HOURLY_SOURCES:
        try:
            raw = Market().equity(symbol).ohlcv(
                length=lookback,
                interval="1H",
                source=source
            )

            df = _normalize_hourly(
                raw=raw,
                symbol=symbol,
                source=source
            )

            _HOURLY_CACHE[key] = (
                time.monotonic(),
                df.copy()
            )

            if len(_HOURLY_CACHE) > 500:
                oldest = min(
                    _HOURLY_CACHE,
                    key=lambda k: _HOURLY_CACHE[k][0]
                )
                del _HOURLY_CACHE[oldest]

            return df

        except Exception as exc:
            errors.append(
                f"{source.upper()}: {str(exc)}"
            )

    raise RuntimeError(
        f"{symbol}: không lấy được dữ liệu H1 từ các nguồn dự phòng. "
        + " | ".join(errors)
    )


def fetch_stock_quote(symbol):
    symbol = normalize_symbol(symbol)
    df = fetch_stock_history(symbol, length=260)
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

        values = result[columns["symbol"]].tolist()

    elif isinstance(result, pd.Series):
        values = result.tolist()

    else:
        values = list(result)

    symbols = []

    for value in values:
        try:
            symbols.append(normalize_symbol(value))
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
        hose_raw = listing.symbols_by_group(group="HOSE")
        hnx_raw = listing.symbols_by_group(group="HNX")
    except Exception as exc:
        raise RuntimeError(
            "Không lấy được danh sách mã HOSE/HNX: "
            f"{exc}"
        ) from exc

    hose_symbols = _extract_symbols(hose_raw)
    hnx_symbols = _extract_symbols(hnx_raw)

    if not hose_symbols:
        raise RuntimeError("Danh sách HOSE trả về rỗng.")

    if not hnx_symbols:
        raise RuntimeError("Danh sách HNX trả về rỗng.")

    symbols = sorted(set(hose_symbols + hnx_symbols))

    if not symbols:
        raise RuntimeError(
            "Không tìm thấy mã cổ phiếu HOSE/HNX."
        )

    _SYMBOL_CACHE = symbols
    _SYMBOL_CACHE_AT = time.monotonic()

    return list(_SYMBOL_CACHE)
