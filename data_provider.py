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
HOURLY_SOURCES = ("kbs", "vci")


def normalize_symbol(symbol):
    symbol = str(symbol).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError(f"Mã cổ phiếu không hợp lệ: {symbol}")
    return symbol


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if str(x) != "").strip("_")
            for col in df.columns
        ]
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _standardize_ohlcv(raw, label="OHLCV"):
    if raw is None:
        raise ValueError(f"Nguồn không trả dữ liệu {label}.")

    df = pd.DataFrame(raw).copy()
    if df.empty:
        raise ValueError(f"Nguồn không trả dữ liệu {label}.")

    df = _flatten_columns(df)

    rename = {
        "time": "date",
        "trading_date": "date",
        "tradingdate": "date",
        "timestamp": "date",
        "vol": "volume",
        "total_volume": "volume",
        "closing_price": "close",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
    }
    df = df.rename(columns=rename)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} thiếu cột: {missing}")

    return df[required].copy()


def _valid_ohlcv(df):
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
    return df.loc[valid].copy()


def _normalize_history(rows):
    df = _standardize_ohlcv(rows, "nến ngày")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("date")
    df = df.drop_duplicates("date", keep="last")
    df = _valid_ohlcv(df)

    return df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }).reset_index(drop=True)


def _remove_incomplete_today_candle(df):
    if df.empty:
        return df

    now = datetime.now(VN_TZ)
    if now.time() < DAILY_CANDLE_READY_TIME:
        dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        today = pd.Timestamp(now.date())
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
        raw = get_ohlcv(symbol, lookback_days=600)
        df = _normalize_history(raw)
        df = _remove_incomplete_today_candle(df)
        df = df.tail(length).reset_index(drop=True)

        if len(df) < 200:
            raise ValueError(
                f"Chỉ có {len(df)} phiên hoàn tất; cần tối thiểu 200."
            )

        df.attrs["source"] = "DNSE/Vietcap - lịch sử nến ngày"
        df.attrs["last_date"] = pd.Timestamp(
            df.iloc[-1]["Date"]
        ).strftime("%d/%m/%Y")

        _HISTORY_CACHE[key] = (time.monotonic(), df.copy())

        if len(_HISTORY_CACHE) > 500:
            oldest = min(_HISTORY_CACHE, key=lambda k: _HISTORY_CACHE[k][0])
            del _HISTORY_CACHE[oldest]

        return df

    except Exception as exc:
        raise RuntimeError(
            f"{symbol}: không lấy được dữ liệu D1: {exc}"
        ) from exc


def _parse_vietnam_datetime(series):
    parsed = []

    for value in series:
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                parsed.append(pd.NaT)
            elif ts.tzinfo is None:
                parsed.append(ts.tz_localize(VN_TZ))
            else:
                parsed.append(ts.tz_convert(VN_TZ))
        except Exception:
            parsed.append(pd.NaT)

    return pd.Series(
        pd.to_datetime(parsed, errors="coerce"),
        index=series.index,
    )


def _normalize_hourly(raw, symbol, source):
    df = _standardize_ohlcv(raw, "nến H1")

    df["date"] = _parse_vietnam_datetime(df["date"])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("date")
    df = df.drop_duplicates("date", keep="last")

    # Xác định đơn vị giá dựa trên tương quan với giá D1.
    daily = fetch_stock_history(symbol, length=220)
    daily_close = float(daily.iloc[-1]["Close"])

    raw_hourly_close = float(df.iloc[-1]["close"])
    if daily_close <= 0 or raw_hourly_close <= 0:
        raise ValueError("Giá H1 hoặc D1 không hợp lệ.")

    ratio = raw_hourly_close / daily_close

    if 0.0005 <= ratio <= 0.005:
        multiplier = 1000.0
    elif 0.80 <= ratio <= 1.20:
        multiplier = 1.0
    else:
        raise ValueError(
            "Không xác định được đơn vị giá H1 so với D1 "
            f"(tỷ lệ hiện tại: {ratio:.5f})."
        )

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * multiplier

    # Chỉ giữ nến đã hoàn tất.
    now = pd.Timestamp.now(tz=VN_TZ)
    current_hour = now.floor("h")
    df = df.loc[df["date"] < current_hour].copy()

    df = _valid_ohlcv(df)

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }).reset_index(drop=True)

    if len(df) < 25:
        raise ValueError(
            f"Chỉ có {len(df)} nến H1 hoàn tất; cần tối thiểu 25."
        )

    hourly_close = float(df.iloc[-1]["Close"])
    difference = abs(hourly_close / daily_close - 1)

    if difference > 0.15:
        raise ValueError(
            f"Giá H1 lệch {difference * 100:.1f}% so với D1."
        )

    last_time = pd.Timestamp(df.iloc[-1]["Date"])
    age_hours = (
        pd.Timestamp.now(tz=VN_TZ) - last_time
    ).total_seconds() / 3600

    if age_hours > 72:
        raise ValueError(
            f"Nến H1 gần nhất đã cũ {age_hours:.1f} giờ."
        )

    df.attrs["source"] = source.upper()
    df.attrs["price_multiplier"] = multiplier
    df.attrs["last_date"] = last_time.strftime("%d/%m/%Y %H:%M")

    return df


def fetch_stock_hourly(symbol, lookback="14d"):
    symbol = normalize_symbol(symbol)
    key = (symbol, str(lookback))

    now = time.monotonic()
    cached = _HOURLY_CACHE.get(key)

    if cached and now - cached[0] < HOURLY_CACHE_SECONDS:
        return cached[1].copy()

    errors = []

    for source in HOURLY_SOURCES:
        try:
            raw = Market().equity(symbol).ohlcv(
                length=lookback,
                interval="1H",
                source=source,
            )

            df = _normalize_hourly(raw, symbol, source)

            _HOURLY_CACHE[key] = (time.monotonic(), df.copy())

            if len(_HOURLY_CACHE) > 500:
                oldest = min(
                    _HOURLY_CACHE,
                    key=lambda k: _HOURLY_CACHE[k][0],
                )
                del _HOURLY_CACHE[oldest]

            return df

        except Exception as exc:
            errors.append(f"{source.upper()}: {exc}")

    raise RuntimeError(
        f"{symbol}: không lấy được H1. " + " | ".join(errors)
    )


def fetch_stock_quote(symbol):
    symbol = normalize_symbol(symbol)
    df = fetch_stock_history(symbol, length=260)
    row = df.iloc[-1]

    return {
        "symbol": symbol,
        "last_price": float(row["Close"]),
        "quote_date": pd.Timestamp(row["Date"]).strftime("%d/%m/%Y"),
        "source": "Giá đóng cửa nến ngày",
    }


def _extract_symbols(result):
    if result is None:
        return []

    if isinstance(result, pd.DataFrame):
        columns = {
            str(c).strip().lower(): c for c in result.columns
        }
        if "symbol" not in columns:
            raise RuntimeError("Danh sách không có cột symbol.")
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

    return sorted(set(symbols))


def fetch_exchange_symbols(force_refresh=False):
    global _SYMBOL_CACHE, _SYMBOL_CACHE_AT

    now = time.monotonic()

    if (
        not force_refresh
        and _SYMBOL_CACHE is not None
        and now - _SYMBOL_CACHE_AT < SYMBOL_CACHE_SECONDS
    ):
        return list(_SYMBOL_CACHE)

    listing = Listing(source="KBS")

    try:
        hose_raw = listing.symbols_by_group(group="HOSE")
        hnx_raw = listing.symbols_by_group(group="HNX")
    except Exception as exc:
        raise RuntimeError(
            f"Không lấy được danh sách HOSE/HNX từ KBS: {exc}"
        ) from exc

    hose = _extract_symbols(hose_raw)
    hnx = _extract_symbols(hnx_raw)

    if not hose:
        raise RuntimeError("Danh sách HOSE rỗng.")
    if not hnx:
        raise RuntimeError("Danh sách HNX rỗng.")

    symbols = sorted(set(hose + hnx))

    if not symbols:
        raise RuntimeError("Không có mã cổ phiếu hợp lệ.")

    _SYMBOL_CACHE = symbols
    _SYMBOL_CACHE_AT = time.monotonic()

    # Lưu thông tin thống kê để app hiển thị.
    fetch_exchange_symbols.last_stats = {
        "hose_count": len(hose),
        "hnx_count": len(hnx),
        "unique_count": len(symbols),
        "duplicate_count": len(hose) + len(hnx) - len(symbols),
        "source": "KBS Listing",
    }

    return list(symbols)


fetch_exchange_symbols.last_stats = {}
