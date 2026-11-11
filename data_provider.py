
import re
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from vn_market_data import get_ohlcv


# =========================================================
# CẤU HÌNH
# =========================================================

HISTORY_CACHE_SECONDS = 900
SYMBOL_CACHE_SECONDS = 21600
HISTORY_CACHE_MAX_ITEMS = 500

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DAILY_CANDLE_READY_TIME = dt_time(15, 30)

VIETCAP_LISTING_URL = (
    "https://trading.vietcap.com.vn/api/price/symbols/getAll"
)

VIETCAP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://trading.vietcap.com.vn/",
    "Origin": "https://trading.vietcap.com.vn",
}

# =========================================================
# BỘ NHỚ TẠM
# =========================================================

_HISTORY_CACHE = {}

_SYMBOL_CACHE = None
_SYMBOL_CACHE_AT = 0.0


# =========================================================
# NGOẠI LỆ
# =========================================================

class InsufficientHistoryError(RuntimeError):
    pass


class SourceRateLimitError(RuntimeError):
    pass


# =========================================================
# KIỂM TRA MÃ
# =========================================================

def normalize_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError(
            f"Mã cổ phiếu không hợp lệ: {symbol}"
        )

    return symbol


# =========================================================
# CHUẨN HÓA OHLCV
# =========================================================

def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(
                str(x)
                for x in col
                if str(x) != ""
            ).strip("_")
            for col in df.columns
        ]

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    return df


def _standardize_ohlcv(raw, label="OHLCV"):
    if raw is None:
        raise ValueError(
            f"Nguồn không trả dữ liệu {label}."
        )

    df = pd.DataFrame(raw).copy()

    if df.empty:
        raise ValueError(
            f"Nguồn không trả dữ liệu {label}."
        )

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

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} thiếu cột: {missing}"
        )

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


# =========================================================
# DỮ LIỆU D1
# =========================================================

def _normalize_history(rows):
    df = _standardize_ohlcv(
        rows,
        "nến ngày",
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna()
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
    )

    df = _valid_ohlcv(df)

    return (
        df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        .reset_index(drop=True)
    )


def _remove_incomplete_today_candle(df):
    if df.empty:
        return df

    now = datetime.now(VN_TZ)

    if now.time() < DAILY_CANDLE_READY_TIME:
        dates = pd.to_datetime(
            df["Date"],
            errors="coerce",
        ).dt.normalize()

        today = pd.Timestamp(now.date())

        df = df.loc[
            dates < today
        ].copy()

    return df.reset_index(drop=True)


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

    try:
        raw = get_ohlcv(
            symbol,
            lookback_days=600,
        )

        df = _normalize_history(raw)

        df = _remove_incomplete_today_candle(df)

        df = (
            df.tail(length)
            .reset_index(drop=True)
        )

        if len(df) < 200:
            raise InsufficientHistoryError(
                f"Chỉ có {len(df)} phiên hoạt động; "
                "cần tối thiểu 200."
            )

        df.attrs["source"] = "vn-market-data"

        df.attrs["last_date"] = pd.Timestamp(
            df.iloc[-1]["Date"]
        ).strftime("%d/%m/%Y")

        _HISTORY_CACHE[key] = (
            time.monotonic(),
            df.copy(),
        )

        if len(_HISTORY_CACHE) > HISTORY_CACHE_MAX_ITEMS:
            oldest = min(
                _HISTORY_CACHE,
                key=lambda item: _HISTORY_CACHE[item][0],
            )

            del _HISTORY_CACHE[oldest]

        return df

    except InsufficientHistoryError:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"{symbol}: không lấy được dữ liệu D1: {exc}"
        ) from exc


# =========================================================
# H1 ĐÃ TẮT
# =========================================================

def fetch_stock_hourly(symbol, lookback="14d"):
    raise RuntimeError(
        "Phân tích H1 đã tạm tắt. "
        "Bot hiện sử dụng dữ liệu D1."
    )


def get_hourly_source_status():
    return {
        "KBS": {
            "blocked": True,
            "remaining_seconds": 0,
            "reason": "Đã tắt H1.",
            "consecutive_rate_limit_hits": 0,
        },
        "VCI": {
            "blocked": True,
            "remaining_seconds": 0,
            "reason": "Đã tắt H1.",
            "consecutive_rate_limit_hits": 0,
        },
    }


def reset_hourly_source_limits():
    return None


# =========================================================
# GIÁ ĐÓNG CỬA GẦN NHẤT
# =========================================================

def fetch_stock_quote(symbol):
    symbol = normalize_symbol(symbol)

    df = fetch_stock_history(
        symbol,
        length=260,
    )

    row = df.iloc[-1]

    return {
        "symbol": symbol,
        "last_price": float(row["Close"]),
        "quote_date": pd.Timestamp(
            row["Date"]
        ).strftime("%d/%m/%Y"),
        "source": "Giá đóng cửa nến ngày",
    }


# =========================================================
# DANH SÁCH MÃ HOSE / HNX
# =========================================================

def _extract_symbols(rows, exchange):
    symbols = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        board = str(
            row.get("board", "")
        ).strip().upper()

        instrument_type = str(
            row.get("type", "")
        ).strip().upper()

        symbol = str(
            row.get("symbol", "")
        ).strip().upper()

        if exchange == "HOSE":
            valid_board = board == "HSX"
        else:
            valid_board = board == "HNX"

        if not valid_board:
            continue

        if instrument_type != "STOCK":
            continue

        try:
            symbols.append(
                normalize_symbol(symbol)
            )
        except ValueError:
            continue

    return sorted(set(symbols))


def _fetch_vietcap_listing():
    errors = []

    for attempt in range(3):
        try:
            with httpx.Client(
                headers=VIETCAP_HEADERS,
                timeout=15.0,
                follow_redirects=True,
            ) as client:

                response = client.get(
                    VIETCAP_LISTING_URL
                )

                response.raise_for_status()

                data = response.json()

                if not isinstance(data, list):
                    raise ValueError(
                        "Dữ liệu danh sách mã không đúng định dạng."
                    )

                if not data:
                    raise ValueError(
                        "Danh sách mã trả về rỗng."
                    )

                return data

        except Exception as exc:
            errors.append(str(exc))

            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(
        "Không lấy được danh sách mã từ Vietcap: "
        + " | ".join(errors)
    )


def fetch_exchange_symbols(force_refresh=False):
    global _SYMBOL_CACHE
    global _SYMBOL_CACHE_AT

    now = time.monotonic()

    if (
        not force_refresh
        and _SYMBOL_CACHE is not None
        and now - _SYMBOL_CACHE_AT < SYMBOL_CACHE_SECONDS
    ):
        return list(_SYMBOL_CACHE)

    rows = _fetch_vietcap_listing()

    hose = _extract_symbols(
        rows,
        "HOSE",
    )

    hnx = _extract_symbols(
        rows,
        "HNX",
    )

    if not hose:
        raise RuntimeError(
            "Danh sách HOSE rỗng."
        )

    if not hnx:
        raise RuntimeError(
            "Danh sách HNX rỗng."
        )

    symbols = sorted(
        set(hose + hnx)
    )

    if not symbols:
        raise RuntimeError(
            "Không có mã cổ phiếu hợp lệ."
        )

    _SYMBOL_CACHE = symbols
    _SYMBOL_CACHE_AT = time.monotonic()

    fetch_exchange_symbols.last_stats = {
        "hose_count": len(hose),
        "hnx_count": len(hnx),
        "unique_count": len(symbols),
        "duplicate_count": (
            len(hose) + len(hnx) - len(symbols)
        ),
        "source": "Vietcap public listing",
    }

    return list(symbols)


fetch_exchange_symbols.last_stats = {}
