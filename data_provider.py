import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from vn_market_data import get_ohlcv
from vnstock import Listing, Market, Reference


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


def fetch_stock_history(symbol, length=260):
    symbol = normalize_symbol(symbol)
    length = max(220, int(length))

    key = (symbol, length)
    now = time.monotonic()

    cached = _HISTORY_CACHE.get(key)

    if cached and now - cached[0] < HISTORY_CACHE_SECONDS:
        return cached[1].copy()

    rows = get_ohlcv(symbol, lookback_days=600)

    df = (
        _normalize_history(rows)
        .tail(length)
        .reset_index(drop=True)
    )

    if len(df) < 200:
        raise ValueError(
            f"{symbol}: chỉ có {len(df)} phiên, cần tối thiểu 200 phiên."
        )

    df.attrs["source"] = "DNSE/Vietcap (lịch sử nến ngày)"

    _HISTORY_CACHE[key] = (time.monotonic(), df.copy())

    if len(_HISTORY_CACHE) > 500:
        oldest = min(
            _HISTORY_CACHE,
            key=lambda k: _HISTORY_CACHE[k][0]
        )
        del _HISTORY_CACHE[oldest]

    return df


def fetch_stock_quote(symbol):
    symbol = normalize_symbol(symbol)
    df = fetch_stock_history(symbol, length=260)
    row = df.iloc[-1]

    return {
        "symbol": symbol,
        "last_price": float(row["Close"]),
        "quote_date": pd.Timestamp(row["Date"]).strftime("%d/%m/%Y"),
        "source": "Giá đóng cửa nến ngày"
    }


def get_market_status():
    """
    Đọc trạng thái phiên từ lịch giao dịch tích hợp trong VNStock.
    Không tự suy đoán chỉ dựa vào giờ trên điện thoại.
    """
    now = datetime.now(VN_TZ)

    try:
        status_df = Reference().market.status()

        if status_df is None or status_df.empty:
            raise RuntimeError("VNStock không trả về trạng thái thị trường.")

        status_df = status_df.copy()
        status_df.columns = [
            str(c).strip().lower()
            for c in status_df.columns
        ]

        if "exchange" not in status_df.columns or "status" not in status_df.columns:
            raise RuntimeError("Thiếu cột exchange/status trong trạng thái thị trường.")

        status_df["exchange"] = (
            status_df["exchange"].astype(str).str.upper().str.strip()
        )

        status_df["status"] = (
            status_df["status"].astype(str).str.upper().str.strip()
        )

        relevant = status_df[
            status_df["exchange"].isin(["HOSE", "HNX"])
        ].copy()

        if relevant.empty:
            raise RuntimeError("Không có trạng thái HOSE/HNX.")

        status_values = set(relevant["status"].tolist())

        is_open = any(
            ("OPEN" in value)
            or ("TRADING" in value)
            for value in status_values
        )

        return {
            "is_open": is_open,
            "status": ", ".join(
                f"{row.exchange}: {row.status}"
                for row in relevant.itertuples()
            ),
            "checked_at": now.strftime("%H:%M:%S %d/%m/%Y"),
            "source": "VNStock Reference().market.status()"
        }

    except Exception as exc:
        raise RuntimeError(
            "Không xác định được trạng thái phiên HOSE/HNX. "
            "Không tự chuyển sang chế độ realtime. Chi tiết: "
            f"{exc}"
        ) from exc


def fetch_market_quotes(symbols):
    """
    Lấy bảng giá nhiều mã bằng một truy vấn Market().quote().
    close_price là giá khớp gần nhất được nguồn công bố.
    """
    clean_symbols = sorted({
        normalize_symbol(s)
        for s in symbols
    })

    if not clean_symbols:
        raise ValueError("Danh sách mã cần lấy bảng giá đang rỗng.")

    try:
        raw = Market().quote(clean_symbols, source="kbs")

    except Exception as exc:
        raise RuntimeError(
            "Không lấy được bảng giá trực tiếp từ VNStock KBS: "
            f"{exc}"
        ) from exc

    if raw is None:
        raise RuntimeError("VNStock không trả về bảng giá.")

    quotes_df = pd.DataFrame(raw)

    if quotes_df.empty:
        raise RuntimeError("Bảng giá VNStock trả về rỗng.")

    quotes_df.columns = [
        str(c).strip().lower()
        for c in quotes_df.columns
    ]

    required = ["symbol", "close_price"]

    missing = [c for c in required if c not in quotes_df.columns]

    if missing:
        raise RuntimeError(
            f"Bảng giá thiếu cột bắt buộc: {missing}"
        )

    quotes_df["symbol"] = (
        quotes_df["symbol"].astype(str).str.upper().str.strip()
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

    if "total_volume" in quotes_df.columns:
        quotes_df["total_volume"] = pd.to_numeric(
            quotes_df["total_volume"],
            errors="coerce"
        )

    if "match_vol" in quotes_df.columns:
        quotes_df["match_vol"] = pd.to_numeric(
            quotes_df["match_vol"],
            errors="coerce"
        )

    quotes_df = quotes_df.dropna(subset=["close_price"])
    quotes_df = quotes_df[quotes_df["close_price"] > 0]

    quotes = {}

    for row in quotes_df.to_dict("records"):
        symbol = row["symbol"]

        if symbol not in clean_symbols:
            continue

        quotes[symbol] = {
            "symbol": symbol,
            "last_price": float(row["close_price"]),
            "reference_price": (
                float(row["reference_price"])
                if pd.notna(row.get("reference_price"))
                else None
            ),
            "total_volume": (
                float(row["total_volume"])
                if pd.notna(row.get("total_volume"))
                else None
            ),
            "match_vol": (
                float(row["match_vol"])
                if pd.notna(row.get("match_vol"))
                else None
            ),
            "exchange": row.get("exchange", ""),
            "source": "VNStock KBS - bảng giá nhiều mã"
        }

    if not quotes:
        raise RuntimeError("Không có mã nào nhận được giá hợp lệ.")

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
            "Không lấy được danh sách mã HOSE/HNX qua VNStock. "
            f"Chi tiết: {exc}"
        ) from exc

    hose_symbols = _extract_symbols(hose_raw)
    hnx_symbols = _extract_symbols(hnx_raw)

    if not hose_symbols:
        raise RuntimeError("Danh sách HOSE trả về rỗng.")

    if not hnx_symbols:
        raise RuntimeError("Danh sách HNX trả về rỗng.")

    symbols = sorted(set(hose_symbols + hnx_symbols))

    if not symbols:
        raise RuntimeError("Không tìm thấy mã cổ phiếu HOSE/HNX.")

    _SYMBOL_CACHE = symbols
    _SYMBOL_CACHE_AT = time.monotonic()

    return list(_SYMBOL_CACHE)
