
import re
import time
import pandas as pd
from vnstock import Market


# Giới hạn tốc độ gọi API
_API_INTERVAL_SECONDS = 7.5
_LAST_API_CALL = 0.0


def wait_api_slot():
    global _LAST_API_CALL

    now = time.monotonic()
    remaining = _API_INTERVAL_SECONDS - (now - _LAST_API_CALL)

    if remaining > 0:
        time.sleep(remaining)

    _LAST_API_CALL = time.monotonic()


def normalize_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    return symbol


def normalize_history(raw):
    if raw is None or raw.empty:
        raise ValueError("Không có dữ liệu lịch sử.")

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    df = df.rename(columns={
        "time": "date",
        "tradingdate": "date",
        "trading_date": "date",
        "vol": "volume"
    })

    required = [
        "date", "open", "high",
        "low", "close", "volume"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Thiếu cột dữ liệu: {missing}")

    df = df[required].copy()
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    for col in required[1:]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna().sort_values("date")
    df = df.drop_duplicates("date", keep="last")

    # Vnstock thường trả giá OHLC theo nghìn đồng.
    if not df.empty and df["close"].median() < 1000:
        df[["open", "high", "low", "close"]] *= 1000

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

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    return df.reset_index(drop=True)


def fetch_stock_history(symbol, length=260):
    symbol = normalize_symbol(symbol)
    errors = []

    for source in ["kbs", "vci"]:
        try:
            wait_api_slot()

            raw = Market().equity(symbol).ohlcv(
                length=int(length),
                interval="1D",
                source=source
            )

            df = normalize_history(raw)

            if len(df) < 200:
                raise ValueError(
                    f"Chỉ có {len(df)} phiên dữ liệu."
                )

            df.attrs["source"] = source.upper()
            return df

        except Exception as exc:
            errors.append(f"{source}: {exc}")

    raise RuntimeError(
        f"Không lấy được lịch sử {symbol}: "
        + " | ".join(errors)
    )


def fetch_stock_quote(symbol):
    symbol = normalize_symbol(symbol)
    errors = []

    for source in ["kbs", "vci"]:
        try:
            wait_api_slot()

            raw = Market().equity(symbol).quote(
                source=source
            )

            if raw is None or raw.empty:
                raise ValueError("Bảng giá rỗng.")

            row = raw.iloc[0].to_dict()

            row = {
                str(k).strip().lower(): v
                for k, v in row.items()
            }

            def number(*keys):
                for key in keys:
                    value = pd.to_numeric(
                        row.get(key),
                        errors="coerce"
                    )

                    if pd.notna(value):
                        return float(value)

                return None

            last = number(
                "close_price",
                "match_price",
                "last_price",
                "price"
            )

            if last is None:
                raise ValueError(
                    "Không tìm thấy giá gần nhất."
                )

            return {
                "symbol": symbol,
                "last_price": last,
                "reference_price": number(
                    "reference_price", "ref_price"
                ),
                "open_price": number(
                    "open_price", "open"
                ),
                "ceiling_price": number(
                    "ceiling_price", "ceiling"
                ),
                "floor_price": number(
                    "floor_price", "floor"
                ),
                "exchange": row.get(
                    "exchange",
                    row.get("com_group_code", "")
                ),
                "bid_price_1": number(
                    "bid_price_1", "bid1"
                ),
                "ask_price_1": number(
                    "ask_price_1", "ask1"
                ),
                "bid_vol_1": number(
                    "bid_vol_1", "bid_vol1"
                ),
                "ask_vol_1": number(
                    "ask_vol_1", "ask_vol1"
                ),
                "source": source.upper()
            }

        except Exception as exc:
            errors.append(f"{source}: {exc}")

    raise RuntimeError(
        f"Không lấy được bảng giá {symbol}: "
        + " | ".join(errors)
    )


def fetch_exchange_symbols():
    # Danh sách chọn lọc HOSE và HNX.
    # Không gọi API Listing bị giới hạn quyền tài khoản Khách.

    hose_symbols = [
        "FPT", "HPG", "SSI", "VCB", "MBB",
        "TCB", "ACB", "CTG", "BID", "VPB",
        "VHM", "VIC", "VNM", "MWG", "GEX",
        "HDB", "STB", "TPB", "LPB", "SHB",
        "VND", "HCM", "VRE", "PNJ", "GMD",
        "DGC", "DPM", "PDR", "DIG", "DXG",
        "NKG", "HSG", "KBC", "KDH", "NVL",
        "POW", "PLX", "VJC", "MSN", "EIB"
    ]

    hnx_symbols = [
        "PVS", "MBS", "SHS", "CEO", "IDC",
        "TNG", "VCS", "LAS", "PLC", "PVC"
    ]

    return hose_symbols + hnx_symbols
