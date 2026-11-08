
import re
import pandas as pd
from vnstock import Market, Listing


def _validate_symbol(symbol):
    symbol = str(symbol).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError("Mã cổ phiếu không hợp lệ.")
    return symbol


def fetch_stock_history(symbol, length=260):
    symbol = _validate_symbol(symbol)
    length = int(length)

    if not 220 <= length <= 2000:
        raise ValueError("Số phiên phải từ 220 đến 2.000.")

    errors = []

    for source in ["kbs", "vci"]:
        try:
            df = Market().equity(symbol).ohlcv(
                length=length,
                interval="1D",
                source=source
            )

            if df is None or df.empty:
                raise ValueError("Không có dữ liệu.")

            df.columns = [str(c).strip().lower() for c in df.columns]
            df = df.rename(columns={
                "time": "date",
                "tradingdate": "date",
                "trading_date": "date"
            })

            required = ["date", "open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                raise ValueError("Thiếu cột dữ liệu OHLCV.")

            df = df[required].copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            for col in required[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna().sort_values("date")
            df = df.drop_duplicates(subset=["date"], keep="last")

            if len(df) < 220:
                raise ValueError("Không đủ 220 phiên dữ liệu.")

            price_cols = ["open", "high", "low", "close"]

            invalid = (
                (df[price_cols] <= 0).any(axis=1)
                | (df["high"] < df["low"])
                | (df["high"] < df["open"])
                | (df["high"] < df["close"])
                | (df["low"] > df["open"])
                | (df["low"] > df["close"])
            )

            if invalid.any():
                raise ValueError("Dữ liệu giá bất thường.")

            # Giá OHLCV cổ phiếu trả về theo nghìn đồng.
            df[price_cols] = df[price_cols] * 1000

            df = df.rename(columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })

            result = df.reset_index(drop=True)
            result.attrs["source"] = source.upper()
            return result

        except Exception as error:
            errors.append(f"{source.upper()}: {error}")

    raise RuntimeError(
        f"Không thể lấy lịch sử {symbol}: " + " | ".join(errors)
    )


def fetch_stock_quote(symbol):
    symbol = _validate_symbol(symbol)
    errors = []

    for source in ["kbs", "vci"]:
        try:
            df = Market().equity(symbol).quote(source=source)

            if df is None or df.empty:
                raise ValueError("Không có bảng giá.")

            df.columns = [str(c).strip().lower() for c in df.columns]
            row = df.iloc[0].to_dict()

            if "close_price" not in row:
                raise ValueError("Thiếu close_price.")

            price = pd.to_numeric(row["close_price"], errors="coerce")
            if pd.isna(price) or price <= 0:
                raise ValueError("Giá không hợp lệ.")

            result = {
                "symbol": symbol,
                "exchange": row.get("exchange"),
                "last_price": float(price),
                "reference_price": row.get("reference_price"),
                "ceiling_price": row.get("ceiling_price"),
                "floor_price": row.get("floor_price"),
                "open_price": row.get("open_price"),
                "high_price": row.get("high_price"),
                "low_price": row.get("low_price"),
                "bid_price_1": row.get("bid_price_1"),
                "ask_price_1": row.get("ask_price_1"),
                "bid_vol_1": row.get("bid_vol_1"),
                "ask_vol_1": row.get("ask_vol_1"),
                "source": source.upper()
            }

            for field in [
                "reference_price", "ceiling_price", "floor_price",
                "open_price", "high_price", "low_price",
                "bid_price_1", "ask_price_1",
                "bid_vol_1", "ask_vol_1"
            ]:
                value = pd.to_numeric(result[field], errors="coerce")
                result[field] = None if pd.isna(value) else float(value)

            return result

        except Exception as error:
            errors.append(f"{source.upper()}: {error}")

    raise RuntimeError(
        f"Không lấy được bảng giá {symbol}: " + " | ".join(errors)
    )


def fetch_exchange_symbols():
    """
    Lấy danh sách mã cổ phiếu HOSE và HNX.
    Không đưa UPCoM vào danh sách.
    """

    errors = []

    for exchange in ["HOSE", "HNX"]:
        try:
            df = Listing().symbols_by_exchange()
            if df is None or df.empty:
                raise ValueError("Không có danh sách mã.")

            df.columns = [str(c).strip().lower() for c in df.columns]

            symbol_col = next(
                (c for c in ["symbol", "ticker", "stock_code"] if c in df.columns),
                None
            )

            exchange_col = next(
                (c for c in ["exchange", "com_group_code", "market"] if c in df.columns),
                None
            )

            if symbol_col is None or exchange_col is None:
                raise ValueError("Không xác định được cột mã hoặc sàn.")

            selected = df[
                df[exchange_col].astype(str).str.upper().str.strip() == exchange
            ]

            errors.extend([])

            if selected.empty:
                continue

            symbols = selected[symbol_col].astype(str).str.upper().str.strip()
            errors.append(("DATA", exchange, symbols.tolist()))

        except Exception as error:
            errors.append(("ERROR", exchange, str(error)))

    valid = [
        item[2] for item in errors
        if isinstance(item, tuple) and item[0] == "DATA"
    ]

    failures = [
        f"{item[1]}: {item[2]}" for item in errors
        if isinstance(item, tuple) and item[0] == "ERROR"
    ]

    symbols = sorted(set(
        symbol
        for group in valid
        for symbol in group
        if re.fullmatch(r"[A-Z0-9]{3,10}", symbol)
    ))

    if not symbols:
        raise RuntimeError(
            "Không lấy được danh sách HOSE/HNX. " + " | ".join(failures)
        )

    return symbols


def fetch_market_quotes(symbols, source="kbs"):
    """
    Lấy bảng giá nhiều mã nếu nguồn dữ liệu hỗ trợ.
    Không tự động chia nhỏ thành hàng trăm lượt gọi.
    """

    symbols = sorted(set(
        _validate_symbol(symbol) for symbol in symbols
    ))

    if not symbols:
        return pd.DataFrame()

    errors = []

    for provider in [source, "vci"]:
        try:
            df = Market().quote(
                symbols=symbols,
                source=provider
            )

            if df is None or df.empty:
                raise ValueError("Nguồn không trả về bảng giá.")

            df.columns = [str(c).strip().lower() for c in df.columns]

            if "symbol" not in df.columns:
                raise ValueError("Bảng giá thiếu cột symbol.")

            df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
            df = df.drop_duplicates(subset=["symbol"], keep="last")
            df.attrs["source"] = provider.upper()

            return df

        except Exception as error:
            errors.append(f"{provider.upper()}: {error}")

    raise RuntimeError(
        "Không lấy được bảng giá nhiều mã: " + " | ".join(errors)
    )
