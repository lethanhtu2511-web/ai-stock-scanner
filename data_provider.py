import re
import pandas as pd
from vnstock import Market


def _validate_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    return symbol


def fetch_stock_history(symbol, length=260):
    symbol = _validate_symbol(symbol)
    length = int(length)

    if length < 220 or length > 2000:
        raise ValueError("Số phiên phải từ 220 đến 2.000.")

    errors = []

    # Ưu tiên KBS, dự phòng VCI
    for source in ["kbs", "vci"]:
        try:
            df = Market().equity(symbol).ohlcv(
                length=length,
                interval="1D",
                source=source
            )

            if df is None or df.empty:
                raise ValueError("Nguồn không trả về dữ liệu.")

            df.columns = [
                str(col).strip().lower()
                for col in df.columns
            ]

            df = df.rename(columns={
                "time": "date",
                "tradingdate": "date",
                "trading_date": "date"
            })

            required = [
                "date", "open", "high",
                "low", "close", "volume"
            ]

            missing = [
                col for col in required
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    "Thiếu cột: " + ", ".join(missing)
                )

            df = df[required].copy()

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            for col in [
                "open", "high", "low",
                "close", "volume"
            ]:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            df = df.dropna()
            df = df.sort_values("date")
            df = df.drop_duplicates(
                subset=["date"],
                keep="last"
            )

            if len(df) < 220:
                raise ValueError("Không đủ 220 phiên dữ liệu.")

            price_cols = [
                "open", "high", "low", "close"
            ]

            if (df[price_cols] <= 0).any().any():
                raise ValueError("Giá dữ liệu không hợp lệ.")

            invalid = (
                (df["high"] < df["low"]) |
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"]) |
                (df["low"] > df["close"])
            )

            if invalid.any():
                raise ValueError("Dữ liệu OHLC bất thường.")

            # Giá OHLCV cổ phiếu trả về theo nghìn đồng.
            # Chuyển sang đồng.
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
        f"Không thể lấy dữ liệu lịch sử {symbol}. "
        + " | ".join(errors)
    )


def fetch_stock_quote(symbol):
    """
    Lấy bảng giá mới nhất của một mã cổ phiếu.
    Giá quote được giữ nguyên theo đơn vị đồng.
    """

    symbol = _validate_symbol(symbol)
    errors = []

    # Ưu tiên KBS, dự phòng VCI
    for source in ["kbs", "vci"]:
        try:
            df = Market().equity(symbol).quote(
                source=source
            )

            if df is None or df.empty:
                raise ValueError("Không có dữ liệu bảng giá.")

            df.columns = [
                str(col).strip().lower()
                for col in df.columns
            ]

            row = df.iloc[0].to_dict()

            if "close_price" not in row:
                raise ValueError(
                    "Bảng giá không có cột close_price."
                )

            last_price = pd.to_numeric(
                row["close_price"],
                errors="coerce"
            )

            if pd.isna(last_price) or last_price <= 0:
                raise ValueError("Giá khớp gần nhất không hợp lệ.")

            result = {
                "symbol": symbol,
                "exchange": row.get("exchange"),
                "last_price": float(last_price),
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

            # Chuyển các trường giá và khối lượng sang số.
            numeric_fields = [
                "reference_price", "ceiling_price",
                "floor_price", "open_price",
                "high_price", "low_price",
                "bid_price_1", "ask_price_1",
                "bid_vol_1", "ask_vol_1"
            ]

            for field in numeric_fields:
                value = pd.to_numeric(
                    result[field],
                    errors="coerce"
                )

                result[field] = (
                    None if pd.isna(value)
                    else float(value)
                )

            return result

        except Exception as error:
            errors.append(f"{source.upper()}: {error}")

    raise RuntimeError(
        f"Không thể lấy bảng giá {symbol}. "
        + " | ".join(errors)
    )
