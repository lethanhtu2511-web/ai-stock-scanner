import re
import pandas as pd
from vnstock import Market


def fetch_stock_history(symbol, length=260):

    symbol = str(symbol).strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError("Mã cổ phiếu không hợp lệ.")

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

            # Giá OHLCV của cổ phiếu là nghìn đồng.
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

            # Ghi lại nguồn dữ liệu thực tế
            result.attrs["source"] = source.upper()

            return result

        except Exception as error:
            errors.append(
                f"{source.upper()}: {error}"
            )

    raise RuntimeError(
        f"Không thể lấy dữ liệu {symbol}. "
        + " | ".join(errors)
    )
