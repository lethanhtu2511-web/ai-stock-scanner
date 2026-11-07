import re
import pandas as pd
from vnstock import Market


def fetch_stock_history(symbol, length=260):
    """
    Lấy dữ liệu lịch sử OHLCV từ Vnstock 5.x.
    Chuẩn hóa giá từ nghìn đồng sang đồng.
    Đầu ra gồm: Date, Open, High, Low, Close, Volume.
    """

    # 1. Kiểm tra mã cổ phiếu
    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Bạn chưa nhập mã cổ phiếu.")

    if not re.fullmatch(r"[A-Z0-9]{3,10}", symbol):
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    # 2. Kiểm tra số phiên cần lấy
    try:
        length = int(length)
    except (TypeError, ValueError):
        raise ValueError("Số phiên dữ liệu không hợp lệ.")

    if length < 220:
        raise ValueError("Cần lấy ít nhất 220 phiên dữ liệu.")

    if length > 2000:
        raise ValueError("Không nên lấy quá 2.000 phiên dữ liệu.")

    try:
        # 3. Lấy dữ liệu nến ngày
        market = Market()
        df = market.equity(symbol).ohlcv(
            length=length,
            interval="1D"
        )

        if df is None or df.empty:
            raise ValueError("Nguồn dữ liệu không trả về dữ liệu.")

        # 4. Chuẩn hóa tên cột
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
                "Dữ liệu thiếu cột: " + ", ".join(missing)
            )

        df = df[required].copy()

        # 5. Chuyển kiểu dữ liệu
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        price_columns = [
            "open", "high", "low", "close"
        ]

        for col in price_columns + ["volume"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna()

        # 6. Sắp xếp và loại bỏ dữ liệu trùng
        df = df.sort_values("date")

        df = df.drop_duplicates(
            subset=["date"],
            keep="last"
        )

        if len(df) < 220:
            raise ValueError(
                "Không đủ 220 phiên dữ liệu hợp lệ."
            )

        # 7. Kiểm tra dữ liệu giá
        for col in price_columns:
            if (df[col] <= 0).any():
                raise ValueError(
                    "Dữ liệu giá có giá trị không hợp lệ."
                )

        if (df["volume"] < 0).any():
            raise ValueError(
                "Dữ liệu khối lượng không hợp lệ."
            )

        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )

        if invalid_ohlc.any():
            raise ValueError(
                "Dữ liệu OHLC có bất thường."
            )

        # 8. Vnstock trả giá cổ phiếu theo nghìn đồng.
        # Ví dụ: 63.0 = 63.000 đồng.
        # Chuyển sang đồng để scanner tính toán thống nhất.
        df[price_columns] = df[price_columns] * 1000

        # 9. Đổi tên cột theo định dạng scanner.py
        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        return df.reset_index(drop=True)

    except Exception as error:
        raise RuntimeError(
            f"Không thể lấy dữ liệu {symbol}: {error}"
        ) from error
