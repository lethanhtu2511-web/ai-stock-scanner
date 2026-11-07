
import pandas as pd
from vnstock import Market


def fetch_stock_history(symbol, length=260):
    """
    Lấy dữ liệu giá lịch sử của một cổ phiếu.
    Chuẩn hóa dữ liệu về định dạng mà scanner.py sử dụng.
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Chưa nhập mã cổ phiếu.")

    if not symbol.isalnum():
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    if length < 200:
        raise ValueError("Cần tối thiểu 200 phiên dữ liệu.")

    try:
        df = Market().equity(symbol).ohlcv(length=length)

        if df is None or df.empty:
            raise ValueError("Nguồn dữ liệu không trả về dữ liệu.")

        # Chuẩn hóa tên cột
        df.columns = [
            str(col).strip().lower()
            for col in df.columns
        ]

        # Chuẩn hóa tên cột ngày giao dịch
        aliases = {
            "time": "date",
            "tradingdate": "date",
            "trading_date": "date",
        }

        df = df.rename(columns=aliases)

        required = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "Thiếu cột dữ liệu: " + ", ".join(missing)
            )

        # Chỉ giữ các trường cần thiết
        df = df[required].copy()

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume"
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

        if len(df) < 200:
            raise ValueError(
                "Không đủ 200 phiên dữ liệu hợp lệ."
            )

        if (df["close"] <= 0).any():
            raise ValueError("Dữ liệu giá không hợp lệ.")

        # Đổi tên cột theo chuẩn scanner.py
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
