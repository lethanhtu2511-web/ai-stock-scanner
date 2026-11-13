
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Giá được xem là mới nếu cập nhật trong vòng 180 giây.
MAX_QUOTE_AGE_SECONDS = 180

# Endpoint lịch sử giao dịch KBS.
# Đây là nguồn thử nghiệm, có thể thay đổi hoặc ngừng hoạt động.
KBS_TRADE_URL = (
    "https://iboard-query.kbsec.com.vn/"
    "api/Market/TradeHistory"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}


def _to_float(value):
    try:
        if value is None:
            return None

        if isinstance(value, str):
            value = value.replace(",", "").strip()

        result = float(value)

        if result <= 0:
            return None

        return result

    except (TypeError, ValueError):
        return None


def _extract_price(data):
    """Thử nhận diện giá từ các cấu trúc JSON phổ biến."""

    if isinstance(data, list):
        for item in reversed(data):
            price = _extract_price(item)
            if price:
                return price

    elif isinstance(data, dict):
        preferred_keys = [
            "matchPrice",
            "matchedPrice",
            "tradePrice",
            "lastPrice",
            "price",
            "close",
        ]

        for key in preferred_keys:
            if key in data:
                price = _to_float(data[key])
                if price:
                    return price

        for value in data.values():
            if isinstance(value, (dict, list)):
                price = _extract_price(value)
                if price:
                    return price

    return None


def fetch_latest_trade(symbol, timeout=5):
    """
    Lấy giá giao dịch gần nhất nếu endpoint còn hoạt động.

    Trả về None nếu không lấy được dữ liệu đáng tin cậy.
    """

    symbol = str(symbol).upper().strip()

    if not symbol:
        return None

    try:
        response = requests.get(
            KBS_TRADE_URL,
            params={"symbol": symbol},
            headers=HEADERS,
            timeout=timeout,
        )

        response.raise_for_status()
        data = response.json()

        price = _extract_price(data)

        if not price:
            return None

        return {
            "symbol": symbol,
            "price": price,
            "source": "KBS",
            "timestamp": time.time(),
            "updated_at": datetime.now(VN_TZ).isoformat(
                timespec="seconds"
            ),
            "is_realtime": False,
            "status": "CHƯA XÁC MINH THỜI GIAN DỮ LIỆU",
        }

    except (requests.RequestException, ValueError, TypeError):
        return None


def get_quote(symbol, daily_price=None):
    """
    Ưu tiên giá giao dịch gần nhất.
    Nếu không có, dùng giá đóng cửa D1 và đánh dấu dữ liệu cũ.

    daily_price: giá đóng cửa D1 từ data_provider.py.
    """

    symbol = str(symbol).upper().strip()

    quote = fetch_latest_trade(symbol)

    if quote:
        age = time.time() - quote["timestamp"]

        if age <= MAX_QUOTE_AGE_SECONDS:
            quote["is_realtime"] = True
            quote["status"] = "GIÁ MỚI - CẦN ĐỐI CHIẾU NGUỒN"
            return quote

    fallback_price = _to_float(daily_price)

    if fallback_price is None:
        return {
            "symbol": symbol,
            "price": None,
            "source": None,
            "timestamp": None,
            "updated_at": None,
            "is_realtime": False,
            "status": "KHÔNG CÓ DỮ LIỆU GIÁ",
        }

    return {
        "symbol": symbol,
        "price": fallback_price,
        "source": "D1",
        "timestamp": None,
        "updated_at": None,
        "is_realtime": False,
        "status": "GIÁ ĐÓNG CỬA D1 - DỮ LIỆU CÓ THỂ CŨ",
    }
