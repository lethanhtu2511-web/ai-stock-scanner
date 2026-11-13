
import time
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

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

        return result if result > 0 else None

    except (TypeError, ValueError):
        return None


def _extract_price(data):
    if isinstance(data, list):
        for item in reversed(data):
            price = _extract_price(item)
            if price:
                return price

    elif isinstance(data, dict):
        keys = [
            "matchPrice",
            "matchedPrice",
            "tradePrice",
            "lastPrice",
            "price",
            "close",
        ]

        for key in keys:
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
    Lấy giá ứng viên từ KBS.

    Không xác nhận realtime vì chưa xác minh được
    timestamp giao dịch do nguồn cung cấp.
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

        if price is None:
            return None

        return {
            "symbol": symbol,
            "price": price,
            "source": "KBS",
            "received_at": time.time(),
            "updated_at": datetime.now(
                VN_TZ
            ).isoformat(timespec="seconds"),
            "is_realtime": False,
            "status": "GIÁ KBS - CHƯA XÁC MINH THỜI GIAN KHỚP",
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ):
        return None


def get_quote(symbol, daily_price=None):
    """
    Ưu tiên giá D1 đã có ngày dữ liệu rõ ràng.

    Giá KBS chưa xác minh timestamp chỉ được trả về
    dưới dạng thông tin tham khảo, không coi là realtime.
    """

    symbol = str(symbol).upper().strip()

    kbs_quote = fetch_latest_trade(symbol)

    fallback_price = _to_float(daily_price)

    if fallback_price is not None:
        return {
            "symbol": symbol,
            "price": fallback_price,
            "source": "D1",
            "timestamp": None,
            "updated_at": None,
            "is_realtime": False,
            "status": "GIÁ ĐÓNG CỬA D1 - CẦN KIỂM TRA NGÀY DỮ LIỆU",
            "unverified_kbs_price": (
                kbs_quote["price"] if kbs_quote else None
            ),
        }

    if kbs_quote:
        return {
            **kbs_quote,
            "timestamp": None,
            "status": "GIÁ KBS CHƯA XÁC MINH - KHÔNG DÙNG LÀM GIÁ REALTIME",
        }

    return {
        "symbol": symbol,
        "price": None,
        "source": None,
        "timestamp": None,
        "updated_at": None,
        "is_realtime": False,
        "status": "KHÔNG CÓ DỮ LIỆU GIÁ",
        "unverified_kbs_price": None,
    }
