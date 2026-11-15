
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

KBS_PRICE_BOARD_URL = (
    "https://kbbuddywts.kbsec.com.vn/"
    "iis-server/investment/stock/iss"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
    "Content-Type": "application/json",
    "Origin": "https://kbbuddywts.kbsec.com.vn",
    "Referer": "https://kbbuddywts.kbsec.com.vn/",
    "x-lang": "vi",
}

BATCH_SIZE = 80
REQUEST_TIMEOUT = 20.0
MAX_RETRIES = 2


class RealtimeDataError(RuntimeError):
    pass


def _normalize_symbol(value):
    return str(value or "").strip().upper()


def _to_number(value):
    if value is None:
        return None

    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()

        number = float(value)

        if number != number:
            return None

        return number

    except (TypeError, ValueError):
        return None


def _first_number(row, keys):
    for key in keys:
        value = _to_number(row.get(key))

        if value is not None and value > 0:
            return value

    return None


def _unwrap_response(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "data",
            "result",
            "items",
            "stocks",
        ):
            value = data.get(key)

            if isinstance(value, list):
                return value

    raise RealtimeDataError(
        "KBS trả về cấu trúc bảng giá không hợp lệ."
    )


def _normalize_quote(row, fetched_at):
    symbol = _normalize_symbol(
        row.get("SB")
        or row.get("symbol")
        or row.get("code")
    )

    if not symbol:
        return None

    # CP là giá khớp hiện tại theo mapping KBS.
    price = _first_number(
        row,
        (
            "CP",
            "match_price",
            "close_price",
            "last_price",
            "price",
        ),
    )

    if price is None:
        return None

    volume = _first_number(
        row,
        (
            "TT",
            "volume_accumulated",
            "accumulated_volume",
            "total_volume",
        ),
    )

    source_time = (
        row.get("t")
        or row.get("time")
        or row.get("timestamp")
    )

    return {
        "symbol": symbol,
        "price": price,
        "volume": volume,
        "open": _first_number(
            row,
            ("OP", "open_price", "open"),
        ),
        "high": _first_number(
            row,
            ("HI", "high_price", "high"),
        ),
        "low": _first_number(
            row,
            ("LO", "low_price", "low"),
        ),
        "reference": _first_number(
            row,
            ("RE", "reference_price", "reference"),
        ),
        "percent_change": _to_number(
            row.get("CHP")
            or row.get("percent_change")
        ),
        "source_time": source_time,
        "fetched_at": fetched_at,
        "source": "KBS public price board",
        "data_type": "intraday_price_board",
    }


def _fetch_batch(client, symbols):
    payload = {
        "code": ",".join(symbols),
    }

    errors = []

    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(
                KBS_PRICE_BOARD_URL,
                json=payload,
            )

            response.raise_for_status()

            data = response.json()
            rows = _unwrap_response(data)

            fetched_at = datetime.now(
                VN_TZ
            ).isoformat(timespec="seconds")

            quotes = {}

            for row in rows:
                if not isinstance(row, dict):
                    continue

                quote = _normalize_quote(
                    row,
                    fetched_at,
                )

                if quote:
                    quotes[quote["symbol"]] = quote

            return quotes

        except Exception as exc:
            errors.append(str(exc))

            if attempt + 1 < MAX_RETRIES:
                time.sleep(1.0)

    raise RealtimeDataError(
        "Không lấy được bảng giá KBS: "
        + " | ".join(errors)
    )


def fetch_realtime_quotes(
    symbols,
    batch_size=BATCH_SIZE,
    progress_callback=None,
):
    """
    Lấy bảng giá trong phiên từ KBS.

    Parameters
    ----------
    symbols:
        Danh sách mã HOSE/HNX.

    batch_size:
        Số mã trong một yêu cầu. Mặc định 80.

    progress_callback:
        Callback nhận (processed, total).

    Returns
    -------
    dict:
        {
            "HPG": {
                "price": ...,
                "volume": ...,
                "source_time": ...,
                "fetched_at": ...,
                "source": ...
            }
        }

    Lưu ý:
    - Không dùng cache nội bộ.
    - Không thay giá thiếu bằng giá đóng cửa cũ.
    - Nguồn công khai có thể bị trễ hoặc từ chối truy cập.
    """

    normalized = sorted({
        _normalize_symbol(symbol)
        for symbol in symbols
        if _normalize_symbol(symbol)
    })

    if not normalized:
        return {}

    batch_size = max(
        1,
        min(int(batch_size), 100),
    )

    total = len(normalized)
    batches = [
        normalized[index:index + batch_size]
        for index in range(0, total, batch_size)
    ]

    quotes = {}
    errors = []

    with httpx.Client(
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:

        processed = 0

        for batch in batches:
            try:
                result = _fetch_batch(
                    client,
                    batch,
                )

                quotes.update(result)

            except Exception as exc:
                errors.append({
                    "symbols": batch,
                    "error": str(exc),
                })

            processed += len(batch)

            if progress_callback:
                progress_callback(
                    processed,
                    total,
                )

    if not quotes:
        detail = (
            errors[0]["error"]
            if errors else "Không có mã nào trả dữ liệu."
        )

        raise RealtimeDataError(
            "Không lấy được dữ liệu giá trong phiên. "
            + detail
        )

    return {
        symbol: quotes[symbol]
        for symbol in quotes
    }


def get_quote_freshness(quote):
    """
    Kiểm tra thời điểm lấy dữ liệu từ nguồn.
    Không suy đoán thời gian khớp lệnh nếu nguồn không trả về.
    """

    if not quote:
        return {
            "status": "MISSING",
            "message": "Không có dữ liệu giá.",
        }

    source_time = quote.get("source_time")

    if not source_time:
        return {
            "status": "UNKNOWN",
            "message": (
                "Nguồn không cung cấp thời điểm cập nhật."
            ),
        }

    return {
        "status": "RECEIVED",
        "message": (
            f"Dữ liệu bảng giá nhận lúc "
            f"{source_time}."
        ),
    }
