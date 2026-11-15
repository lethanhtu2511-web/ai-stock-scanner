
import time
from datetime import datetime, time as dt_time
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

# Không chấp nhận dữ liệu có thời điểm cập nhật từ nguồn quá 5 phút.
MAX_QUOTE_AGE_SECONDS = 300


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


def _parse_source_time(value):
    """
    Chuyển thời gian do nguồn dữ liệu cung cấp thành datetime
    có múi giờ Việt Nam.

    Không sử dụng fetched_at để thay thế source_time.
    Nếu nguồn không cung cấp thời gian có thể xác minh,
    hàm trả về None.
    """

    if value is None:
        return None

    try:
        # Unix timestamp dạng số hoặc chuỗi số.
        numeric = None

        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            raw = value.strip()

            if raw.isdigit():
                numeric = float(raw)

        if numeric is not None:
            # Phân biệt timestamp giây và mili giây.
            if numeric > 10_000_000_000:
                numeric = numeric / 1000.0

            parsed = datetime.fromtimestamp(
                numeric,
                tz=VN_TZ,
            )

            return parsed

        if not isinstance(value, str):
            return None

        raw = value.strip()

        # ISO datetime hoặc định dạng ngày giờ phổ biến.
        normalized = raw.replace("Z", "+00:00")

        try:
            parsed = datetime.fromisoformat(normalized)

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=VN_TZ)
            else:
                parsed = parsed.astimezone(VN_TZ)

            return parsed

        except ValueError:
            pass

        # Một số bảng giá chỉ trả giờ HH:MM:SS.
        # Chỉ gắn ngày hiện tại; bước kiểm tra phiên giao dịch
        # sẽ ngăn việc coi giờ cũ vào ngày nghỉ là giá realtime.
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                parsed_time = datetime.strptime(
                    raw,
                    fmt,
                ).time()

                now = datetime.now(VN_TZ)

                return datetime.combine(
                    now.date(),
                    parsed_time,
                    tzinfo=VN_TZ,
                )

            except ValueError:
                continue

    except (TypeError, ValueError, OverflowError, OSError):
        return None

    return None


def _is_market_session(now=None):
    """
    Kiểm tra khung giờ giao dịch thông thường của HOSE/HNX.
    Đây chỉ là điều kiện cần, không thay thế xác minh timestamp
    thực tế từ nguồn dữ liệu.
    """

    now = now or datetime.now(VN_TZ)
    now = now.astimezone(VN_TZ)

    if now.weekday() >= 5:
        return False

    current_time = now.time()

    morning = (
        dt_time(9, 0)
        <= current_time
        <= dt_time(11, 30)
    )

    afternoon = (
        dt_time(13, 0)
        <= current_time
        <= dt_time(15, 0)
    )

    return morning or afternoon


def _quote_freshness(source_time, now=None):
    now = now or datetime.now(VN_TZ)
    now = now.astimezone(VN_TZ)

    parsed_time = _parse_source_time(source_time)

    if parsed_time is None:
        return {
            "is_fresh": False,
            "source_time_parsed": None,
            "age_seconds": None,
            "status": "UNKNOWN",
            "message": "Nguồn không cung cấp thời điểm cập nhật hợp lệ.",
        }

    age_seconds = (
        now - parsed_time
    ).total_seconds()

    # Không nhận timestamp ở tương lai hoặc quá cũ.
    if age_seconds < -30:
        return {
            "is_fresh": False,
            "source_time_parsed": parsed_time.isoformat(),
            "age_seconds": round(age_seconds, 1),
            "status": "INVALID_TIME",
            "message": "Thời điểm cập nhật từ nguồn nằm trong tương lai.",
        }

    if age_seconds > MAX_QUOTE_AGE_SECONDS:
        return {
            "is_fresh": False,
            "source_time_parsed": parsed_time.isoformat(),
            "age_seconds": round(age_seconds, 1),
            "status": "STALE",
            "message": "Giá từ nguồn đã cũ quá 5 phút.",
        }

    if not _is_market_session(now):
        return {
            "is_fresh": False,
            "source_time_parsed": parsed_time.isoformat(),
            "age_seconds": round(age_seconds, 1),
            "status": "OUTSIDE_SESSION",
            "message": "Hiện không nằm trong khung giờ giao dịch.",
        }

    if not _is_market_session(parsed_time):
        return {
            "is_fresh": False,
            "source_time_parsed": parsed_time.isoformat(),
            "age_seconds": round(age_seconds, 1),
            "status": "OUTSIDE_SESSION",
            "message": "Thời điểm cập nhật không thuộc khung giờ giao dịch.",
        }

    return {
        "is_fresh": True,
        "source_time_parsed": parsed_time.isoformat(),
        "age_seconds": round(age_seconds, 1),
        "status": "FRESH",
        "message": "Thời điểm cập nhật từ nguồn còn mới.",
    }


def _normalize_quote(row, fetched_at):
    symbol = _normalize_symbol(
        row.get("SB")
        or row.get("symbol")
        or row.get("code")
    )

    if not symbol:
        return None

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

    freshness = _quote_freshness(source_time)

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
        "source_time_parsed": freshness["source_time_parsed"],
        "quote_age_seconds": freshness["age_seconds"],
        "quote_freshness": freshness["status"],
        "quote_is_fresh": freshness["is_fresh"],
        "quote_freshness_message": freshness["message"],
        "fetched_at": fetched_at,
        "source": "KBS public price board",
        "data_type": (
            "verified_intraday"
            if freshness["is_fresh"]
            else "unverified_price_board"
        ),
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
    Lấy dữ liệu bảng giá KBS.

    Lưu ý quan trọng:
    - fetched_at chỉ là thời điểm bot nhận phản hồi.
    - Chỉ quote có source_time hợp lệ, còn mới và thuộc
      khung giờ giao dịch mới được đánh dấu quote_is_fresh=True.
    - Không tự thay giá thiếu bằng giá đóng cửa D1.
    - Dữ liệu không xác minh được vẫn được trả về để hiển thị,
      nhưng scanner không được dùng làm giá realtime xác nhận mua.
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
            "Không lấy được dữ liệu bảng giá KBS. "
            + detail
        )

    return {
        symbol: quotes[symbol]
        for symbol in quotes
    }


def get_quote_freshness(quote):
    """
    Trả trạng thái xác minh độ mới của báo giá.
    Không xem fetched_at là thời điểm khớp lệnh.
    """

    if not quote:
        return {
            "status": "MISSING",
            "is_fresh": False,
            "message": "Không có dữ liệu giá.",
        }

    source_time = quote.get("source_time")

    result = _quote_freshness(source_time)

    return {
        "status": result["status"],
        "is_fresh": result["is_fresh"],
        "source_time": result["source_time_parsed"],
        "age_seconds": result["age_seconds"],
        "message": result["message"],
    }
