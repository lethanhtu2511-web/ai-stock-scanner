
import os
import time
import logging
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from data_provider import fetch_exchange_symbols
from scanner import scan_market
from realtime_provider import (
    fetch_realtime_quotes,
    RealtimeDataError,
)
from scan_store import save_scan_results, get_scan_summary

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

SCAN_INTERVAL = max(
    60,
    int(os.getenv("SCAN_INTERVAL", "300"))
)

MAX_SYMBOLS = max(
    0,
    int(os.getenv("MAX_SYMBOLS", "0"))
)

DAILY_SCAN_TIME = dt_time(15, 35)

DB_PATH = os.getenv(
    "DATABASE_PATH",
    "data/stock_scanner.sqlite3"
)

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
).upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("stock_scanner_worker")


STRATEGY_NAMES = {
    "SWING": "Lướt sóng 2–5 phiên",
    "REVERSAL": "Đảo chiều 5–10 phiên",
    "PULLBACK": "Pullback 3–7 phiên",
}


def should_run_daily_scan():
    """Chỉ quét tự động sau 15:35 các ngày trong tuần."""

    now = datetime.now(VN_TZ)

    if now.weekday() >= 5:
        return False

    if now.time() < DAILY_SCAN_TIME:
        return False

    try:
        summary = get_scan_summary(DB_PATH)
        latest = summary.get("latest_scan_at")

        if not latest:
            return True

        latest_time = datetime.fromisoformat(latest)

        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=VN_TZ)

        latest_time = latest_time.astimezone(VN_TZ)

        return not (
            latest_time.date() == now.date()
            and latest_time.time() >= DAILY_SCAN_TIME
        )

    except Exception:
        logger.exception(
            "Không kiểm tra được lịch sử quét."
        )
        return True


def get_symbols():
    try:
        symbols = fetch_exchange_symbols(
            force_refresh=False
        )

        if not symbols:
            logger.warning(
                "Danh sách mã cổ phiếu rỗng."
            )
            return []

        result = sorted({
            str(symbol).strip().upper()
            for symbol in symbols
            if (
                symbol
                and str(symbol).strip().isalnum()
            )
        })

        if MAX_SYMBOLS > 0:
            result = result[:MAX_SYMBOLS]

        stats = getattr(
            fetch_exchange_symbols,
            "last_stats",
            {}
        )

        logger.info(
            "Nguồn danh sách: %s | Số mã: %s",
            stats.get("source", "Không rõ"),
            len(result)
        )

        return result

    except Exception:
        logger.exception(
            "Không lấy được danh sách HOSE/HNX."
        )
        return []


def convert_results(scan_results):
    """Chuẩn hóa kết quả trước khi lưu SQLite."""

    rows = []

    fields = [
        "data_date",
        "reward_risk",
        "stop_distance_pct",
        "target_distance_pct",
        "atr_pct",
        "relative_strength",
        "avg_turnover_20d",
        "median_turnover_20d",
        "market_context",
        "blockers",
        "reasons",
        "penalties",
        "d1_close",
        "realtime_price",
        "realtime_volume",
        "quote_fetched_at",
        "quote_source",
        "price_data_status",
    ]

    for item in scan_results:
        symbol = str(
            item.get("symbol", "")
        ).strip().upper()

        strategy = str(
            item.get("strategy", "")
        ).strip().upper()

        if not symbol or strategy not in STRATEGY_NAMES:
            continue

        row = {
            "symbol": symbol,
            "strategy": strategy,
            "signal": item.get("signal", "WAIT"),
            "score": item.get("score"),
            "price": item.get("price"),
            "stop_loss": item.get("stop_loss"),
            "target_price": item.get("target_price"),
            "reason": item.get("reason", ""),
        }

        for field in fields:
            if field in item:
                row[field] = item[field]

        rows.append(row)

    return rows


def _emit_progress(callback, percent, message):
    """Gửi tiến độ cho giao diện; lỗi giao diện không làm hỏng lượt quét."""

    if callback is None:
        return

    try:
        callback(
            max(0, min(100, int(percent))),
            str(message),
        )
    except Exception:
        logger.exception(
            "Không cập nhật được tiến độ giao diện."
        )


def run_scan(progress_callback=None):
    """
    Một lượt quét gồm hai giai đoạn:

    1. Lấy giá trong phiên từ nguồn realtime.
    2. Phân tích chỉ báo trên dữ liệu nến D1.

    Không dùng giá D1 thay thế rồi gắn nhãn realtime.
    """

    started = time.time()

    logger.info(
        "Bắt đầu quét HOSE/HNX: giá trong phiên + phân tích D1."
    )

    _emit_progress(
        progress_callback,
        0,
        "Đang lấy danh sách cổ phiếu..."
    )

    symbols = get_symbols()

    if not symbols:
        logger.error(
            "Không có mã cổ phiếu để quét."
        )

        _emit_progress(
            progress_callback,
            100,
            "Không lấy được danh sách cổ phiếu."
        )

        return 0

    total = len(symbols)

    logger.info(
        "Chuẩn bị quét %s mã.",
        total
    )

    # -------------------------------------------------
    # GIAI ĐOẠN 1: LẤY GIÁ TRONG PHIÊN
    # -------------------------------------------------

    realtime_quotes = {}

    _emit_progress(
        progress_callback,
        2,
        f"Đang lấy giá trong phiên cho {total} mã..."
    )

    def quote_progress(current, quote_total):
        percent = (
            5 + 25 * current / max(quote_total, 1)
        )

        _emit_progress(
            progress_callback,
            percent,
            f"Đang lấy giá trong phiên: "
            f"{current}/{quote_total} mã"
        )

    try:
        realtime_quotes = fetch_realtime_quotes(
            symbols,
            progress_callback=quote_progress,
        )

        realtime_quotes = (
            realtime_quotes
            if isinstance(realtime_quotes, dict)
            else {}
        )

        quote_count = len(realtime_quotes)

        coverage = (
            quote_count / total * 100
            if total else 0
        )

        logger.info(
            "Lấy được giá trong phiên: %s/%s mã (%.1f%%).",
            quote_count,
            total,
            coverage
        )

        if quote_count < total:
            logger.warning(
                "Một số mã chưa có giá trong phiên. "
                "Các mã này sẽ không được xác nhận BUY."
            )

    except RealtimeDataError as exc:
        logger.warning(
            "Nguồn giá trong phiên không khả dụng: %s",
            exc
        )

    except Exception:
        logger.exception(
            "Lỗi khi lấy dữ liệu giá trong phiên."
        )

    quote_count = len(realtime_quotes)

    _emit_progress(
        progress_callback,
        30,
        f"Đã lấy giá trong phiên cho {quote_count}/{total} mã. "
        "Bắt đầu phân tích D1..."
    )

    # -------------------------------------------------
    # GIAI ĐOẠN 2: PHÂN TÍCH D1
    # -------------------------------------------------

    def d1_progress(current, d1_total):
        percent = (
            30 + 70 * current / max(d1_total, 1)
        )

        _emit_progress(
            progress_callback,
            percent,
            f"Đang phân tích D1: {current}/{d1_total} mã"
        )

        if current % 50 == 0 or current == d1_total:
            logger.info(
                "Tiến độ phân tích D1: %s/%s mã",
                current,
                d1_total
            )

    try:
        report = scan_market(
            max_symbols=None,
            progress_callback=d1_progress,
            symbols=symbols,
            realtime_quotes=realtime_quotes,
        )

        results = report.get("results", [])

        rows = convert_results(results)

        saved = (
            save_scan_results(
                rows,
                db_path=DB_PATH
            )
            if rows else 0
        )

        elapsed = round(
            time.time() - started,
            2
        )

        logger.info(
            "Hoàn thành quét | "
            "Mã xử lý: %s | "
            "Dòng kết quả: %s | "
            "Giá trong phiên: %s/%s | "
            "MUA: %s | "
            "CHỜ: %s | "
            "THEO DÕI: %s | "
            "BỎ QUA: %s | "
            "BỊ LỌC: %s | "
            "Thiếu lịch sử: %s | "
            "Lỗi D1: %s | "
            "Đã lưu: %s | "
            "Thời gian: %s giây",
            report.get("scanned", 0),
            report.get("result_count", len(results)),
            quote_count,
            total,
            report.get("confirmed_buy_count", 0),
            report.get("wait_count", 0),
            report.get("watch_count", 0),
            report.get("skip_count", 0),
            report.get("filtered_count", 0),
            report.get("insufficient_history_count", 0),
            report.get("d1_error_count", 0),
            saved,
            elapsed,
        )

        strategy_counts = report.get(
            "strategy_counts",
            {}
        )

        for strategy, name in STRATEGY_NAMES.items():
            stats = strategy_counts.get(strategy, {})

            logger.info(
                "%s | MUA: %s | CHỜ: %s | "
                "THEO DÕI: %s | BỎ QUA: %s | "
                "BỊ LỌC: %s",
                name,
                stats.get("buy", 0),
                stats.get("wait", 0),
                stats.get("watch", 0),
                stats.get("skip", 0),
                stats.get("filtered", 0),
            )

        errors = report.get("errors", [])

        for error in errors[:10]:
            logger.warning(
                "Lỗi mã %s | Giai đoạn %s | %s",
                error.get("symbol"),
                error.get("stage"),
                error.get("error"),
            )

        if len(errors) > 10:
            logger.warning(
                "Còn %s lỗi khác không hiển thị.",
                len(errors) - 10
            )

        _emit_progress(
            progress_callback,
            100,
            f"Hoàn thành: đã lưu {saved} dòng kết quả."
        )

        return saved

    except Exception:
        logger.exception(
            "Lượt quét thất bại."
        )

        _emit_progress(
            progress_callback,
            100,
            "Quét thất bại. Hãy kiểm tra nhật ký lỗi."
        )

        return 0


def run_worker():
    logger.info(
        "AI Stock Scanner Worker đã khởi động."
    )

    logger.info(
        "Chỉ báo kỹ thuật: D1."
    )

    logger.info(
        "Giá xác nhận: dữ liệu trong phiên nếu nguồn cung cấp."
    )

    logger.info(
        "Ba chiến lược: SWING, REVERSAL, PULLBACK."
    )

    logger.info(
        "Thời điểm quét tự động: sau %s.",
        DAILY_SCAN_TIME.strftime("%H:%M")
    )

    logger.info(
        "Chu kỳ kiểm tra: %s giây.",
        SCAN_INTERVAL
    )

    while True:
        try:
            if should_run_daily_scan():
                logger.info(
                    "Đến thời điểm quét tự động trong ngày."
                )

                saved = run_scan()

                if saved > 0:
                    logger.info(
                        "Đã lưu %s kết quả. "
                        "Không quét lặp lại trong ngày.",
                        saved
                    )
                else:
                    logger.warning(
                        "Chưa lưu được kết quả. "
                        "Worker sẽ thử lại ở chu kỳ tiếp theo."
                    )

            else:
                logger.info(
                    "Chưa đến giờ quét hoặc hôm nay đã quét."
                )

        except Exception:
            logger.exception(
                "Worker gặp lỗi không dự kiến."
            )

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run_worker()
