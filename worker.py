
import os
import time
import logging
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from data_provider import fetch_exchange_symbols
from scanner import scan_market
from scan_store import save_scan_results

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

SCAN_INTERVAL = max(60, int(os.getenv("SCAN_INTERVAL", "300")))
MAX_SYMBOLS = max(0, int(os.getenv("MAX_SYMBOLS", "0")))

DB_PATH = os.getenv(
    "DATABASE_PATH",
    "data/stock_scanner.sqlite3"
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("stock_scanner_worker")


def is_trading_time():
    now = datetime.now(VN_TZ)

    if now.weekday() >= 5:
        return False

    current = now.time()

    return (
        dt_time(9, 0) <= current <= dt_time(11, 30)
        or dt_time(13, 0) <= current <= dt_time(14, 45)
    )


def get_symbols():
    try:
        symbols = fetch_exchange_symbols(force_refresh=False)

        if not symbols:
            logger.warning("Danh sách mã cổ phiếu rỗng.")
            return []

        result = sorted({
            str(symbol).strip().upper()
            for symbol in symbols
            if symbol and str(symbol).strip().isalnum()
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
        logger.exception("Không lấy được danh sách HOSE/HNX.")
        return []


def convert_results(scan_results):
    """
    Chuẩn hóa kết quả scanner.py sang cấu trúc scan_store.py.

    Chỉ lưu tín hiệu tổng hợp của scanner.
    Không giả lập ba chiến lược Swing, Reversal, Pullback
    khi bộ phân tích chưa tạo ra điểm riêng cho từng chiến lược.
    """

    rows = []

    for item in scan_results:
        symbol = str(item.get("symbol", "")).strip().upper()

        if not symbol:
            continue

        rows.append({
            "symbol": symbol,
            "strategy": "CORE",
            "signal": item.get("signal", "WAIT"),
            "score": item.get("score", 0),
            "price": item.get("price", 0),
            "stop_loss": item.get("stop_loss"),
            "target_price": item.get("target_price"),
            "reason": item.get("reason", ""),
        })

    return rows


def run_scan():
    started = time.time()

    logger.info("Bắt đầu quét thị trường bằng dữ liệu D1.")

    symbols = get_symbols()

    if not symbols:
        logger.error("Không có mã để quét.")
        return 0

    logger.info("Chuẩn bị quét %s mã.", len(symbols))

    def progress(current, total):
        if current % 50 == 0 or current == total:
            logger.info(
                "Tiến độ: %s/%s mã",
                current,
                total
            )

    try:
        report = scan_market(
            max_symbols=None,
            progress_callback=progress,
            symbols=symbols,
        )

        results = report.get("results", [])

        rows = convert_results(results)

        saved = save_scan_results(
            rows,
            db_path=DB_PATH
        ) if rows else 0

        elapsed = round(time.time() - started, 2)

        logger.info(
            "Hoàn thành quét D1 | "
            "Tổng mã: %s | "
            "BUY: %s | "
            "EARLY_BUY: %s | "
            "NEAR_BUY: %s | "
            "WAIT: %s | "
            "FILTERED: %s | "
            "Thiếu lịch sử: %s | "
            "Lỗi dữ liệu: %s | "
            "Đã lưu: %s | "
            "Thời gian: %s giây",
            report.get("scanned", 0),
            report.get("confirmed_buy_count", 0),
            report.get("early_buy_count", 0),
            report.get("near_buy_count", 0),
            report.get("wait_count", 0),
            report.get("filtered_count", 0),
            report.get("insufficient_history_count", 0),
            report.get("d1_error_count", 0),
            saved,
            elapsed,
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

        return saved

    except Exception:
        logger.exception("Lượt quét thất bại.")
        return 0


def run_worker():
    logger.info("AI Stock Scanner Worker đã khởi động.")
    logger.info("Khung phân tích: D1.")
    logger.info("Chu kỳ kiểm tra: %s giây.", SCAN_INTERVAL)

    while True:
        try:
            if is_trading_time():
                run_scan()
            else:
                logger.info(
                    "Ngoài giờ giao dịch. Worker đang chờ."
                )

        except Exception:
            logger.exception("Worker gặp lỗi không dự kiến.")

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run_worker()
