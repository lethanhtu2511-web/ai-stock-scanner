
import os
import time
import logging
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from data_provider import fetch_exchange_symbols
from scanner import scan_market
from scan_store import save_scan_results, get_scan_summary

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Kiểm tra lịch chạy định kỳ, nhưng chỉ phân tích nến ngày D1.
SCAN_INTERVAL = max(
    60,
    int(os.getenv("SCAN_INTERVAL", "300"))
)

MAX_SYMBOLS = max(
    0,
    int(os.getenv("MAX_SYMBOLS", "0"))
)

# Chờ nến ngày hoàn tất trước khi quét.
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
    """
    Chỉ quét từ 15:35 các ngày giao dịch.
    Nếu đã có kết quả sau 15:35 hôm nay thì không quét lại.
    """

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

        if (
            latest_time.date() == now.date()
            and latest_time.time() >= DAILY_SCAN_TIME
        ):
            return False

        return True

    except Exception:
        logger.exception(
            "Không kiểm tra được lịch sử quét. "
            "Worker sẽ thử quét."
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
    """
    Chuyển kết quả scanner thành các dòng lưu SQLite.
    Mỗi mã có tối đa 3 dòng:
    SWING, REVERSAL, PULLBACK.
    """

    rows = []

    for item in scan_results:
        symbol = str(
            item.get("symbol", "")
        ).strip().upper()

        strategy = str(
            item.get("strategy", "")
        ).strip().upper()

        if not symbol or strategy not in STRATEGY_NAMES:
            continue

        rows.append({
            "symbol": symbol,
            "strategy": strategy,
            "signal": item.get("signal", "WAIT"),
            "score": item.get("score"),
            "price": item.get("price"),
            "stop_loss": item.get("stop_loss"),
            "target_price": item.get("target_price"),
            "reason": item.get("reason", ""),
        })

    return rows


def run_scan():
    started = time.time()

    logger.info(
        "Bắt đầu quét HOSE/HNX bằng dữ liệu nến D1."
    )

    symbols = get_symbols()

    if not symbols:
        logger.error(
            "Không có mã cổ phiếu để quét."
        )
        return 0

    logger.info(
        "Chuẩn bị phân tích %s mã, tối đa %s kết quả chiến lược.",
        len(symbols),
        len(symbols) * 3
    )

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
            "Hoàn thành quét D1 | "
            "Mã đã xử lý: %s | "
            "Tổng dòng kết quả: %s | "
            "MUA: %s | "
            "CHỜ: %s | "
            "THEO DÕI: %s | "
            "BỎ QUA: %s | "
            "BỊ LỌC: %s | "
            "Thiếu lịch sử: %s | "
            "Lỗi dữ liệu: %s | "
            "Đã lưu: %s | "
            "Thời gian: %s giây",
            report.get("scanned", 0),
            report.get("result_count", len(results)),
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

        return saved

    except Exception:
        logger.exception(
            "Lượt quét D1 thất bại."
        )
        return 0


def run_worker():
    logger.info(
        "AI Stock Scanner Worker đã khởi động."
    )

    logger.info(
        "Khung phân tích: D1."
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
                    "Đến thời điểm quét D1 trong ngày."
                )

                saved = run_scan()

                if saved > 0:
                    logger.info(
                        "Đã lưu %s kết quả chiến lược. "
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
                    "Chưa đến giờ quét D1 "
                    "hoặc hôm nay đã quét."
                )

        except Exception:
            logger.exception(
                "Worker gặp lỗi không dự kiến."
            )

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run_worker()
