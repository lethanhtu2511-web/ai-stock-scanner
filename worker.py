
import os
import time
import logging
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from data_provider import (
    fetch_exchange_symbols,
    fetch_stock_history,
)

from strategy_engine import analyze_stock
from scan_store import save_scan_results

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "0"))

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

    morning_start = dt_time(9, 0)
    morning_end = dt_time(11, 30)

    afternoon_start = dt_time(13, 0)
    afternoon_end = dt_time(14, 45)

    return (
        morning_start <= current <= morning_end
        or afternoon_start <= current <= afternoon_end
    )


def get_symbols():
    symbols = fetch_exchange_symbols(force_refresh=False)

    if symbols is None:
        return []

    if hasattr(symbols, "tolist"):
        symbols = symbols.tolist()

    if isinstance(symbols, dict):
        symbols = list(symbols.keys())

    if isinstance(symbols, (list, tuple, set)):
        result = []

        for item in symbols:
            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("ticker")
            else:
                symbol = str(item)

            if symbol:
                result.append(str(symbol).upper().strip())

        symbols = result

    else:
        return []

    symbols = sorted(set(
        symbol for symbol in symbols
        if symbol and symbol.isalnum()
    ))

    if MAX_SYMBOLS > 0:
        symbols = symbols[:MAX_SYMBOLS]

    return symbols


def scan_one_symbol(symbol):
    try:
        df = fetch_stock_history(symbol, length=260)

        if df is None or len(df) < 60:
            return []

        analysis = analyze_stock(symbol, df)

        results = []

        for strategy_code, strategy_data in analysis[
            "strategies"
        ].items():

            results.append({
                "symbol": symbol,
                "strategy": strategy_code,
                "signal": strategy_data["signal"],
                "score": strategy_data["score"],
                "price": analysis["price"],
                "stop_loss": analysis["stop_loss"],
                "target_price": analysis["target_price"],
                "reason": "; ".join(
                    strategy_data["reasons"]
                ),
            })

        return results

    except Exception as exc:
        logger.warning(
            "Lỗi phân tích %s: %s",
            symbol,
            exc
        )
        return []


def run_scan():
    started = time.time()

    logger.info("Bắt đầu quét thị trường D1.")

    symbols = get_symbols()

    if not symbols:
        logger.warning("Không lấy được danh sách mã cổ phiếu.")
        return 0

    logger.info("Số mã cần quét: %s", len(symbols))

    all_results = []
    success = 0

    for index, symbol in enumerate(symbols, start=1):
        results = scan_one_symbol(symbol)

        if results:
            all_results.extend(results)
            success += 1

        if index % 50 == 0:
            logger.info(
                "Tiến độ: %s/%s mã",
                index,
                len(symbols)
            )

    if all_results:
        saved = save_scan_results(
            all_results,
            db_path=DB_PATH
        )
    else:
        saved = 0

    elapsed = round(time.time() - started, 2)

    logger.info(
        "Hoàn thành: %s/%s mã có dữ liệu, "
        "%s kết quả được lưu, thời gian %s giây.",
        success,
        len(symbols),
        saved,
        elapsed
    )

    return saved


def run_worker():
    logger.info("AI Stock Scanner Worker đã khởi động.")

    while True:
        try:
            if is_trading_time():
                run_scan()
            else:
                logger.info(
                    "Ngoài giờ giao dịch. Worker đang chờ."
                )

        except Exception:
            logger.exception("Worker gặp lỗi.")

        time.sleep(max(60, SCAN_INTERVAL))


if __name__ == "__main__":
    run_worker()
