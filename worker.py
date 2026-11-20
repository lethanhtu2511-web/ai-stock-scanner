
import os
import time
import logging
import sqlite3
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

from data_provider import fetch_exchange_symbols, clear_history_cache
from scanner import scan_market
from realtime_provider import (
    fetch_realtime_quotes,
    RealtimeDataError,
)
from scan_store import save_scan_results, get_scan_summary
from scan_lock import scan_lock
from backup_manager import backup_database
from runtime_control import get_control, worker_heartbeat, complete_manual_scan


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
DAILY_BACKUP_TIME = dt_time(15, 45)
BACKUP_DIR = os.getenv("BACKUP_DIR", "/app/data/backups")
BACKUP_RETENTION = max(1, int(os.getenv("BACKUP_RETENTION", "14")))

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


def is_market_session(now=None):
    now = now or datetime.now(VN_TZ)

    if now.weekday() >= 5:
        return False

    current = now.time()
    morning = dt_time(9, 0) <= current <= dt_time(11, 30)
    afternoon = dt_time(13, 0) <= current <= dt_time(15, 0)
    return morning or afternoon


def should_run_intraday_scan():
    """Quét lại giá và tín hiệu trong phiên theo chu kỳ 300 giây."""
    now = datetime.now(VN_TZ)
    if not is_market_session(now):
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

        age = (now - latest_time).total_seconds()
        return age >= SCAN_INTERVAL - 15

    except Exception:
        logger.exception("Không kiểm tra được thời điểm quét gần nhất.")
        return True


def daily_scan_due(now, last_run_date=None):
    """Quyết định quét D1 dựa trên giờ địa phương và ngày chạy đã lưu."""
    if now.weekday() >= 5 or now.time() < DAILY_SCAN_TIME:
        return False
    return last_run_date != now.date().isoformat()


def should_run_daily_scan():
    """Chỉ quét tự động sau 15:35 các ngày trong tuần."""
    now = datetime.now(VN_TZ)
    try:
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            row = conn.execute(
                "SELECT run_date FROM worker_daily_runs WHERE job_name = 'd1_scan'"
            ).fetchone()
        last_run_date = row[0] if row else None
        return daily_scan_due(now, last_run_date)
    except sqlite3.OperationalError:
        # Bảng chưa tồn tại ở lần chạy đầu.
        return daily_scan_due(now, None)
    except Exception:
        logger.exception("Không kiểm tra được trạng thái quét D1 hằng ngày.")
        return daily_scan_due(now, None)


def mark_daily_scan_complete(rows_saved=0):
    """Ghi nhận đã thử quét D1 hôm nay, kể cả khi không có dòng kết quả."""
    with sqlite3.connect(DB_PATH, timeout=30) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_daily_runs (
                job_name TEXT PRIMARY KEY,
                run_date TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                rows_saved INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            INSERT INTO worker_daily_runs(job_name, run_date, completed_at, rows_saved)
            VALUES ('d1_scan', ?, ?, ?)
            ON CONFLICT(job_name) DO UPDATE SET
                run_date = excluded.run_date,
                completed_at = excluded.completed_at,
                rows_saved = excluded.rows_saved
        """, (
            datetime.now(VN_TZ).date().isoformat(),
            datetime.now(VN_TZ).isoformat(timespec="seconds"),
            int(rows_saved or 0),
        ))


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
        "kbs_quote_price",
        "realtime_volume",
        "quote_source_time",
        "quote_fetched_at",
        "quote_age_seconds",
        "quote_source",
        "price_data_status",
        "price_data_message",
        "live_change_pct",
        "market_regime",
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


class ScanExecutionError(RuntimeError):
    """Raised when a scan could not complete and persist a result batch."""


def _run_scan_unlocked(progress_callback=None):
    """
    Một lượt quét gồm hai giai đoạn:

    1. Lấy bảng giá KBS.
    2. Phân tích chỉ báo trên dữ liệu nến D1.

    Chỉ giá có timestamp nguồn hợp lệ, mới và trong giờ giao dịch
    mới được scanner chấp nhận là giá realtime.
    """

    started = time.time()

    now_vn = datetime.now(VN_TZ)
    if (
        now_vn.weekday() < 5
        and now_vn.time() >= DAILY_SCAN_TIME
    ):
        clear_history_cache()

    logger.info(
        "Bắt đầu quét HOSE/HNX: bảng giá KBS + phân tích D1."
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
        raise ScanExecutionError("Không lấy được danh sách cổ phiếu; lượt quét chưa hoàn tất.")

    total = len(symbols)

    logger.info(
        "Chuẩn bị quét %s mã.",
        total
    )

    # GIAI ĐOẠN 1: LẤY BẢNG GIÁ

    realtime_quotes = {}

    _emit_progress(
        progress_callback,
        2,
        f"Đang lấy bảng giá cho {total} mã..."
    )

    def quote_progress(current, quote_total):
        percent = (
            5 + 25 * current / max(quote_total, 1)
        )
        _emit_progress(
            progress_callback,
            percent,
            f"Đang lấy bảng giá: {current}/{quote_total} mã"
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

        verified_count = sum(
            1
            for quote in realtime_quotes.values()
            if (
                isinstance(quote, dict)
                and quote.get("quote_is_fresh") is True
            )
        )

        coverage = (
            quote_count / total * 100
            if total else 0
        )

        logger.info(
            "Nhận phản hồi bảng giá: %s/%s mã (%.1f%%).",
            quote_count,
            total,
            coverage
        )

        logger.info(
            "Nguồn đánh dấu giá mới: %s mã.",
            verified_count
        )

        if quote_count < total:
            logger.warning(
                "Một số mã không có dữ liệu bảng giá."
            )

    except RealtimeDataError as exc:
        logger.warning(
            "Nguồn giá KBS không khả dụng: %s",
            exc
        )

    except Exception:
        logger.exception(
            "Lỗi khi lấy dữ liệu bảng giá."
        )

    quote_count = len(realtime_quotes)

    _emit_progress(
        progress_callback,
        30,
        f"Đã nhận dữ liệu bảng giá cho {quote_count}/{total} mã. "
        "Bắt đầu phân tích D1..."
    )

    # GIAI ĐOẠN 2: PHÂN TÍCH D1

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

        if not report.get("market_data_available"):
            logger.warning(
                "Không lấy được dữ liệu VN-Index: %s. "
                "Không thể áp dụng đầy đủ bộ lọc thị trường.",
                report.get("market_data_error"),
            )

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
            "Nhận bảng giá: %s/%s | "
            "Realtime xác minh: %s | "
            "Giá phiên gần nhất: %s | "
            "Chưa xác minh: %s | "
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
            report.get("realtime_count", 0),
            report.get("last_session_count", 0),
            report.get("unverified_quote_count", 0),
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

        raise ScanExecutionError("Lượt quét thất bại; chưa ghi nhận hoàn tất.") from None




class ScanAlreadyRunningError(RuntimeError):
    """Raised when another process currently owns the market scan lock."""


def run_scan(progress_callback=None):
    """
    Chạy quét với khóa SQLite dùng chung giữa giao diện và worker.
    Chỉ một tiến trình được quét tại một thời điểm.
    """
    with scan_lock(DB_PATH) as acquired:
        if not acquired:
            raise ScanAlreadyRunningError(
                "Một lượt quét khác đang chạy. Vui lòng đợi lượt đó hoàn tất."
            )

        return _run_scan_unlocked(
            progress_callback=progress_callback
        )


def should_run_daily_backup():
    """Tạo một bản sao lưu SQLite sau khi hoàn tất lượt D1 cuối ngày."""
    now = datetime.now(VN_TZ)
    if now.weekday() >= 5 or now.time() < DAILY_BACKUP_TIME:
        return False
    try:
        from pathlib import Path
        backups = sorted(
            Path(BACKUP_DIR).glob(f"{os.path.splitext(os.path.basename(DB_PATH))[0]}_*.sqlite3"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not backups:
            return True
        latest = datetime.fromtimestamp(backups[0].stat().st_mtime, VN_TZ)
        return latest.date() != now.date()
    except Exception:
        logger.exception("Không kiểm tra được bản sao lưu gần nhất.")
        return True


def run_daily_backup():
    path = backup_database(
        DB_PATH,
        backup_dir=BACKUP_DIR,
        keep=BACKUP_RETENTION,
    )
    logger.info("Đã tạo bản sao lưu SQLite: %s", path)
    return path


def run_worker():
    logger.info("AI Stock Scanner Worker đã khởi động; điều khiển qua Supabase.")
    last_continuous_scan = 0.0
    last_scan_iso = None
    last_daily_date = None

    while True:
        loop_started = time.monotonic()
        try:
            control = get_control()
            mode = control.get("mode", "on_demand")
            requested = control.get("manual_scan_requested_at")
            completed = control.get("manual_scan_completed_at")
            manual_pending = bool(requested and requested != completed)
            now = datetime.now(VN_TZ)
            do_scan = manual_pending
            reason = "yêu cầu quét thủ công từ giao diện" if manual_pending else ""

            if mode == "continuous" and is_market_session(now):
                if loop_started - last_continuous_scan >= SCAN_INTERVAL:
                    do_scan = True
                    reason = "chế độ chạy liên tục"
            if mode == "continuous" and daily_scan_due(now, last_daily_date):
                do_scan = True
                reason = "quét D1 cuối ngày"

            if do_scan:
                logger.info("Bắt đầu quét: %s", reason)
                worker_heartbeat(status="running", last_scan_at=last_scan_iso)
                saved = run_scan()
                last_scan_iso = datetime.now(VN_TZ).isoformat(timespec="seconds")
                last_continuous_scan = time.monotonic()
                if reason == "quét D1 cuối ngày":
                    last_daily_date = now.date().isoformat()
                    mark_daily_scan_complete(saved)
                if manual_pending:
                    complete_manual_scan(requested)
                worker_heartbeat(status="idle", last_scan_at=last_scan_iso,
                                 symbols_monitored=MAX_SYMBOLS or None)
                logger.info("Quét hoàn tất; đã lưu %s dòng.", saved)
            else:
                worker_heartbeat(status="idle", last_scan_at=last_scan_iso,
                                 symbols_monitored=MAX_SYMBOLS or None)

            if mode == "continuous" and not is_market_session(now) and should_run_daily_backup():
                run_daily_backup()

        except Exception as exc:
            logger.exception("Worker gặp lỗi không dự kiến.")
            try:
                worker_heartbeat(status="error", last_scan_at=last_scan_iso, error=exc)
            except Exception:
                logger.exception("Không cập nhật được heartbeat.")

        elapsed = time.monotonic() - loop_started
        time.sleep(max(3, min(10, 10 - elapsed)))


if __name__ == "__main__":
    run_worker()
