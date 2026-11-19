"""SQLite online backup and guarded restore helpers for the scanner database."""
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_RETENTION = 14


def _integrity_ok(path):
    conn = sqlite3.connect(str(path), timeout=30)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        return bool(result and result[0] == "ok")
    finally:
        conn.close()


def backup_database(db_path, backup_dir=None, keep=DEFAULT_RETENTION):
    """Create a consistent SQLite online backup, then atomically publish it."""
    source = Path(db_path)
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy database: {source}")
    if keep < 1:
        raise ValueError("Số bản sao lưu giữ lại phải ít nhất là 1.")

    target_dir = Path(backup_dir) if backup_dir else source.parent / "backups"
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
    final_path = target_dir / f"{source.stem}_{stamp}.sqlite3"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{source.stem}_", suffix=".tmp", dir=str(target_dir)
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with sqlite3.connect(str(source), timeout=30) as src:
            with sqlite3.connect(str(temp_path), timeout=30) as dst:
                src.backup(dst)
        if not _integrity_ok(temp_path):
            raise sqlite3.DatabaseError("Bản sao lưu không vượt qua integrity_check.")
        os.replace(temp_path, final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    backups = sorted(
        target_dir.glob(f"{source.stem}_*.sqlite3"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for obsolete in backups[keep:]:
        obsolete.unlink(missing_ok=True)

    return str(final_path)


def restore_database(backup_path, db_path, confirm=False):
    """Restore only after explicit confirmation; caller must stop app/worker."""
    if not confirm:
        raise PermissionError(
            "Khôi phục sẽ thay thế database hiện tại. "
            "Hãy dừng app/worker và truyền confirm=True sau khi xác nhận."
        )

    backup = Path(backup_path)
    target = Path(db_path)
    if not backup.is_file():
        raise FileNotFoundError(f"Không tìm thấy bản sao lưu: {backup}")
    if not _integrity_ok(backup):
        raise sqlite3.DatabaseError("Bản sao lưu bị lỗi, từ chối khôi phục.")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.stem}_restore_", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)

    try:
        with sqlite3.connect(str(backup), timeout=30) as src:
            with sqlite3.connect(str(temp_path), timeout=30) as dst:
                src.backup(dst)
        if not _integrity_ok(temp_path):
            raise sqlite3.DatabaseError("Database phục hồi không hợp lệ.")
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return str(target)
