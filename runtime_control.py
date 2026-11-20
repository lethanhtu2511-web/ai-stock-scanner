"""Supabase-backed global runtime control for the Windows scanner worker."""
from __future__ import annotations
from datetime import datetime, timezone
import os
import socket
import uuid
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _client():
    try:
        from portfolio import _get_supabase
        return _get_supabase()
    except Exception:
        return None

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def get_control():
    client = _client()
    if client is None:
        return {"mode": "on_demand", "worker_status": "offline",
                "error": "Chưa cấu hình Supabase trên môi trường này."}
    rows = client.table("runtime_control").select("*").eq("id", 1).limit(1).execute().data or []
    if not rows:
        raise RuntimeError("Không tìm thấy runtime_control id=1.")
    return rows[0]

def set_mode(mode):
    if mode not in {"continuous", "on_demand"}:
        raise ValueError("Mode không hợp lệ.")
    client = _client()
    if client is None:
        raise RuntimeError("Chưa cấu hình Supabase.")
    client.table("runtime_control").update({
        "mode": mode, "updated_at": utc_now()
    }).eq("id", 1).execute()
    return get_control()

def request_manual_scan():
    client = _client()
    if client is None:
        raise RuntimeError("Chưa cấu hình Supabase.")
    requested = utc_now()
    client.table("runtime_control").update({
        "manual_scan_requested_at": requested,
        "updated_at": requested,
    }).eq("id", 1).execute()
    return requested

def worker_heartbeat(status="idle", last_scan_at=None, last_quote_at=None,
                     symbols_monitored=None, error=None):
    client = _client()
    if client is None:
        return False
    worker_id = os.getenv("SCANNER_WORKER_ID") or socket.gethostname()
    payload = {
        "worker_id": worker_id,
        "worker_status": status,
        "heartbeat_at": utc_now(),
        "updated_at": utc_now(),
        "last_error": str(error)[:1000] if error else None,
    }
    if last_scan_at is not None:
        payload["last_scan_at"] = last_scan_at
    if last_quote_at is not None:
        payload["last_quote_at"] = last_quote_at
    if symbols_monitored is not None:
        payload["symbols_monitored"] = int(symbols_monitored)
    client.table("runtime_control").update(payload).eq("id", 1).execute()
    return True

def complete_manual_scan(requested_at):
    client = _client()
    if client is None:
        return False
    client.table("runtime_control").update({
        "manual_scan_completed_at": requested_at,
        "updated_at": utc_now(),
        "worker_status": "idle",
        "last_error": None,
    }).eq("id", 1).execute()
    return True
