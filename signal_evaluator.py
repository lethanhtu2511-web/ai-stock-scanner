"""Historical self-evaluation for scanner signals.

Compares a signal-day completed D1 close with later completed D1 closes.
This is a research metric, NOT an executable fill or a win-probability estimate.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime


HORIZONS = (1, 3, 5, 10)


def _session_date(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date()
    raw = str(value).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw[:19], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _positive(value):
    try:
        number = float(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def evaluate_signal_history(rows):
    """Return aggregate forward-close outcomes by strategy, signal type and horizon.

    One prediction is counted per symbol/strategy/type/session. Repeated 5-minute
    app scans do not inflate the sample. Outcomes use later available D1 closes.
    """
    rows = list(rows or [])
    daily_closes = defaultdict(dict)
    candidates = {}

    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        strategy = str(row.get("strategy") or "").strip().upper()
        session = _session_date(row.get("data_date"))
        close = _positive(row.get("d1_close") or row.get("price"))
        if not symbol or not strategy or session is None or close is None:
            continue

        key = (symbol, strategy)
        daily_closes[key][session] = close

        signal = str(row.get("signal") or "").upper()
        early = str(row.get("early_signal") or "").upper()
        signal_types = []
        if signal == "BUY":
            signal_types.append("BUY")
        if early in {"EARLY_BREAKOUT", "PREPARE"}:
            signal_types.append("EARLY")

        for signal_type in signal_types:
            candidate_key = (symbol, strategy, signal_type, session)
            scanned_at = str(row.get("scanned_at") or "")
            existing = candidates.get(candidate_key)
            # Keep the first observation of the setup on that session.
            if existing is None or scanned_at < existing["scanned_at"]:
                candidates[candidate_key] = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "signal_type": signal_type,
                    "session": session,
                    "entry_close": close,
                    "scanned_at": scanned_at,
                }

    grouped = defaultdict(list)
    for prediction in candidates.values():
        grouped[(prediction["strategy"], prediction["signal_type"])].append(prediction)

    results = []
    for (strategy, signal_type), predictions in sorted(grouped.items()):
        for horizon in HORIZONS:
            returns = []
            for prediction in predictions:
                closes = daily_closes[(prediction["symbol"], prediction["strategy"])]
                sessions = sorted(day for day in closes if day >= prediction["session"])
                try:
                    target_day = sessions[sessions.index(prediction["session"]) + horizon]
                except (ValueError, IndexError):
                    continue
                exit_close = closes[target_day]
                returns.append((exit_close / prediction["entry_close"] - 1) * 100)

            if not returns:
                continue
            results.append({
                "strategy": strategy,
                "signal_type": signal_type,
                "horizon": horizon,
                "sample_size": len(returns),
                "win_rate_pct": round(sum(value > 0 for value in returns) / len(returns) * 100, 1),
                "average_return_pct": round(sum(returns) / len(returns), 2),
                "median_return_pct": round(sorted(returns)[len(returns) // 2], 2),
                "best_return_pct": round(max(returns), 2),
                "worst_return_pct": round(min(returns), 2),
            })

    total_predictions = len(candidates)
    matured = sum(item["sample_size"] for item in results if item["horizon"] == 5)
    return {
        "predictions_recorded": total_predictions,
        "five_session_matured": matured,
        "open_predictions": max(0, total_predictions - matured),
        "metrics": results,
        "method_note": (
            "Đối chiếu giá đóng cửa D1 ngày phát tín hiệu với giá đóng cửa sau "
            "1/3/5/10 phiên; loại trùng các lần quét trong cùng ngày. Chưa tính "
            "giá khớp lệnh, phí/thuế, trượt giá, đường đi intraday hoặc việc chạm "
            "stop/target trước ngày đo."
        ),
    }
