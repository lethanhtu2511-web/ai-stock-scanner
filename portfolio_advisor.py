"""Conservative, explainable sell/hold recommendations for existing positions.

Recommendations are advisory only. Prices supplied by the scanner are in VND
after the caller converts the scanner's thousand-VND price fields.
"""
from math import floor


def _number(value):
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _partial_quantity(quantity):
    """Sell about half, rounded down to a 100-share board lot."""
    lots = floor((int(quantity) * 0.5) / 100)
    return lots * 100


def advise_position(position, market, strategy_rows, settings):
    """Return a deterministic sell/hold recommendation for one holding."""
    quantity = int(position.get("quantity") or 0)
    avg_cost = _number(position.get("avg_cost"))
    price = _number(market.get("price"))
    status = market.get("price_status", "UNKNOWN")
    rows = [row for row in strategy_rows if isinstance(row, dict)]
    result = {
        "action": "REVIEW",
        "label": "CHỜ ĐÁNH GIÁ",
        "quantity": 0,
        "reference_price": price,
        "reason": "",
        "data_status": status,
        "pnl_pct": None,
        "stop_price": None,
        "target_price": None,
    }
    if quantity <= 0:
        result["reason"] = "Không có số lượng cổ phiếu đang nắm giữ."
        return result
    if avg_cost is None or avg_cost <= 0 or price is None or price <= 0:
        result["reason"] = "Thiếu giá vốn hoặc giá thị trường hợp lệ; không đưa ra lệnh bán."
        return result
    if status == "D1_STALE":
        result["reason"] = "Dữ liệu D1 đã cũ; cần cập nhật giá trước khi quyết định."
        return result

    fee_sell = float(settings.get("fee_sell", 0.0015))
    sell_tax = float(settings.get("sell_tax", 0.001))
    net_price = price * (1 - fee_sell - sell_tax)
    result["pnl_pct"] = round((net_price / avg_cost - 1) * 100, 2)

    valid_stops = [
        _number(row.get("stop_loss")) * 1000
        for row in rows if _number(row.get("stop_loss")) is not None
    ]
    valid_targets = [
        _number(row.get("target_price")) * 1000
        for row in rows if _number(row.get("target_price")) is not None
    ]
    stop = max(valid_stops) if valid_stops else None
    target = min(valid_targets) if valid_targets else None
    result["stop_price"] = stop
    result["target_price"] = target

    # Risk protection takes priority over every other signal.
    if stop is not None and price <= stop:
        result.update({
            "action": "SELL_ALL",
            "label": "BÁN HẾT",
            "quantity": quantity,
            "reason": (
                f"Giá hiện tại đã chạm/vượt xuống dưới ngưỡng cắt lỗ "
                f"tham chiếu {stop:,.0f} đ; ưu tiên bảo vệ vốn."
            ),
        })
        return result

    # Take partial profit when price reaches a strategy target.
    if target is not None and price >= target:
        partial = _partial_quantity(quantity)
        if partial <= 0:
            result.update({
                "action": "SELL_ALL", "label": "BÁN HẾT", "quantity": quantity,
                "reason": "Giá đã chạm mục tiêu; vị thế chỉ còn một lô nên không thể bán một phần theo lô 100 cp.",
            })
        else:
            result.update({
                "action": "SELL_PART", "label": "BÁN MỘT PHẦN",
                "quantity": partial,
                "reason": (
                    f"Giá đã chạm/vượt mục tiêu tham chiếu {target:,.0f} đ; "
                    f"cân nhắc chốt khoảng một nửa vị thế, phần còn lại tiếp tục theo dõi."
                ),
            })
        return result

    # A broad deterioration signal is required; one weak strategy is not enough.
    weak_rows = [
        row for row in rows
        if row.get("signal") in ("SKIP", "FILTERED")
        and (_number(row.get("score")) or 0) < 50
    ]
    regime_values = {row.get("market_regime") for row in rows}
    bearish = "BEARISH" in regime_values
    if bearish and len(weak_rows) >= 2:
        partial = _partial_quantity(quantity)
        if partial <= 0:
            result.update({
                "action": "SELL_ALL", "label": "BÁN HẾT", "quantity": quantity,
                "reason": "Cả bối cảnh thị trường và các chiến lược đều suy yếu; vị thế chỉ còn một lô 100 cp.",
            })
        else:
            result.update({
                "action": "SELL_PART", "label": "BÁN MỘT PHẦN",
                "quantity": partial,
                "reason": (
                    "VN-Index đang ở trạng thái BEARISH và ít nhất hai chiến lược "
                    "đánh giá mã yếu (SKIP/FILTERED, dưới 50 điểm); cân nhắc giảm một nửa vị thế."
                ),
            })
        return result

    if not rows:
        result["reason"] = "Chưa có kết quả phân tích mới cho mã này; tiếp tục theo dõi, chưa tự động đề xuất bán."
    elif status in ("UNVERIFIED", "D1_ONLY", "LAST_SESSION"):
        result["action"] = "HOLD"
        result["label"] = "GIỮ / CHỜ XÁC NHẬN"
        result["reason"] = "Chưa xác minh được giá realtime; hiện chưa có điều kiện bán rõ ràng. Kiểm tra lại khi có giá mới."
    else:
        result["action"] = "HOLD"
        result["label"] = "GIỮ"
        result["reason"] = "Chưa chạm ngưỡng cắt lỗ/mục tiêu và chưa có sự suy yếu đồng thuận từ các chiến lược."
    return result
