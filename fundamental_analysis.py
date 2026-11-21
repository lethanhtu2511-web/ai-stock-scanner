"""Transparent fundamental-quality checks from explicitly supplied company data.

No data vendor is called here. Missing or stale fields remain unavailable and
cannot silently improve a company's score.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any


def assess_fundamentals(metrics: dict[str, Any] | None, *, as_of=None, max_age_days: int = 120) -> dict:
    m = metrics if isinstance(metrics, dict) else {}
    required = ("revenue_growth_yoy", "net_profit_growth_yoy", "roe", "debt_to_equity", "operating_cash_flow")
    missing = [key for key in required if m.get(key) is None]
    data_date = m.get("period_end")
    age_days = None
    if data_date:
        try:
            end = datetime.fromisoformat(str(data_date)[:10]).date()
            today = as_of if isinstance(as_of, date) else date.today()
            age_days = (today - end).days
        except (TypeError, ValueError):
            missing.append("period_end")
    else:
        missing.append("period_end")
    stale = age_days is not None and (age_days < 0 or age_days > max_age_days)
    if stale:
        missing.append("freshness")
    if missing:
        return {
            "status": "UNAVAILABLE" if len(missing) == len(required) + 1 else "INCOMPLETE",
            "score": None, "coverage_pct": round((len(required) - sum(k in missing for k in required)) / len(required) * 100, 1),
            "missing_fields": list(dict.fromkeys(missing)), "period_end": data_date,
            "age_days": age_days,
            "note": "Không chấm điểm cơ bản khi thiếu dữ liệu trọng yếu hoặc kỳ báo cáo đã cũ."
        }

    score, positives, risks = 0, [], []
    revenue = float(m["revenue_growth_yoy"])
    profit = float(m["net_profit_growth_yoy"])
    roe = float(m["roe"])
    debt = float(m["debt_to_equity"])
    cash = float(m["operating_cash_flow"])
    if revenue > 0:
        score += 15; positives.append("Doanh thu tăng trưởng")
    else:
        risks.append("Doanh thu suy giảm")
    if profit > 0:
        score += 25; positives.append("Lợi nhuận sau thuế tăng trưởng")
    else:
        risks.append("Lợi nhuận suy giảm")
    if roe >= 15:
        score += 25; positives.append("ROE đạt từ 15%")
    elif roe < 8:
        risks.append("ROE thấp dưới 8%")
    else:
        score += 10
    if debt <= 1:
        score += 20; positives.append("Đòn bẩy nợ ở mức vừa phải")
    elif debt >= 2:
        risks.append("Đòn bẩy nợ cao")
    else:
        score += 10
    if cash > 0:
        score += 15; positives.append("Dòng tiền kinh doanh dương")
    else:
        risks.append("Dòng tiền kinh doanh âm")
    return {
        "status": "AVAILABLE", "score": score, "coverage_pct": 100.0,
        "period_end": data_date, "age_days": age_days,
        "positives": positives, "risks": risks,
        "note": "Điểm chất lượng tham khảo từ bộ chỉ tiêu tối thiểu; không thay thế đọc báo cáo tài chính và không tạo tín hiệu mua."
    }
