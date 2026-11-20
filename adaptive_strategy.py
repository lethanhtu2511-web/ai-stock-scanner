"""Research-only adaptive scoring layer; never overrides production BUY signals.

The baseline strategy engine remains authoritative until this profile passes
out-of-sample validation. Adaptive scores are ranking/diagnostic outputs only.
"""
from __future__ import annotations

HARD_RISK_TERMS = (
    "biến động", "atr", "thanh khoản", "gtgd", "bán ròng mạnh",
    "reward/risk", "tỷ lệ lợi nhuận/rủi ro", "dữ liệu",
    "realtime", "quote", "mua đuổi", "áp lực bán",
)

HARD_RISK_PHRASES = ("phá đáy", "mất ema50", "đã mất ema50")

SOFT_CONFIRMATION_TERMS = (
    "rsi", "macd", "khối lượng", "nến", "ema20", "ema50",
    "obv", "mfi", "pullback", "hồi phục", "xu hướng",
    "đường trung bình", "cấu trúc giá", "chưa lấy lại",
    "cách xa ema20", "chưa ở vùng pullback",
)


def classify_market_regime(market_df):
    """Classify only from bars available at the evaluation timestamp."""
    import pandas as pd

    if not isinstance(market_df, pd.DataFrame) or market_df.empty:
        return "UNKNOWN"
    d = market_df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "close" not in d:
        return "UNKNOWN"
    close = pd.to_numeric(d["close"], errors="coerce").dropna()
    if len(close) < 50:
        return "UNKNOWN"
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma20_prev5 = float(close.rolling(20).mean().iloc[-6])
    latest = float(close.iloc[-1])
    if latest > ma20 > ma50 and ma20 > ma20_prev5:
        return "BULLISH"
    if latest < ma20 < ma50 and ma20 < ma20_prev5:
        return "BEARISH"
    return "NEUTRAL"


def adaptive_assessment(detail, *, liquidity_ok=True, risk_plan=None,
                        atr_pct=None, foreign_net_ratio=None,
                        market_regime="UNKNOWN", strategy="SWING"):
    """Return an experimental score/status without producing an executable BUY."""
    blockers = list(detail.get("blockers", []) or [])
    reasons = list(detail.get("reasons", []) or [])
    hard = []
    soft = []
    for blocker in blockers:
        lower = str(blocker).lower()
        if any(phrase in lower for phrase in HARD_RISK_PHRASES):
            hard.append(str(blocker))
        elif any(term in lower for term in HARD_RISK_TERMS):
            hard.append(str(blocker))
        elif any(term in lower for term in SOFT_CONFIRMATION_TERMS):
            soft.append(str(blocker))
        else:
            # Fail closed: unknown blockers are never silently downgraded.
            hard.append(str(blocker))

    if not liquidity_ok:
        hard.append("Không đạt bộ lọc thanh khoản")
    if risk_plan is None or float(risk_plan.get("reward_risk") or 0) < 2.0:
        hard.append("Không đạt reward/risk tối thiểu 2.0")
    if atr_pct is not None and (float(atr_pct) <= 0 or float(atr_pct) > 7):
        hard.append("ATR ngoài ngưỡng an toàn")
    if foreign_net_ratio is not None and float(foreign_net_ratio) <= -0.02:
        hard.append("Khối ngoại bán ròng mạnh 5 phiên")

    score = max(0, min(100, int(detail.get("score", 0) or 0)))
    # Soft confirmations reduce conviction, but do not automatically veto a setup.
    score = max(0, score - min(24, 6 * len(soft)))
    if market_regime == "BULLISH":
        score = min(100, score + (4 if strategy in {"SWING", "PULLBACK"} else 0))
    elif market_regime == "BEARISH":
        score = max(0, score - (8 if strategy in {"SWING", "PULLBACK"} else 3))

    if hard:
        status = "RISK_BLOCKED"
    elif score >= 72 and len(reasons) >= 3:
        status = "HIGH_CONVICTION_SETUP"
    elif score >= 58:
        status = "SETUP_WATCH"
    else:
        status = "LOW_CONVICTION"

    return {
        "profile": "ADAPTIVE_RESEARCH_V1",
        "score": score,
        "status": status,
        "executable_buy": False,
        "hard_risks": list(dict.fromkeys(hard)),
        "soft_confirmations_missing": list(dict.fromkeys(soft)),
        "market_regime": market_regime,
        "note": "Điểm nghiên cứu; chưa phải tín hiệu mua và không thay thế bộ quy tắc production.",
    }
