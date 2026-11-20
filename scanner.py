
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from strategy_engine import (
    analyze_stock as analyze_strategy_stock,
    STRATEGIES,
)
from early_signal_engine import evaluate_early_signal
from foreign_flow_provider import summarize_foreign_flow
from adaptive_strategy import adaptive_assessment, classify_market_regime

# =====================================================
# AI STOCK SCANNER 3.0 - D1 + VERIFIED REALTIME
# =====================================================

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

MIN_HISTORY = 200
MIN_AVG_TURNOVER = 5_000_000_000
MIN_MEDIAN_TURNOVER = 3_000_000_000
MIN_RR = 2.0

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

MAX_QUOTE_AGE_SECONDS = 300

BLOCKED = {
    "NVB": "Mã bị chặn thủ công",
}

RISK_CONFIG = {
    "SWING": {
        "stop_atr": 1.5,
        "target_atr": 3.5,
    },
    "REVERSAL": {
        "stop_atr": 1.8,
        "target_atr": 4.0,
    },
    "PULLBACK": {
        "stop_atr": 1.3,
        "target_atr": 3.0,
    },
}


# =====================================================
# UTILITIES
# =====================================================

def _number(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _parse_datetime(value):
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            numeric = float(value)

            if numeric > 10_000_000_000:
                numeric /= 1000.0

            return datetime.fromtimestamp(
                numeric,
                tz=VN_TZ,
            )

        if isinstance(value, str):
            raw = value.strip()

            if not raw:
                return None

            if raw.isdigit():
                numeric = float(raw)

                if numeric > 10_000_000_000:
                    numeric /= 1000.0

                return datetime.fromtimestamp(
                    numeric,
                    tz=VN_TZ,
                )

            parsed = datetime.fromisoformat(
                raw.replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=VN_TZ)
            else:
                parsed = parsed.astimezone(VN_TZ)

            return parsed

    except (TypeError, ValueError, OverflowError, OSError):
        return None

    return None


def _is_market_session(now):
    if now.weekday() >= 5:
        return False

    current_time = now.time()

    morning = (
        current_time.hour > 9
        or (
            current_time.hour == 9
            and current_time.minute >= 0
        )
    ) and (
        current_time.hour < 11
        or (
            current_time.hour == 11
            and current_time.minute <= 30
        )
    )

    afternoon = (
        current_time.hour > 13
        or (
            current_time.hour == 13
            and current_time.minute >= 0
        )
    ) and (
        current_time.hour < 15
        or (
            current_time.hour == 15
            and current_time.minute == 0
        )
    )

    return morning or afternoon


def _validate_history(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dữ liệu D1 không hợp lệ.")

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    required = [
        "Date", "Open", "High",
        "Low", "Close", "Volume",
    ]

    missing = [
        c for c in required
        if c not in d.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu cột dữ liệu: {missing}"
        )

    d["Date"] = pd.to_datetime(
        d["Date"],
        errors="coerce",
    )

    for col in required[1:]:
        d[col] = pd.to_numeric(
            d[col],
            errors="coerce",
        )

    d = (
        d.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    valid = (
        (d["Open"] > 0)
        & (d["Close"] > 0)
        & (
            d["High"]
            >= d[["Open", "Close", "Low"]].max(axis=1)
        )
        & (
            d["Low"]
            <= d[["Open", "Close"]].min(axis=1)
        )
        & (d["Volume"] >= 0)
    )

    d = d.loc[valid].reset_index(drop=True)

    if len(d) < MIN_HISTORY:
        raise ValueError(
            f"INSUFFICIENT_HISTORY: "
            f"{len(d)}/{MIN_HISTORY} phiên D1"
        )

    return d


# =====================================================
# REALTIME QUOTE VALIDATION
# =====================================================

def _validate_realtime_quote(symbol, quote):
    """
    Chỉ xác nhận realtime khi:
    - Mã cổ phiếu trùng khớp.
    - Giá hợp lệ.
    - Nguồn đánh dấu quote_is_fresh=True.
    - Có source_time hợp lệ.
    - Timestamp không quá 5 phút.
    - Timestamp không nằm trong tương lai.
    - Hiện tại và timestamp đều thuộc khung giờ giao dịch.

    fetched_at chỉ là thời điểm bot nhận dữ liệu,
    không được dùng thay source_time.
    """

    if not isinstance(quote, dict):
        return None

    quote_symbol = str(
        quote.get("symbol", "")
    ).strip().upper()

    if quote_symbol != symbol:
        return None

    price = _number(
        quote.get("price"),
        default=0.0,
    )

    if price <= 0:
        return None

    if quote.get("quote_is_fresh") is not True:
        return None

    source_time = _parse_datetime(
        quote.get("source_time_parsed")
        or quote.get("source_time")
    )

    if source_time is None:
        return None

    now = datetime.now(VN_TZ)

    age_seconds = (
        now - source_time
    ).total_seconds()

    if age_seconds < -30:
        return None

    if age_seconds > MAX_QUOTE_AGE_SECONDS:
        return None

    if not _is_market_session(now):
        return None

    if not _is_market_session(source_time):
        return None

    return {
        **quote,
        "symbol": symbol,
        "price": price,
        "source_time_parsed": source_time.isoformat(),
        "quote_age_seconds": round(age_seconds, 1),
        "quote_is_fresh": True,
    }


def _quote_display_status(quote):
    if not isinstance(quote, dict):
        return (
            "D1_ONLY",
            "Không có báo giá KBS hợp lệ; đang dùng giá D1.",
        )

    freshness = quote.get("quote_freshness")

    if freshness == "OUTSIDE_SESSION":
        return (
            "LAST_SESSION",
            "Giá bảng KBS ngoài giờ giao dịch; "
            "không xác nhận là giá realtime.",
        )

    if freshness == "STALE":
        return (
            "LAST_SESSION",
            "Báo giá KBS đã cũ; không xác nhận là giá realtime.",
        )

    if freshness in ("UNKNOWN", "INVALID_TIME"):
        return (
            "UNVERIFIED",
            "Không xác minh được thời điểm cập nhật từ KBS.",
        )

    return (
        "UNVERIFIED",
        "Báo giá KBS chưa đáp ứng điều kiện xác nhận realtime.",
    )


# =====================================================
# RISK / REWARD
# =====================================================

def _net_rr(price, stop, target):
    """
    Tính tỷ lệ lợi nhuận/rủi ro sau phí,
    thuế bán và trượt giá giả định.
    """

    cost = price * (
        1 + BUY_FEE + SLIPPAGE
    )

    stop_net = stop * (
        1 - SELL_FEE - SELL_TAX - SLIPPAGE
    )

    target_net = target * (
        1 - SELL_FEE - SELL_TAX - SLIPPAGE
    )

    risk = cost - stop_net
    reward = target_net - cost

    if risk <= 0 or reward <= 0:
        return 0.0

    return reward / risk


def _risk_plan(price, atr, resistance, strategy):
    config = RISK_CONFIG.get(strategy)

    if not config or price <= 0 or atr <= 0:
        return None

    stop = round(
        price - config["stop_atr"] * atr
    )

    atr_target = (
        price + config["target_atr"] * atr
    )

    if resistance > price:
        target = round(
            min(resistance, atr_target)
        )
    else:
        target = round(atr_target)

    if stop <= 0 or target <= price:
        return None

    rr = _net_rr(
        price,
        stop,
        target,
    )

    return {
        "stop_loss": stop,
        "target_price": target,
        "reward_risk": round(rr, 2),
        "stop_distance_pct": round(
            (price - stop) / price * 100,
            2,
        ),
        "target_distance_pct": round(
            (target - price) / price * 100,
            2,
        ),
    }


# =====================================================
# SIGNAL
# =====================================================

def _strategy_signal(detail, risk_plan):
    signal = detail.get(
        "signal",
        "BỎ QUA",
    )

    blockers = list(
        detail.get("blockers", [])
    )

    if signal == "MUA":
        if risk_plan is None:
            blockers.append(
                "Không xác định được kế hoạch rủi ro"
            )

            return "WAIT", blockers

        if risk_plan["reward_risk"] < MIN_RR:
            blockers.append(
                f"RR sau phí chỉ "
                f"{risk_plan['reward_risk']:.2f}, "
                f"yêu cầu tối thiểu {MIN_RR:.1f}"
            )

            return "WAIT", blockers

        return "BUY", blockers

    mapping = {
        "CHỜ XÁC NHẬN": "WAIT",
        "THEO DÕI": "WATCH",
        "BỎ QUA": "SKIP",
    }

    return mapping.get(signal, "WAIT"), blockers


def _make_reason(signal, blockers, reasons, penalties):
    if signal == "BUY":
        return (
            "Đạt điều kiện kỹ thuật, "
            "xác nhận chiến lược và RR sau phí"
        )

    if signal == "WAIT":
        items = blockers[:5]

        if not items:
            items = penalties[:3]

        return (
            "Chưa đủ điều kiện MUA: "
            + "; ".join(items)
            if items else "Cần thêm xác nhận"
        )

    if signal == "WATCH":
        items = reasons[:3]

        return (
            "Theo dõi thêm: "
            + "; ".join(items)
            if items else "Chưa đủ điều kiện giao dịch"
        )

    if signal == "FILTERED":
        return (
            "Không đạt bộ lọc thanh khoản hoặc dữ liệu"
        )

    return (
        "Không phù hợp: "
        + "; ".join(blockers[:4])
        if blockers else "Không đạt điều kiện chiến lược"
    )


def analyze_historical_signal(df, symbol, market_df=None, foreign_flow=None):
    """Evaluate a completed D1 bar with production technical/liquidity/RR rules.

    Unlike live scanning, this intentionally omits the realtime-quote gate:
    the historical signal is formed at the completed D1 close, then simulated
    from the next session open. It does not use future bars.
    """
    d = _validate_history(df)
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        raise ValueError("Thiếu mã cổ phiếu.")

    analysis = analyze_strategy_stock(symbol, d, market_df=market_df)
    turnover = d["Close"] * d["Volume"]
    previous_turnover = turnover.shift(1).dropna().tail(20)
    if len(previous_turnover) < 20:
        raise ValueError("Không đủ 20 phiên để đánh giá thanh khoản.")

    liquidity_blockers = []
    avg_turnover = float(previous_turnover.mean())
    median_turnover = float(previous_turnover.median())
    if avg_turnover < MIN_AVG_TURNOVER:
        liquidity_blockers.append("GTGD bình quân 20 phiên dưới 5 tỷ")
    if median_turnover < MIN_MEDIAN_TURNOVER:
        liquidity_blockers.append("GTGD trung vị 20 phiên dưới 3 tỷ")

    price = float(d.iloc[-1]["Close"])
    turnover_5d = float((d["Close"] * d["Volume"]).tail(5).sum())
    foreign = summarize_foreign_flow(
        foreign_flow,
        as_of=pd.Timestamp(d.iloc[-1]["Date"]),
        market_turnover_5d=turnover_5d,
    )
    atr = _number(analysis.get("atr"))
    resistance = _number(analysis.get("resistance"))
    regime = classify_market_regime(market_df)
    strategies = {}

    for strategy, detail in analysis["strategies"].items():
        risk_plan = _risk_plan(price, atr, resistance, strategy)
        if liquidity_blockers:
            signal = "FILTERED"
            blockers = list(detail.get("blockers", [])) + liquidity_blockers
        else:
            signal, signal_blockers = _strategy_signal(detail, risk_plan)
            blockers = list(detail.get("blockers", [])) + signal_blockers
            # Strong, persistent foreign net selling is a risk veto, not a buy trigger.
            if signal == "BUY" and foreign["foreign_net_5d"] is not None and foreign["foreign_net_ratio"] is not None and foreign["foreign_net_ratio"] <= -0.02:
                signal = "WAIT"
                blockers.append("Khối ngoại bán ròng mạnh 5 phiên; chờ áp lực bán suy yếu")
        adaptive = adaptive_assessment(
            detail,
            liquidity_ok=not bool(liquidity_blockers),
            risk_plan=risk_plan,
            atr_pct=analysis.get("atr_pct"),
            foreign_net_ratio=foreign.get("foreign_net_ratio"),
            market_regime=regime,
            strategy=strategy,
        )
        strategies[strategy] = {
            "signal": signal,
            "confirmed": signal == "BUY",
            "adaptive_research": adaptive,
            "score": max(0, min(100, int(detail.get("score", 0)) + foreign["foreign_score_adjustment"])),
            "stop_loss": risk_plan["stop_loss"] if risk_plan else None,
            "target_price": risk_plan["target_price"] if risk_plan else None,
            "reward_risk": risk_plan["reward_risk"] if risk_plan else None,
            "blockers": list(dict.fromkeys(blockers)),
            "reasons": list(detail.get("reasons", [])),
            **foreign,
        }

    return {
        "symbol": symbol,
        "price": price,
        "atr": analysis.get("atr"),
        "strategies": strategies,
        "relative_strength": analysis.get("relative_strength"),
    }


# =====================================================
# ANALYZE ONE STOCK
# =====================================================

def _classify_market_regime(market_df):
    """Phân loại xu hướng VN-Index để hiển thị; không thay đổi điểm tín hiệu."""
    if not isinstance(market_df, pd.DataFrame) or market_df.empty:
        return "UNKNOWN"

    d = market_df.copy()
    d.columns = [str(col).strip().lower() for col in d.columns]
    if "close" not in d.columns:
        return "UNKNOWN"

    close = pd.to_numeric(d["close"], errors="coerce").dropna()
    if len(close) < 50:
        return "UNKNOWN"

    latest = float(close.iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma20_prev5 = float(close.rolling(20).mean().iloc[-6])

    if latest > ma20 > ma50 and ma20 > ma20_prev5:
        return "BULLISH"
    if latest < ma20 < ma50 and ma20 < ma20_prev5:
        return "BEARISH"
    return "NEUTRAL"


def analyze_stock(
    df,
    symbol=None,
    realtime_quote=None,
    market_df=None,
    foreign_flow=None,
):
    symbol = str(
        symbol or ""
    ).strip().upper()

    if not symbol:
        raise ValueError("Thiếu mã cổ phiếu.")

    if symbol in BLOCKED:
        return [{
            "symbol": symbol,
            "strategy": strategy,
            "signal": "FILTERED",
            "score": 0,
            "reason": BLOCKED[symbol],
        } for strategy in STRATEGIES]

    d = _validate_history(df)

    last = d.iloc[-1]

    d1_close = float(last["Close"])
    data_date = pd.Timestamp(last["Date"])

    today = pd.Timestamp.now(
        tz=VN_TZ
    ).normalize().tz_localize(None)

    age = (
        today - data_date.normalize()
    ).days

    if age < 0 or age > 7:
        message = (
            f"Dữ liệu D1 đã cũ: {age} ngày"
        )

        return [{
            "symbol": symbol,
            "strategy": strategy,
            "signal": "WAIT",
            "score": 0,
            "price": round(d1_close),
            "d1_close": round(d1_close),
            "data_date": data_date.strftime("%d/%m/%Y"),
            "price_data_status": "D1_STALE",
            "price_data_message": message,
            "reason": message,
            "blockers": [message],
        } for strategy in STRATEGIES]

    quote = _validate_realtime_quote(
        symbol,
        realtime_quote,
    )

    has_realtime = quote is not None
    early_signal = evaluate_early_signal(
        d.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        }),
        quote=quote,
    )

    raw_quote = (
        realtime_quote
        if isinstance(realtime_quote, dict)
        else None
    )

    raw_quote_price = None

    if raw_quote:
        candidate = _number(
            raw_quote.get("price"),
            default=0.0,
        )

        if candidate > 0:
            raw_quote_price = candidate

    if has_realtime:
        price = quote["price"]
        price_data_status = "REALTIME"
        price_data_message = (
            "Giá realtime đã xác minh từ KBS."
        )
    else:
        price = d1_close

        price_data_status, price_data_message = (
            _quote_display_status(raw_quote)
        )

    turnover = (
        d["Close"] * d["Volume"]
    )

    previous_turnover = (
        turnover.shift(1).dropna().tail(20)
    )

    if len(previous_turnover) < 20:
        raise ValueError(
            "Không đủ 20 phiên để đánh giá thanh khoản."
        )

    avg_turnover = float(
        previous_turnover.mean()
    )

    median_turnover = float(
        previous_turnover.median()
    )

    liquidity_blockers = []

    if avg_turnover < MIN_AVG_TURNOVER:
        liquidity_blockers.append(
            "GTGD bình quân 20 phiên dưới 5 tỷ"
        )

    if median_turnover < MIN_MEDIAN_TURNOVER:
        liquidity_blockers.append(
            "GTGD trung vị 20 phiên dưới 3 tỷ"
        )

    analysis = analyze_strategy_stock(
        symbol,
        d,
        market_df=market_df,
    )
    turnover_5d = float((d["Close"] * d["Volume"]).tail(5).sum())
    foreign = summarize_foreign_flow(
        foreign_flow,
        as_of=data_date,
        market_turnover_5d=turnover_5d,
    )

    # Nhãn bối cảnh chỉ để hiển thị; không thay đổi điểm hay điều kiện BUY/WAIT.
    analysis["market_regime"] = _classify_market_regime(market_df)

    atr = _number(
        analysis.get("atr")
    )

    resistance = _number(
        analysis.get("resistance")
    )

    rows = []

    for strategy, detail in analysis["strategies"].items():
        if strategy not in RISK_CONFIG:
            continue

        score = int(
            detail.get("score", 0)
        )

        reasons = list(
            detail.get("reasons", [])
        )

        blockers = list(
            detail.get("blockers", [])
        )

        penalties = list(
            detail.get("penalties", [])
        )

        risk_plan = _risk_plan(
            price,
            atr,
            resistance,
            strategy,
        )

        if liquidity_blockers:
            signal = "FILTERED"

            blockers.extend(
                liquidity_blockers
            )

        else:
            signal, signal_blockers = _strategy_signal(
                detail,
                risk_plan,
            )

            blockers.extend(
                signal_blockers
            )

        live_change_pct = (
            (price / d1_close - 1) * 100
            if has_realtime and d1_close > 0 else None
        )

        # Chặn mua khi giá đang suy yếu hoặc tăng quá nóng.
        if signal == "BUY" and live_change_pct is not None:
            if live_change_pct <= -1.5:
                signal = "WAIT"
                blockers.append(
                    f"Giá realtime giảm {live_change_pct:.2f}% "
                    "so với giá đóng cửa D1; "
                    "động lượng trong phiên suy yếu"
                )

            elif live_change_pct >= 4.0:
                signal = "WAIT"
                blockers.append(
                    f"Giá realtime tăng {live_change_pct:.2f}% "
                    "so với giá đóng cửa D1; "
                    "rủi ro mua đuổi"
                )

        # Khối ngoại bán ròng mạnh liên tục là điều kiện cảnh báo rủi ro.
        if signal == "BUY" and foreign["foreign_net_5d"] is not None and foreign["foreign_net_ratio"] is not None and foreign["foreign_net_ratio"] <= -0.02:
            signal = "WAIT"
            blockers.append(
                "Khối ngoại bán ròng mạnh 5 phiên; chờ áp lực bán suy yếu"
            )

        # Không xác nhận BUY nếu không có quote realtime hợp lệ.
        if signal == "BUY" and not has_realtime:
            signal = "WAIT"

            blockers.append(
                "Chưa xác minh được giá realtime từ nguồn; "
                "không xác nhận tín hiệu mua"
            )

        blockers = list(
            dict.fromkeys(blockers)
        )

        reason = _make_reason(
            signal,
            blockers,
            reasons,
            penalties,
        )

        adaptive = adaptive_assessment(
            detail,
            liquidity_ok=not bool(liquidity_blockers),
            risk_plan=risk_plan,
            atr_pct=analysis.get("atr_pct"),
            foreign_net_ratio=foreign.get("foreign_net_ratio"),
            market_regime=analysis.get("market_regime", "UNKNOWN"),
            strategy=strategy,
        )

        row = {
            "symbol": symbol,
            "strategy": strategy,
            "strategy_name": STRATEGIES.get(
                strategy,
                strategy,
            ),
            "signal": signal,
            "score": max(0, min(100, score + foreign["foreign_score_adjustment"])),
            **foreign,
            "early_signal": early_signal.get("early_signal", "UNAVAILABLE"),
            "early_score": early_signal.get("early_score", 0),
            "early_reasons": early_signal.get("early_reasons", []),
            "early_risks": early_signal.get("early_risks", []),
            "early_volume_ratio": early_signal.get("early_volume_ratio"),
            "early_resistance": early_signal.get("early_resistance"),
            "early_quote_verified": early_signal.get("early_quote_verified", False),

            "price": round(price),
            "d1_close": round(d1_close),

            "realtime_price": (
                round(price)
                if has_realtime else None
            ),

            "kbs_quote_price": (
                round(raw_quote_price)
                if raw_quote_price is not None
                else None
            ),

            "realtime_volume": (
                quote.get("volume")
                if has_realtime else None
            ),

            "quote_source_time": (
                quote.get("source_time_parsed")
                if has_realtime else (
                    raw_quote.get("source_time")
                    if raw_quote else None
                )
            ),

            "quote_fetched_at": (
                raw_quote.get("fetched_at")
                if raw_quote else None
            ),

            "quote_age_seconds": (
                quote.get("quote_age_seconds")
                if has_realtime else (
                    raw_quote.get("quote_age_seconds")
                    if raw_quote else None
                )
            ),

            "quote_source": (
                raw_quote.get("source")
                if raw_quote else None
            ),

            "price_data_status": price_data_status,
            "price_data_message": price_data_message,

            "live_change_pct": (
                round(live_change_pct, 2)
                if live_change_pct is not None else None
            ),

            "market_regime": analysis.get(
                "market_regime",
                "UNKNOWN",
            ),
            "adaptive_research": adaptive,

            "data_date": data_date.strftime(
                "%d/%m/%Y"
            ),

            "stop_loss": None,
            "target_price": None,
            "reward_risk": None,
            "stop_distance_pct": None,
            "target_distance_pct": None,

            "reason": reason,
            "blockers": blockers,
            "reasons": reasons,
            "penalties": penalties,

            "relative_strength": analysis.get(
                "relative_strength"
            ),

            "avg_turnover_20d": round(
                avg_turnover
            ),

            "median_turnover_20d": round(
                median_turnover
            ),

            "atr_pct": analysis.get(
                "atr_pct"
            ),

            "market_context": (
                f"VN-Index: {analysis.get('market_regime', 'UNKNOWN')}"
                if analysis.get("market_regime") != "UNKNOWN"
                else "Chưa có dữ liệu VN-Index"
            ),
        }

        if (
            risk_plan is not None
            and signal in ("BUY", "WAIT", "WATCH")
        ):
            row["stop_loss"] = risk_plan["stop_loss"]
            row["target_price"] = risk_plan["target_price"]
            row["reward_risk"] = risk_plan["reward_risk"]

            row["stop_distance_pct"] = (
                risk_plan["stop_distance_pct"]
            )

            row["target_distance_pct"] = (
                risk_plan["target_distance_pct"]
            )

        rows.append(row)

    return rows


# =====================================================
# SCAN MARKET
# =====================================================

def scan_market(
    max_symbols=None,
    progress_callback=None,
    symbols=None,
    realtime_quotes=None,
):
    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history,
        InsufficientHistoryError,
    )

    if symbols is None:
        symbols = fetch_exchange_symbols()

    symbols = sorted({
        str(s).strip().upper()
        for s in symbols
        if s
    })

    if max_symbols is not None:
        symbols = symbols[
            :max(0, int(max_symbols))
        ]

    realtime_quotes = realtime_quotes or {}

    # Lấy VN-Index một lần cho toàn bộ lượt quét.
    market_df = None
    market_data_error = None

    try:
        market_df = fetch_stock_history(
            "VNINDEX",
            length=260,
            is_index=True,
        )

    except Exception as exc:
        market_data_error = str(exc)

    results = []
    errors = []

    insufficient = 0
    d1_errors = 0

    for index, symbol in enumerate(symbols, 1):

        try:
            history = fetch_stock_history(
                symbol,
                length=260,
            )

            foreign_flow = None
            try:
                from foreign_flow_provider import fetch_foreign_flow
                foreign_flow = fetch_foreign_flow(symbol, length=20)
            except Exception:
                pass
            stock_results = analyze_stock(
                history,
                symbol,
                realtime_quote=realtime_quotes.get(symbol),
                market_df=market_df,
                foreign_flow=foreign_flow,
            )

            results.extend(stock_results)

        except InsufficientHistoryError as exc:
            insufficient += 1

            errors.append({
                "symbol": symbol,
                "stage": "INSUFFICIENT_HISTORY",
                "error": str(exc),
            })

        except Exception as exc:
            message = str(exc)

            if "INSUFFICIENT_HISTORY" in message:
                insufficient += 1
                stage = "INSUFFICIENT_HISTORY"
            else:
                d1_errors += 1
                stage = "D1"

            errors.append({
                "symbol": symbol,
                "stage": stage,
                "error": message,
            })

        finally:
            if progress_callback:
                progress_callback(
                    index,
                    len(symbols),
                )

    priority = {
        "BUY": 5,
        "WAIT": 4,
        "WATCH": 3,
        "SKIP": 2,
        "FILTERED": 1,
    }

    results.sort(
        key=lambda item: (
            priority.get(
                item.get("signal"),
                0,
            ),
            _number(item.get("score")),
            _number(item.get("reward_risk")),
        ),
        reverse=True,
    )

    count = lambda signal: sum(
        item.get("signal") == signal
        for item in results
    )

    strategy_counts = {}

    for strategy in STRATEGIES:

        strategy_rows = [
            item for item in results
            if item.get("strategy") == strategy
        ]

        strategy_counts[strategy] = {
            "total": len(strategy_rows),

            "buy": sum(
                item["signal"] == "BUY"
                for item in strategy_rows
            ),

            "wait": sum(
                item["signal"] == "WAIT"
                for item in strategy_rows
            ),

            "watch": sum(
                item["signal"] == "WATCH"
                for item in strategy_rows
            ),

            "skip": sum(
                item["signal"] == "SKIP"
                for item in strategy_rows
            ),

            "filtered": sum(
                item["signal"] == "FILTERED"
                for item in strategy_rows
            ),
        }

    realtime_count = sum(
        item.get("price_data_status") == "REALTIME"
        for item in results
    )

    last_session_count = sum(
        item.get("price_data_status") == "LAST_SESSION"
        for item in results
    )

    unverified_count = sum(
        item.get("price_data_status") == "UNVERIFIED"
        for item in results
    )

    return {
        "results": results,
        "scanned": len(symbols),
        "result_count": len(results),

        "confirmed_buy_count": count("BUY"),
        "wait_count": count("WAIT"),
        "watch_count": count("WATCH"),
        "skip_count": count("SKIP"),
        "filtered_count": count("FILTERED"),

        "liquidity_filtered_count": count("FILTERED"),
        "insufficient_history_count": insufficient,
        "d1_error_count": d1_errors,
        "error_count": len(errors),

        "realtime_count": realtime_count,
        "last_session_count": last_session_count,
        "unverified_quote_count": unverified_count,

        "strategy_counts": strategy_counts,

        "market_data_available": market_df is not None,
        "market_data_error": market_data_error,

        "errors": errors,
    }
