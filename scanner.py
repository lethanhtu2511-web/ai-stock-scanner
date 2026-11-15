
import numpy as np
import pandas as pd

from strategy_engine import (
    analyze_stock as analyze_strategy_stock,
    STRATEGIES,
)

# =====================================================
# AI STOCK SCANNER 2.2 - D1 + REALTIME PRICE
# =====================================================

MIN_HISTORY = 200
MIN_AVG_TURNOVER = 5_000_000_000
MIN_MEDIAN_TURNOVER = 3_000_000_000
MIN_RR = 2.0

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

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
    if not isinstance(quote, dict):
        return None

    quote_symbol = str(
        quote.get("symbol", "")
    ).strip().upper()

    if quote_symbol and quote_symbol != symbol:
        return None

    price = _number(
        quote.get("price"),
        default=0.0,
    )

    if price <= 0:
        return None

    return {
        **quote,
        "symbol": symbol,
        "price": price,
    }


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
        items = reasons[:3] or blockers[:3]

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


# =====================================================
# ANALYZE ONE STOCK
# =====================================================

def analyze_stock(
    df,
    symbol=None,
    realtime_quote=None,
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

    today = pd.Timestamp.now().normalize()

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
            "reason": message,
            "blockers": [message],
        } for strategy in STRATEGIES]

    # Kiểm tra giá trong phiên.
    quote = _validate_realtime_quote(
        symbol,
        realtime_quote,
    )

    has_realtime = quote is not None

    realtime_price = (
        quote["price"]
        if has_realtime
        else None
    )

    # Nếu có giá trong phiên thì dùng giá đó.
    # Nếu không có thì tạm tính bằng giá D1,
    # nhưng không xác nhận tín hiệu BUY.
    price = (
        realtime_price
        if has_realtime
        else d1_close
    )

    price_data_status = (
        "REALTIME"
        if has_realtime
        else "D1_ONLY"
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

    # Các chỉ báo kỹ thuật vẫn tính trên D1.
    analysis = analyze_strategy_stock(
        symbol,
        d,
    )

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

        # Không xác nhận BUY nếu không có giá trong phiên.
        if signal == "BUY" and not has_realtime:
            signal = "WAIT"

            blockers.append(
                "Chưa lấy được giá trong phiên; "
                "chưa xác nhận điều kiện mua hiện tại"
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

        row = {
            "symbol": symbol,
            "strategy": strategy,
            "strategy_name": STRATEGIES.get(
                strategy,
                strategy,
            ),
            "signal": signal,
            "score": score,

            # Giá được dùng để lập kế hoạch rủi ro.
            "price": round(price),

            # Giá tham chiếu D1 và giá trong phiên.
            "d1_close": round(d1_close),
            "realtime_price": (
                round(realtime_price)
                if has_realtime else None
            ),
            "realtime_volume": (
                quote.get("volume")
                if has_realtime else None
            ),
            "quote_fetched_at": (
                quote.get("fetched_at")
                if has_realtime else None
            ),
            "quote_source": (
                quote.get("source")
                if has_realtime else None
            ),
            "price_data_status": price_data_status,

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
                "Có dữ liệu VN-Index"
                if analysis.get("relative_strength") is not None
                else "Chưa có dữ liệu VN-Index"
            ),
        }

        # Công bố kế hoạch khi có thể tính toán.
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

            stock_results = analyze_stock(
                history,
                symbol,
                realtime_quote=realtime_quotes.get(symbol),
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
        "strategy_counts": strategy_counts,
        "errors": errors,
    }
