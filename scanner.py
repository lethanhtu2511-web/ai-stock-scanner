
import numpy as np
import pandas as pd

from strategy_engine import (
    analyze_stock as analyze_strategy_stock,
    STRATEGIES,
)

# =====================================================
# AI STOCK SCANNER 2.0 - D1 MULTI STRATEGY
# =====================================================

MIN_HISTORY = 200
MIN_AVG_TURNOVER = 5_000_000_000
MIN_MEDIAN_TURNOVER = 3_000_000_000
MIN_RR = 2.0

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

BLOCKED = {"NVB": "Mã bị chặn thủ công"}


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
        "Low", "Close", "Volume"
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Thiếu cột dữ liệu: {missing}")

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")

    for col in required[1:]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d = (
        d.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    valid = (
        (d["Open"] > 0)
        & (d["Close"] > 0)
        & (d["High"] >= d[["Open", "Close", "Low"]].max(axis=1))
        & (d["Low"] <= d[["Open", "Close"]].min(axis=1))
        & (d["Volume"] >= 0)
    )

    d = d.loc[valid].reset_index(drop=True)

    if len(d) < MIN_HISTORY:
        raise ValueError(
            f"INSUFFICIENT_HISTORY: {len(d)}/{MIN_HISTORY} phiên D1"
        )

    return d


def _net_rr(price, stop, target):
    cost = price * (1 + BUY_FEE + SLIPPAGE)

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


def _risk_plan(price, atr, resistance):
    if price <= 0 or atr <= 0:
        return None

    stop = round(price - 1.5 * atr)

    if resistance > price:
        target = round(resistance)
    else:
        target = round(price + 3 * atr)

    if stop <= 0 or target <= price:
        return None

    rr = _net_rr(price, stop, target)

    return {
        "stop_loss": stop,
        "target_price": target,
        "reward_risk": round(rr, 2),
    }


def _strategy_signal(result, risk_plan):
    signal = result.get("signal", "BỎ QUA")
    blockers = list(result.get("blockers", []))

    if signal == "MUA":
        if risk_plan is None:
            blockers.append(
                "Không xác định được kế hoạch cắt lỗ/mục tiêu"
            )
            return "CHỜ XÁC NHẬN", blockers

        if risk_plan["reward_risk"] < MIN_RR:
            blockers.append(
                f"RR chỉ {risk_plan['reward_risk']:.2f}, "
                f"yêu cầu tối thiểu {MIN_RR:.1f}"
            )
            return "CHỜ XÁC NHẬN", blockers

        return "BUY", blockers

    mapping = {
        "CHỜ XÁC NHẬN": "WAIT",
        "THEO DÕI": "WATCH",
        "BỎ QUA": "SKIP",
    }

    return mapping.get(signal, "WAIT"), blockers


def analyze_stock(df, symbol=None):
    symbol = str(symbol or "").strip().upper()

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
    price = float(last["Close"])
    data_date = pd.Timestamp(last["Date"])

    age = (
        pd.Timestamp.now().normalize()
        - data_date.normalize()
    ).days

    if age < 0 or age > 7:
        message = f"Dữ liệu D1 đã cũ: {age} ngày"

        return [{
            "symbol": symbol,
            "strategy": strategy,
            "signal": "WAIT",
            "score": 0,
            "price": round(price),
            "reason": message,
        } for strategy in STRATEGIES]

    turnover = d["Close"] * d["Volume"]
    previous_turnover = turnover.shift(1).dropna().tail(20)

    if len(previous_turnover) < 20:
        raise ValueError(
            "Không đủ 20 phiên để đánh giá thanh khoản."
        )

    avg_turnover = float(previous_turnover.mean())
    median_turnover = float(previous_turnover.median())

    liquidity_blockers = []

    if avg_turnover < MIN_AVG_TURNOVER:
        liquidity_blockers.append(
            "GTGD bình quân 20 phiên dưới 5 tỷ"
        )

    if median_turnover < MIN_MEDIAN_TURNOVER:
        liquidity_blockers.append(
            "GTGD trung vị 20 phiên dưới 3 tỷ"
        )

    # Gọi bộ phân tích 3 chiến lược, không gọi lại hàm này.
    analysis = analyze_strategy_stock(symbol, d)

    atr = _number(analysis.get("atr"))
    resistance = _number(analysis.get("resistance"))

    risk_plan = _risk_plan(
        price,
        atr,
        resistance
    )

    rows = []

    for strategy, detail in analysis["strategies"].items():
        score = int(detail.get("score", 0))
        reasons = list(detail.get("reasons", []))
        blockers = list(detail.get("blockers", []))

        blockers.extend(liquidity_blockers)

        if liquidity_blockers:
            signal = "FILTERED"
        else:
            signal, signal_blockers = _strategy_signal(
                detail,
                risk_plan
            )
            blockers.extend(signal_blockers)

        if signal == "BUY":
            reason = "Đạt điều kiện mua và RR tối thiểu"
        elif signal == "WAIT":
            reason = "Chưa đủ xác nhận: " + "; ".join(blockers[:4])
        elif signal == "WATCH":
            reason = "Đang theo dõi: " + "; ".join(
                reasons[:3] or blockers[:3]
            )
        elif signal == "FILTERED":
            reason = "Không đạt bộ lọc thanh khoản"
        else:
            reason = "Chưa phù hợp: " + "; ".join(
                blockers[:3] or detail.get("penalties", [])[:3]
            )

        row = {
            "symbol": symbol,
            "strategy": strategy,
            "strategy_name": STRATEGIES.get(strategy, strategy),
            "signal": signal,
            "score": score,
            "price": round(price),
            "data_date": data_date.strftime("%d/%m/%Y"),
            "stop_loss": None,
            "target_price": None,
            "reward_risk": None,
            "reason": reason,
            "blockers": list(dict.fromkeys(blockers)),
            "reasons": reasons,
            "relative_strength": analysis.get("relative_strength"),
            "avg_turnover_20d": round(avg_turnover),
            "median_turnover_20d": round(median_turnover),
            "atr_pct": analysis.get("atr_pct"),
        }

        if risk_plan is not None:
            row["stop_loss"] = risk_plan["stop_loss"]
            row["target_price"] = risk_plan["target_price"]
            row["reward_risk"] = risk_plan["reward_risk"]

        rows.append(row)

    return rows


def scan_market(
    max_symbols=None,
    progress_callback=None,
    symbols=None
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
        symbols = symbols[:max(0, int(max_symbols))]

    results = []
    errors = []
    insufficient = 0
    d1_errors = 0

    for index, symbol in enumerate(symbols, 1):
        try:
            history = fetch_stock_history(
                symbol,
                length=260
            )

            stock_results = analyze_stock(
                history,
                symbol
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
                progress_callback(index, len(symbols))

    priority = {
        "BUY": 5,
        "WAIT": 4,
        "WATCH": 3,
        "SKIP": 2,
        "FILTERED": 1,
    }

    results.sort(
        key=lambda item: (
            priority.get(item.get("signal"), 0),
            _number(item.get("score")),
            _number(item.get("reward_risk")),
        ),
        reverse=True
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
