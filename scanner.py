
import numpy as np
import pandas as pd

# =====================================================
# AI STOCK SCANNER - D1
# =====================================================

MIN_SCORE = 78
EARLY_MIN_SCORE = 70
NEAR_MIN_SCORE = 70
MIN_RR = 2.0

MIN_HISTORY = 200
MIN_AVG_TURNOVER = 5_000_000_000
MIN_MEDIAN_TURNOVER = 3_000_000_000

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

BLOCKED = {"NVB": "Mã bị chặn thủ công"}
MANUALLY_BLOCKED_SYMBOLS = BLOCKED


def _number(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _validate(df):
    required = [
        "Date", "Open", "High",
        "Low", "Close", "Volume"
    ]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Dữ liệu D1 không hợp lệ")

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    missing = [c for c in required if c not in d]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in required[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = (
        d.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    valid = (
        (d["Open"] > 0)
        & (d["Close"] > 0)
        & (d["Low"] > 0)
        & (d["Volume"] >= 0)
        & (d["High"] >= d[["Open", "Close", "Low"]].max(axis=1))
        & (d["Low"] <= d[["Open", "Close"]].min(axis=1))
    )
    d = d.loc[valid].reset_index(drop=True)

    if len(d) < MIN_HISTORY:
        raise ValueError(
            f"INSUFFICIENT_HISTORY: {len(d)}/{MIN_HISTORY} phiên"
        )
    return d


def calculate_indicators(df):
    d = _validate(df)

    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    v = d["Volume"]

    d["TURNOVER"] = c * v
    previous_turnover = d["TURNOVER"].shift(1)
    d["AVG_TURNOVER_20D"] = previous_turnover.rolling(20).mean()
    d["MEDIAN_TURNOVER_20D"] = previous_turnover.rolling(20).median()

    for n in (20, 50, 200):
        d[f"MA{n}"] = c.rolling(n).mean()

    d["MA20_PREVIOUS_5D"] = d["MA20"].shift(5)

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0)
    d["RSI"] = rsi.mask((gain == 0) & (loss == 0), 50)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(
        span=9, adjust=False
    ).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SIGNAL"]
    d["MACD_HIST_PREVIOUS"] = d["MACD_HIST"].shift(1)

    d["VOLUME_MA20"] = v.shift(1).rolling(20).mean()
    d["VOLUME_RATIO"] = v / d["VOLUME_MA20"].replace(0, np.nan)
    d["VOLUME_MA5_PREVIOUS"] = v.shift(1).rolling(5).mean()
    d["VOLUME_DRYUP_RATIO"] = (
        d["VOLUME_MA5_PREVIOUS"]
        / d["VOLUME_MA20"].replace(0, np.nan)
    )

    prev = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev).abs(),
        (l - prev).abs()
    ], axis=1).max(axis=1)

    d["ATR14"] = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    d["ATR14_PCT"] = d["ATR14"] / c * 100

    for n in (20, 50, 60):
        d[f"HIGH{n}_PREVIOUS"] = h.shift(1).rolling(n).max()

    d["LOW20_PREVIOUS"] = l.shift(1).rolling(20).min()
    d["RANGE20"] = (
        h.rolling(20).max() - l.rolling(20).min()
    ) / c * 100
    d["RANGE20_PREVIOUS_5D"] = d["RANGE20"].shift(5)

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[
        "MA200", "RSI", "ATR14", "VOLUME_RATIO",
        "HIGH60_PREVIOUS", "RANGE20_PREVIOUS_5D"
    ])

    if d.empty:
        raise ValueError("Không đủ dữ liệu chỉ báo")
    return d.reset_index(drop=True)


def calculate_score(r):
    score = 0
    reasons = []

    def add(points, message):
        nonlocal score
        score += points
        reasons.append(message)

    if r.Close > r.MA20:
        add(8, "Giá trên MA20")
    if r.MA20 > r.MA50:
        add(7, "MA20 trên MA50")
    if r.MA50 > r.MA200:
        add(5, "MA50 trên MA200")

    if r.VOLUME_RATIO >= 1.5:
        add(10, "Khối lượng tăng mạnh")
    elif r.VOLUME_RATIO >= 1.2:
        add(7, "Khối lượng cải thiện")

    if r.TURNOVER > r.AVG_TURNOVER_20D * 1.2:
        add(10, "Giá trị giao dịch tăng")

    if r.Close > r.Open and r.VOLUME_RATIO >= 1:
        add(5, "Nến tăng có khối lượng")

    if r.MACD > r.MACD_SIGNAL:
        add(10, "MACD tích cực")

    if 52 <= r.RSI <= 65:
        add(10, "RSI khỏe")
    elif 48 <= r.RSI <= 68:
        add(5, "RSI chấp nhận được")

    if r.Close > r.HIGH20_PREVIOUS:
        if r.VOLUME_RATIO >= 1.2:
            add(15, "Breakout 20 phiên có dòng tiền")
        else:
            add(7, "Vượt đỉnh thiếu khối lượng")
    elif r.Close >= r.HIGH20_PREVIOUS * 0.97:
        add(8, "Áp sát đỉnh 20 phiên")

    position = (r.Close - r.Low) / max(r.High - r.Low, 1e-9)
    if r.Close > r.Open and position >= 0.75:
        add(10, "Nến đóng gần đỉnh")
    elif r.Close > r.Open and position >= 0.55:
        add(6, "Nến tăng tích cực")

    if 1.5 <= r.ATR14_PCT <= 4:
        add(10, "ATR phù hợp")
    elif 1 <= r.ATR14_PCT <= 5.5:
        add(7, "ATR chấp nhận được")
    elif r.ATR14_PCT <= 6.5:
        add(3, "ATR cần theo dõi")

    return min(score, 100), reasons


def calculate_early_score(r):
    score = 0
    reasons = []
    penalties = []

    def add(points, message):
        nonlocal score
        score += points
        if points >= 0:
            reasons.append(message)
        else:
            penalties.append(message)

    slope = (
        (r.MA20 / r.MA20_PREVIOUS_5D - 1) * 100
        if r.MA20_PREVIOUS_5D > 0 else -99
    )
    proximity = r.Close / r.HIGH20_PREVIOUS

    add(12 if r.Close >= r.MA20 else -8, "Giá so với MA20")

    if r.MA20 >= r.MA50:
        add(8, "MA20 trên MA50")
    elif r.MA20 >= r.MA50 * 0.99:
        add(4, "MA20 sát MA50")
    else:
        add(-5, "MA20 dưới MA50")

    if slope >= 0.3:
        add(8, "MA20 tăng")
    elif slope >= 0:
        add(4, "MA20 ổn định")
    else:
        add(-5, "MA20 giảm")

    if 0.97 <= proximity <= 1.015:
        add(18, "Giá sát đỉnh 20 phiên")
    elif 0.94 <= proximity < 0.97:
        add(12, "Giá gần đỉnh")
    elif 0.90 <= proximity < 0.94:
        add(5, "Giá đang hồi phục")
    else:
        add(-5, "Vị trí giá chưa phù hợp")

    vr = r.VOLUME_RATIO
    if 0.85 <= vr <= 1.8:
        add(10, "Khối lượng chấp nhận được")
    elif 0.65 <= vr < 0.85:
        add(5, "Khối lượng hơi thấp")
    elif vr > 1.8:
        add(5, "Khối lượng tăng mạnh")
    else:
        add(-12, "Khối lượng quá thấp")

    if r.VOLUME_DRYUP_RATIO <= 0.85:
        add(8, "Khối lượng nền co hẹp")
    elif r.VOLUME_DRYUP_RATIO <= 1:
        add(5, "Khối lượng nền ổn định")
    elif r.VOLUME_DRYUP_RATIO > 1.5:
        add(-5, "Khối lượng nền bất thường")

    if 1 <= r.ATR14_PCT <= 4.5:
        add(8, "ATR phù hợp")
    elif r.ATR14_PCT > 6.5:
        add(-8, "ATR quá cao")
    else:
        add(3, "ATR cần theo dõi")

    if r.RANGE20 <= r.RANGE20_PREVIOUS_5D * 1.05:
        add(8, "Biên độ tích lũy ổn định")

    if 50 <= r.RSI <= 65:
        add(10, "RSI khỏe")
    elif 45 <= r.RSI < 50 or 65 < r.RSI <= 68:
        add(5, "RSI chấp nhận được")
    else:
        add(-10 if r.RSI > 75 else -5, "RSI chưa phù hợp")

    if r.MACD > r.MACD_SIGNAL:
        add(8, "MACD tích cực")
    elif r.MACD_HIST > r.MACD_HIST_PREVIOUS:
        add(5, "MACD cải thiện")
    else:
        add(-5, "MACD chưa xác nhận")

    position = (r.Close - r.Low) / max(r.High - r.Low, 1e-9)
    if r.Close >= r.Open and position >= 0.65:
        add(8, "Nến giữ giá tốt")
    elif position < 0.3:
        add(-8, "Nến đóng cửa yếu")

    return max(0, min(int(score), 100)), reasons, penalties


def _net_rr(price, stop, target):
    cost = price * (1 + BUY_FEE + SLIPPAGE)
    stop_net = stop * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    target_net = target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    risk = cost - stop_net
    reward = target_net - cost
    return reward / risk if risk > 0 and reward > 0 else 0.0


def _risk_plan(r, price):
    atr = _number(r.ATR14)
    if atr <= 0:
        return None, "ATR không hợp lệ"

    # Cắt lỗ theo ATR, không tự nhận là mức hỗ trợ chắc chắn.
    stop = price - 1.5 * atr
    if stop <= 0:
        return None, "Cắt lỗ không hợp lệ"

    levels = sorted(set(
        float(x) for x in [
            r.HIGH20_PREVIOUS,
            r.HIGH50_PREVIOUS,
            r.HIGH60_PREVIOUS
        ]
        if np.isfinite(x) and x > price
    ))

    if not levels:
        return None, "Không có kháng cự lịch sử phía trên giá"

    # Không bỏ qua kháng cự gần để chọn mục tiêu xa.
    target = levels[0]
    rr = _net_rr(price, stop, target)

    if rr < MIN_RR:
        return None, (
            f"RR tới kháng cự gần nhất chỉ {rr:.2f}, "
            f"yêu cầu {MIN_RR:.1f}"
        )

    # Kiểm tra lại RR sau khi làm tròn giá.
    stop = round(stop)
    target = round(target)
    rr = _net_rr(price, stop, target)

    if rr < MIN_RR:
        return None, f"RR sau làm tròn chỉ {rr:.2f}"

    return {
        "stop_loss": stop,
        "target_price": target,
        "resistance": target,
        "reward_risk": round(rr, 2)
    }, ""


def _checks(values):
    return {name: bool(ok) for name, ok in values.items()}


def _failed(checks):
    return [name for name, ok in checks.items() if not ok]


def analyze_stock(df, symbol=None):
    symbol = str(symbol or "").strip().upper()

    try:
        if symbol in BLOCKED:
            return {
                "symbol": symbol,
                "signal": "FILTERED",
                "reason": BLOCKED[symbol],
                "blockers": [BLOCKED[symbol]]
            }

        d = calculate_indicators(df)
        r = d.iloc[-1]
        price = float(r.Close)

        data_date = pd.Timestamp(r.Date)
        age = (pd.Timestamp.now().normalize() - data_date.normalize()).days

        avg = float(r.AVG_TURNOVER_20D)
        median = float(r.MEDIAN_TURNOVER_20D)

        common = {
            "symbol": symbol,
            "price": round(price),
            "data_date": data_date.strftime("%d/%m/%Y"),
            "rsi": round(float(r.RSI), 2),
            "volume_ratio": round(float(r.VOLUME_RATIO), 2),
            "atr_pct": round(float(r.ATR14_PCT), 2),
            "avg_turnover_20d": round(avg),
            "median_turnover_20d": round(median)
        }

        if age < 0 or age > 7:
            message = f"Dữ liệu D1 không mới: lệch {age} ngày"
            return {
                **common, "signal": "WAIT",
                "reason": message,
                "blockers": [message]
            }

        liquidity = []
        if avg < MIN_AVG_TURNOVER:
            liquidity.append("GTGD bình quân 20 phiên dưới 5 tỷ")
        if median < MIN_MEDIAN_TURNOVER:
            liquidity.append("GTGD trung vị 20 phiên dưới 3 tỷ")

        if liquidity:
            return {
                **common,
                "signal": "FILTERED",
                "reason": "; ".join(liquidity),
                "blockers": liquidity,
                "failed_buy": liquidity,
                "failed_early": liquidity,
                "failed_near": liquidity
            }

        confirmed, reasons = calculate_score(r)
        early, early_reasons, penalties = calculate_early_score(r)
        plan, plan_error = _risk_plan(r, price)

        extension = (price / r.MA20 - 1) * 100
        proximity = price / r.HIGH20_PREVIOUS
        vr = float(r.VOLUME_RATIO)
        rsi = float(r.RSI)

        # Điều kiện BUY xác nhận.
        buy = _checks({
            f"BUY score >= {MIN_SCORE}": confirmed >= MIN_SCORE,
            "RSI 45-68": 45 <= rsi <= 68,
            "Giá trên MA20": price > r.MA20,
            "Giá không vượt MA20 quá 8%": 0 <= extension <= 8,
            "Khối lượng >= 0.8 lần TB20": vr >= 0.8,
            "RR tới kháng cự gần nhất >= 2": plan is not None
        })

        # EARLY: ưu tiên sát điểm bứt phá.
        early_checks = _checks({
            f"EARLY score >= {EARLY_MIN_SCORE}": early >= EARLY_MIN_SCORE,
            "Giá >= 99% MA20": price >= r.MA20 * 0.99,
            "RSI 45-68": 45 <= rsi <= 68,
            "Không vượt MA20 quá 5%": extension <= 5,
            "Giá bằng 96%-101.5% đỉnh 20 phiên":
                0.96 <= proximity <= 1.015,
            "Khối lượng >= 0.65 lần TB20": vr >= 0.65,
            "RR tới kháng cự gần nhất >= 2": plan is not None
        })

        # NEAR: tín hiệu theo dõi, không phải lệnh mua.
        near = _checks({
            f"EARLY score >= {NEAR_MIN_SCORE}": early >= NEAR_MIN_SCORE,
            "Giá >= 98% MA20": price >= r.MA20 * 0.98,
            "RSI 45-68": 45 <= rsi <= 68,
            "Không vượt MA20 quá 8%": extension <= 8,
            "Giá >= 93% đỉnh 20 phiên": proximity >= 0.93,
            "Khối lượng >= 0.60 lần TB20": vr >= 0.60,
            "RR tới kháng cự gần nhất >= 2": plan is not None
        })

        failed_buy = _failed(buy)
        failed_early = _failed(early_checks)
        failed_near = _failed(near)

        if not failed_buy:
            signal = "BUY"
            score = confirmed
            active_reasons = reasons
            blockers = []
        elif not failed_early:
            signal = "EARLY_BUY"
            score = early
            active_reasons = early_reasons
            blockers = failed_buy
        elif not failed_near:
            signal = "NEAR_BUY"
            score = early
            active_reasons = early_reasons
            blockers = failed_buy + failed_early
        else:
            signal = "WAIT"
            score = confirmed
            active_reasons = reasons
            blockers = (
                ["BUY: " + x for x in failed_buy]
                + ["EARLY: " + x for x in failed_early]
                + ["NEAR: " + x for x in failed_near]
            )

        if signal == "BUY":
            reason = "Đủ điều kiện BUY xác nhận D1"
        elif signal == "EARLY_BUY":
            reason = "Đủ điều kiện EARLY; chưa xác nhận BUY"
        elif signal == "NEAR_BUY":
            reason = "Đạt điều kiện theo dõi NEAR; chưa xác nhận mua"
        else:
            reason = "Chưa đạt: " + "; ".join(
                failed_buy[:2] + failed_early[:2] + failed_near[:2]
            )

        result = {
            **common,
            "signal": signal,
            "score": score,
            "confirmed_score": confirmed,
            "early_score": early,
            "reason": reason,
            "reasons": active_reasons,
            "early_reasons": early_reasons,
            "penalties": penalties,
            "blockers": list(dict.fromkeys(blockers)),
            "conditions": buy,
            "buy_conditions": buy,
            "early_conditions": early_checks,
            "near_conditions": near,
            "failed_buy": failed_buy,
            "failed_early": failed_early,
            "failed_near": failed_near,
            "risk_note": plan_error,
            "extension_pct": round(extension, 2),
            "proximity_pct": round(proximity * 100, 2),
            "stop_loss": None,
            "target_price": None,
            "reward_risk": None
        }

        if plan:
            result.update(plan)

        return result

    except Exception as exc:
        message = str(exc)
        return {
            "symbol": symbol,
            "signal": "WAIT",
            "reason": message,
            "blockers": [message],
            "failed_buy": [message],
            "failed_early": [message],
            "failed_near": [message]
        }


def scan_market(max_symbols=None, progress_callback=None, symbols=None):
    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history,
        InsufficientHistoryError
    )

    if symbols is None:
        symbols = fetch_exchange_symbols()

    symbols = sorted(set(
        str(s).strip().upper() for s in symbols if s
    ))

    if max_symbols is not None:
        symbols = symbols[:max(0, int(max_symbols))]

    results = []
    errors = []
    insufficient = 0
    d1_errors = 0

    for i, symbol in enumerate(symbols, 1):
        try:
            if symbol in BLOCKED:
                results.append({
                    "symbol": symbol,
                    "signal": "FILTERED",
                    "reason": BLOCKED[symbol],
                    "blockers": [BLOCKED[symbol]]
                })
                continue

            history = fetch_stock_history(symbol, length=260)
            results.append(analyze_stock(history, symbol))

        except InsufficientHistoryError as exc:
            insufficient += 1
            errors.append({
                "symbol": symbol,
                "stage": "INSUFFICIENT_HISTORY",
                "error": str(exc)
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
                "error": message
            })
        finally:
            if progress_callback:
                progress_callback(i, len(symbols))

    priority = {
        "BUY": 4, "EARLY_BUY": 3,
        "NEAR_BUY": 2, "WAIT": 1,
        "FILTERED": 0
    }

    results.sort(
        key=lambda x: (
            priority.get(x.get("signal"), 0),
            _number(x.get("score")),
            _number(x.get("reward_risk"))
        ),
        reverse=True
    )

    count = lambda signal: sum(
        x.get("signal") == signal for x in results
    )

    return {
        "results": results,
        "scanned": len(symbols),
        "buy_count": count("BUY") + count("EARLY_BUY"),
        "confirmed_buy_count": count("BUY"),
        "early_buy_count": count("EARLY_BUY"),
        "near_buy_count": count("NEAR_BUY"),
        "wait_count": count("WAIT"),
        "filtered_count": count("FILTERED"),
        "liquidity_filtered_count": sum(
            x.get("signal") == "FILTERED"
            and "GTGD" in str(x.get("reason", ""))
            for x in results
        ),
        "insufficient_history_count": insufficient,
        "d1_error_count": d1_errors,
        "d1_candidate_count": count("BUY") + count("EARLY_BUY"),
        "error_count": len(errors),
        "errors": errors
    }
