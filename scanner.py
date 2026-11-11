
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

MIN_SCORE = 78
EARLY_MIN_SCORE = 70
NEAR_MIN_SCORE = 70
MIN_REWARD_RISK = 2.0

MIN_AVG_TURNOVER_20D = 5_000_000_000
MIN_MEDIAN_TURNOVER_20D = 3_000_000_000
MIN_HISTORY_SESSIONS = 200

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

EARLY_MAX_EXTENSION_PCT = 5.0
MANUALLY_BLOCKED_SYMBOLS = {
    "NVB": "Mã đang được người dùng chặn thủ công."
}


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Dữ liệu D1 không hợp lệ.")

    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Thiếu cột D1: {missing}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = (
        data.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    valid = (
        (data["Open"] > 0)
        & (data["High"] >= data["Low"])
        & (data["Low"] > 0)
        & (data["Close"] > 0)
        & (data["Volume"] >= 0)
        & (data["High"] >= data["Open"])
        & (data["High"] >= data["Close"])
        & (data["Low"] <= data["Open"])
        & (data["Low"] <= data["Close"])
    )

    data = data.loc[valid].reset_index(drop=True)

    if len(data) < MIN_HISTORY_SESSIONS:
        raise ValueError(
            f"INSUFFICIENT_HISTORY: {len(data)} phiên; "
            f"yêu cầu tối thiểu {MIN_HISTORY_SESSIONS}."
        )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # Giá trị giao dịch ước tính
    data["TURNOVER"] = close * volume

    # Nền thanh khoản chỉ lấy 20 phiên trước,
    # không đưa phiên đang đánh giá vào mẫu.
    data["AVG_TURNOVER_20D"] = (
        data["TURNOVER"].shift(1).rolling(20).mean()
    )
    data["MEDIAN_TURNOVER_20D"] = (
        data["TURNOVER"].shift(1).rolling(20).median()
    )

    # Moving averages
    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()
    data["MA200"] = close.rolling(200).mean()
    data["MA20_PREVIOUS_5D"] = data["MA20"].shift(5)

    # RSI Wilder
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    data["RSI"] = rsi

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9, adjust=False
    ).mean()
    data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]
    data["MACD_HIST_PREVIOUS"] = data["MACD_HIST"].shift(1)

    # Volume baseline: 20 phiên trước
    data["VOLUME_MA20"] = volume.shift(1).rolling(20).mean()
    data["VOLUME_MA5"] = volume.rolling(5).mean()
    data["VOLUME_MA5_PREVIOUS"] = (
        volume.shift(1).rolling(5).mean()
    )

    data["VOLUME_RATIO"] = (
        volume / data["VOLUME_MA20"].replace(0, np.nan)
    )

    data["VOLUME_DRYUP_RATIO"] = (
        data["VOLUME_MA5_PREVIOUS"]
        / data["VOLUME_MA20"].replace(0, np.nan)
    )

    # True Range và Wilder ATR
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    data["ATR14"] = tr.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    data["ATR14_PCT"] = data["ATR14"] / close * 100

    # Kháng cự chỉ lấy dữ liệu trước phiên hiện tại
    data["HIGH20_PREVIOUS"] = high.shift(1).rolling(20).max()
    data["HIGH50_PREVIOUS"] = high.shift(1).rolling(50).max()
    data["HIGH60_PREVIOUS"] = high.shift(1).rolling(60).max()

    data["LOW20_PREVIOUS"] = low.shift(1).rolling(20).min()

    data["RANGE20"] = (
        high.rolling(20).max() - low.rolling(20).min()
    ) / close * 100

    data["RANGE20_PREVIOUS_5D"] = data["RANGE20"].shift(5)

    data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError("Không còn dữ liệu hợp lệ sau khi tính chỉ báo.")

    return data


# =========================================================
# CONFIRMED SCORE
# =========================================================

def calculate_score(row):
    score = 0
    reasons = []

    # Nhóm xu hướng: tối đa 20 điểm
    trend = 0

    if row.Close > row.MA20:
        trend += 8
        reasons.append("Giá trên MA20")

    if row.MA20 > row.MA50:
        trend += 7
        reasons.append("MA20 trên MA50")

    if row.MA50 > row.MA200:
        trend += 5
        reasons.append("MA50 trên MA200")

    score += min(trend, 20)

    # Nhóm dòng tiền: tối đa 25 điểm
    flow = 0

    if row.VOLUME_RATIO >= 1.5:
        flow += 10
        reasons.append("Khối lượng tăng ít nhất 50% so với nền 20 phiên")
    elif row.VOLUME_RATIO >= 1.2:
        flow += 7
        reasons.append("Khối lượng cải thiện")

    if row.TURNOVER > row.AVG_TURNOVER_20D * 1.2:
        flow += 10
        reasons.append("Giá trị giao dịch tăng")

    if row.Close > row.Open and row.Volume >= row.VOLUME_MA20:
        flow += 5
        reasons.append("Nến tăng có khối lượng xác nhận")

    score += min(flow, 25)

    # MACD
    if row.MACD > row.MACD_SIGNAL:
        score += 10
        reasons.append("MACD tích cực")

    # RSI
    if 52 <= row.RSI <= 65:
        score += 10
        reasons.append("RSI trong vùng khỏe")
    elif 48 <= row.RSI < 52 or 65 < row.RSI <= 68:
        score += 5
        reasons.append("RSI chấp nhận được")

    # Breakout: cần cả giá và thanh khoản
    breakout = row.Close > row.HIGH20_PREVIOUS
    near_high = row.Close >= row.HIGH20_PREVIOUS * 0.97

    if breakout and row.VOLUME_RATIO >= 1.2:
        score += 15
        reasons.append("Breakout đỉnh 20 phiên có thanh khoản")
    elif breakout:
        score += 7
        reasons.append("Vượt đỉnh 20 phiên nhưng thanh khoản chưa xác nhận")
    elif near_high:
        score += 8
        reasons.append("Áp sát đỉnh 20 phiên")

    # Chất lượng nến
    candle_range = max(float(row.High - row.Low), 1e-9)
    close_position = (float(row.Close) - float(row.Low)) / candle_range

    if row.Close > row.Open and close_position >= 0.75:
        score += 10
        reasons.append("Nến tăng đóng cửa gần mức cao nhất")
    elif row.Close > row.Open and close_position >= 0.55:
        score += 6
        reasons.append("Nến tăng tương đối tốt")

    # ATR
    atr_pct = float(row.ATR14_PCT)

    if 1.5 <= atr_pct <= 4:
        score += 10
        reasons.append("Biến động phù hợp lướt sóng")
    elif 1 <= atr_pct < 1.5 or 4 < atr_pct <= 5.5:
        score += 7
        reasons.append("Biến động chấp nhận được")
    elif atr_pct <= 6.5:
        score += 3
        reasons.append("Biến động thấp hoặc cần thận trọng")

    return min(int(score), 100), reasons


# =========================================================
# EARLY SCORE
# =========================================================

def calculate_early_score(row):
    score = 0
    reasons = []

    high20 = float(row.HIGH20_PREVIOUS)
    proximity = float(row.Close) / high20 if high20 > 0 else 0

    ma20_prev = float(row.MA20_PREVIOUS_5D)

    slope = (
        (float(row.MA20) / ma20_prev - 1) * 100
        if ma20_prev > 0 else -99
    )

    if row.Close >= row.MA20:
        score += 12
        reasons.append("Giá giữ trên MA20")

    if row.MA20 >= row.MA50 * 0.99:
        score += 8
        reasons.append("MA20 áp sát hoặc trên MA50")

    if slope >= 0:
        score += 8
        reasons.append("MA20 đi ngang hoặc tăng")

    if 0.96 <= proximity <= 1.015:
        score += 18
        reasons.append("Giá sát vùng đỉnh 20 phiên")
    elif 0.93 <= proximity < 0.96:
        score += 10
        reasons.append("Giá tiến gần vùng đỉnh")

    if row.VOLUME_DRYUP_RATIO <= 0.85:
        score += 12
        reasons.append("Khối lượng co lại trước khả năng bứt phá")
    elif row.VOLUME_DRYUP_RATIO <= 1:
        score += 8
        reasons.append("Khối lượng tích lũy ổn định")

    if 0.75 <= row.VOLUME_RATIO <= 1.8:
        score += 8
        reasons.append("Khối lượng chấp nhận được")

    if row.ATR14_PCT <= 4.5:
        score += 8
        reasons.append("ATR không quá lớn")

    if row.RANGE20 <= row.RANGE20_PREVIOUS_5D * 1.05:
        score += 8
        reasons.append("Biên độ thu hẹp hoặc ổn định")

    if 48 <= row.RSI <= 65:
        score += 10
        reasons.append("RSI khỏe, chưa quá nóng")
    elif 45 <= row.RSI < 48 or 65 < row.RSI <= 68:
        score += 5
        reasons.append("RSI trong vùng theo dõi")

    if (
        row.MACD > row.MACD_SIGNAL
        or row.MACD_HIST > row.MACD_HIST_PREVIOUS
    ):
        score += 8
        reasons.append("MACD/histogram cải thiện")

    candle_range = max(float(row.High - row.Low), 1e-9)
    close_position = (float(row.Close) - float(row.Low)) / candle_range

    if row.Close >= row.Open and close_position >= 0.60:
        score += 8
        reasons.append("Nến giữ giá tốt")

    return min(int(score), 100), reasons


# =========================================================
# RISK PLAN
# =========================================================

def _net_reward_risk(price, stop, target):
    buy_cost = price * (1 + BUY_FEE + SLIPPAGE)

    stop_proceeds = stop * (
        1 - SELL_FEE - SELL_TAX - SLIPPAGE
    )

    target_proceeds = target * (
        1 - SELL_FEE - SELL_TAX - SLIPPAGE
    )

    risk = buy_cost - stop_proceeds
    reward = target_proceeds - buy_cost

    if risk <= 0 or reward <= 0:
        return 0.0

    return reward / risk


def _risk_plan(row, price):
    atr = float(row.ATR14)
    stop = price - 1.5 * atr

    if stop <= 0 or stop >= price:
        return None

    candidates = [
        float(row.HIGH20_PREVIOUS),
        float(row.HIGH50_PREVIOUS),
        float(row.HIGH60_PREVIOUS),
    ]

    resistance_levels = sorted(set(
        level for level in candidates
        if np.isfinite(level) and level > price * 1.01
    ))

    if not resistance_levels:
        return None

    target = resistance_levels[0]
    rr = _net_reward_risk(price, stop, target)

    if rr < MIN_REWARD_RISK:
        return None

    return {
        "stop_loss": round(stop),
        "target_price": round(target),
        "reward_risk": round(rr, 2),
        "resistance": round(target),
    }


# =========================================================
# ANALYZE ONE STOCK
# =========================================================

def analyze_stock(df, symbol=None):
    symbol = str(symbol or "").strip().upper()

    try:
        if symbol in MANUALLY_BLOCKED_SYMBOLS:
            return {
                "symbol": symbol,
                "signal": "FILTERED",
                "reason": MANUALLY_BLOCKED_SYMBOLS[symbol],
            }

        data = calculate_indicators(df)
        row = data.iloc[-1]

        price = float(row.Close)
        data_date = pd.Timestamp(row.Date).strftime("%d/%m/%Y")

        age_days = (
            pd.Timestamp.now().normalize()
            - pd.Timestamp(row.Date).normalize()
        ).days

        if age_days > 7:
            return {
                "symbol": symbol,
                "signal": "WAIT",
                "reason": f"Dữ liệu D1 cũ {age_days} ngày.",
                "data_date": data_date,
            }

        avg_turnover = float(row.AVG_TURNOVER_20D)
        median_turnover = float(row.MEDIAN_TURNOVER_20D)

        common = {
            "symbol": symbol,
            "price": round(price),
            "data_date": data_date,
            "rsi": round(float(row.RSI), 2),
            "volume_ratio": round(float(row.VOLUME_RATIO), 2),
            "avg_turnover_20d": round(avg_turnover),
            "median_turnover_20d": round(median_turnover),
        }

        if (
            avg_turnover < MIN_AVG_TURNOVER_20D
            or median_turnover < MIN_MEDIAN_TURNOVER_20D
        ):
            return {
                **common,
                "signal": "FILTERED",
                "reason": "Không đạt tiêu chuẩn thanh khoản.",
                "blockers": ["GTGD bình quân hoặc trung vị dưới ngưỡng."],
            }

        score, reasons = calculate_score(row)
        early_score, early_reasons = calculate_early_score(row)
        plan = _risk_plan(row, price)

        extension = (
            (price / float(row.MA20) - 1) * 100
            if row.MA20 > 0 else 999
        )

        high20 = float(row.HIGH20_PREVIOUS)
        proximity = price / high20 if high20 > 0 else 0

        rsi_ok = 45 <= float(row.RSI) <= 68
        trend_ok = price > float(row.MA20)
        extension_ok = extension <= 8

        buy_conditions = {
            "Điểm kỹ thuật": score >= MIN_SCORE,
            "RSI": rsi_ok,
            "Giá trên MA20": trend_ok,
            "Không mua quá xa MA20": extension_ok,
            "Kế hoạch RR": plan is not None,
        }

        blockers = [
            f"Điểm kỹ thuật {score}, yêu cầu {MIN_SCORE}."
            for not_ok in [score < MIN_SCORE] if not_ok
        ]

        if not rsi_ok:
            blockers.append(f"RSI {row.RSI:.1f} ngoài vùng 45–68.")

        if not trend_ok:
            blockers.append("Giá chưa nằm trên MA20.")

        if not extension_ok:
            blockers.append(f"Giá cách MA20 {extension:.1f}%.")

        if plan is None:
            blockers.append(
                "Không có mục tiêu kháng cự khách quan đạt RR 2:1."
            )

        # BUY
        if all(buy_conditions.values()):
            return {
                **common,
                **plan,
                "signal": "BUY",
                "score": score,
                "confirmed_score": score,
                "early_score": early_score,
                "reasons": reasons,
                "blockers": [],
                "conditions": buy_conditions,
                "reason": "Đạt điều kiện BUY xác nhận D1.",
            }

        # EARLY BUY
        early_ok = (
            early_score >= EARLY_MIN_SCORE
            and price >= float(row.MA20) * 0.99
            and rsi_ok
            and extension <= EARLY_MAX_EXTENSION_PCT
            and 0.96 <= proximity <= 1.015
            and plan is not None
        )

        if early_ok:
            return {
                **common,
                **plan,
                "signal": "EARLY_BUY",
                "score": early_score,
                "confirmed_score": score,
                "early_score": early_score,
                "reasons": early_reasons,
                "blockers": [],
                "conditions": buy_conditions,
                "reason": "EARLY BUY theo D1; cần theo dõi xác nhận.",
            }

        # NEAR BUY: điểm tốt nhưng còn thiếu xác nhận
        near_conditions = (
            early_score >= NEAR_MIN_SCORE
            and price >= float(row.MA20) * 0.98
            and rsi_ok
            and extension <= 8
            and proximity >= 0.93
        )

        if near_conditions:
            return {
                **common,
                "signal": "NEAR_BUY",
                "score": max(score, early_score),
                "confirmed_score": score,
                "early_score": early_score,
                "reasons": early_reasons,
                "blockers": blockers,
                "conditions": buy_conditions,
                "reason": "Cổ phiếu gần đạt điều kiện mua; chờ xác nhận.",
            }

        return {
            **common,
            "signal": "WAIT",
            "score": score,
            "confirmed_score": score,
            "early_score": early_score,
            "reasons": reasons if score >= early_score else early_reasons,
            "blockers": blockers,
            "conditions": buy_conditions,
            "reason": "Chưa đủ điều kiện mua; tiếp tục theo dõi.",
        }

    except Exception as exc:
        return {
            "symbol": symbol,
            "signal": "WAIT",
            "reason": str(exc),
            "blockers": [str(exc)],
            "data_date": None,
        }


# =========================================================
# MARKET SCANNER
# =========================================================

def scan_market(max_symbols=None, progress_callback=None, symbols=None):
    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history,
        InsufficientHistoryError,
    )

    if symbols is None:
        symbols = fetch_exchange_symbols()

    symbols = sorted(set(
        str(s).strip().upper() for s in symbols if s
    ))

    if max_symbols is not None:
        symbols = symbols[:max(0, int(max_symbols))]

    total = len(symbols)
    results = []
    errors = []

    filtered_count = 0
    liquidity_filtered_count = 0
    wait_count = 0
    near_count = 0
    d1_error_count = 0
    insufficient_history_count = 0

    for index, symbol in enumerate(symbols, start=1):
        try:
            if symbol in MANUALLY_BLOCKED_SYMBOLS:
                filtered_count += 1
                continue

            try:
                history = fetch_stock_history(symbol, length=260)
            except InsufficientHistoryError as exc:
                insufficient_history_count += 1
                errors.append({
                    "symbol": symbol,
                    "stage": "INSUFFICIENT_HISTORY",
                    "error": str(exc),
                })
                continue
            except Exception as exc:
                message = str(exc)
                stage = (
                    "INSUFFICIENT_HISTORY"
                    if "INSUFFICIENT_HISTORY" in message
                    else "D1"
                )
                if stage == "INSUFFICIENT_HISTORY":
                    insufficient_history_count += 1
                else:
                    d1_error_count += 1

                errors.append({
                    "symbol": symbol,
                    "stage": stage,
                    "error": message,
                })
                continue

            result = analyze_stock(history, symbol=symbol)
            signal = result.get("signal")

            if signal == "FILTERED":
                filtered_count += 1
                if "thanh khoản" in str(result.get("reason", "")).lower():
                    liquidity_filtered_count += 1
            elif signal == "WAIT":
                wait_count += 1
            elif signal == "NEAR_BUY":
                near_count += 1

            results.append(result)

        except Exception as exc:
            d1_error_count += 1
            errors.append({
                "symbol": symbol,
                "stage": "D1_UNKNOWN",
                "error": str(exc),
            })

        finally:
            if progress_callback:
                progress_callback(index, total)

    priority = {
        "BUY": 4,
        "EARLY_BUY": 3,
        "NEAR_BUY": 2,
        "WAIT": 1,
        "FILTERED": 0,
    }

    results.sort(
        key=lambda item: (
            priority.get(item.get("signal"), 0),
            item.get("score", 0),
            item.get("reward_risk", 0),
        ),
        reverse=True,
    )

    confirmed_count = sum(
        x.get("signal") == "BUY" for x in results
    )
    early_count = sum(
        x.get("signal") == "EARLY_BUY" for x in results
    )

    return {
        "results": results,
        "scanned": total,
        "buy_count": confirmed_count + early_count,
        "confirmed_buy_count": confirmed_count,
        "early_buy_count": early_count,
        "near_buy_count": near_count,
        "filtered_count": filtered_count,
        "liquidity_filtered_count": liquidity_filtered_count,
        "wait_count": wait_count,
        "d1_error_count": d1_error_count,
        "insufficient_history_count": insufficient_history_count,
        "d1_candidate_count": confirmed_count + early_count,
        "error_count": len(errors),
        "errors": errors,
    }
