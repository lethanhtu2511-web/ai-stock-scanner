
import numpy as np
import pandas as pd

# =========================================================
# CẤU HÌNH CHIẾN LƯỢC
# =========================================================

MIN_SCORE = 78
EARLY_MIN_SCORE = 70
MIN_REWARD_RISK = 2.0

MIN_AVG_TURNOVER_20D = 5_000_000_000
MIN_MEDIAN_TURNOVER_20D = 3_000_000_000
MIN_HISTORY_SESSIONS = 200

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

EARLY_MAX_EXTENSION_PCT = 5.0
EARLY_HIGH20_MIN_DISTANCE = 0.96
EARLY_HIGH20_MAX_DISTANCE = 1.015

MANUALLY_BLOCKED_SYMBOLS = {
    "NVB": "Mã đang được người dùng chặn thủ công do cảnh báo ứng dụng."
}


# =========================================================
# TÍNH CHỈ BÁO
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
            f"INSUFFICIENT_HISTORY: chỉ có {len(data)} phiên; "
            f"yêu cầu tối thiểu {MIN_HISTORY_SESSIONS} phiên."
        )

    close = data["Close"]

    data["TURNOVER"] = close * data["Volume"]
    data["AVG_TURNOVER_20D"] = data["TURNOVER"].rolling(20).mean()
    data["MEDIAN_TURNOVER_20D"] = data["TURNOVER"].rolling(20).median()

    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()
    data["MA200"] = close.rolling(200).mean()
    data["MA20_PREVIOUS_5D"] = data["MA20"].shift(5)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]
    data["MACD_HIST_PREVIOUS"] = data["MACD_HIST"].shift(1)

    data["VOLUME_MA20"] = data["Volume"].rolling(20).mean()
    data["VOLUME_MA5"] = data["Volume"].rolling(5).mean()

    data["VOLUME_RATIO"] = (
        data["Volume"] / data["VOLUME_MA20"].replace(0, np.nan)
    )

    data["VOLUME_DRYUP_RATIO"] = (
        data["VOLUME_MA5"] / data["VOLUME_MA20"].replace(0, np.nan)
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["ATR14"] = tr.rolling(14).mean()
    data["ATR14_PCT"] = data["ATR14"] / close * 100

    data["HIGH20_PREVIOUS"] = data["High"].shift(1).rolling(20).max()
    data["HIGH50_PREVIOUS"] = data["High"].shift(1).rolling(50).max()
    data["HIGH60_PREVIOUS"] = data["High"].shift(1).rolling(60).max()

    data["RANGE20"] = (
        data["High"].rolling(20).max()
        - data["Low"].rolling(20).min()
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
# ĐIỂM KỸ THUẬT D1
# =========================================================

def calculate_score(row):
    score = 0
    reasons = []

    if row.Close > row.MA20:
        score += 7
        reasons.append("Giá trên MA20")

    if row.MA20 > row.MA50:
        score += 7
        reasons.append("MA20 trên MA50")

    if row.MA50 > row.MA200:
        score += 6
        reasons.append("MA50 trên MA200")

    if row.VOLUME_RATIO >= 1.5:
        score += 10
        reasons.append("Khối lượng cao hơn TB20 ít nhất 50%")
    elif row.VOLUME_RATIO >= 1.2:
        score += 7
        reasons.append("Khối lượng cải thiện")

    if row.TURNOVER > row.AVG_TURNOVER_20D * 1.2:
        score += 10
        reasons.append("Giá trị giao dịch tăng")

    if row.Close > row.Open and row.Volume >= row.VOLUME_MA20:
        score += 5
        reasons.append("Nến tăng có khối lượng xác nhận")

    if row.MACD > row.MACD_SIGNAL:
        score += 7
        reasons.append("MACD tích cực")

    if 52 <= row.RSI <= 65:
        score += 8
        reasons.append("RSI trong vùng khỏe")
    elif 48 <= row.RSI < 52 or 65 < row.RSI <= 68:
        score += 5
        reasons.append("RSI chấp nhận được")

    if row.Close > row.HIGH20_PREVIOUS:
        score += 15
        reasons.append("Vượt đỉnh 20 phiên")
    elif row.Close >= row.HIGH20_PREVIOUS * 0.97:
        score += 8
        reasons.append("Áp sát đỉnh 20 phiên")

    candle_range = max(float(row.High - row.Low), 1e-9)
    close_position = (float(row.Close) - float(row.Low)) / candle_range

    if row.Close > row.Open and close_position >= 0.75:
        score += 10
        reasons.append("Nến tăng đóng cửa gần mức cao nhất")
    elif row.Close > row.Open and close_position >= 0.55:
        score += 6
        reasons.append("Nến tăng tương đối tốt")

    atr_pct = float(row.ATR14_PCT)

    if 1.5 <= atr_pct <= 4:
        score += 15
        reasons.append("Biến động phù hợp lướt sóng")
    elif 1 <= atr_pct < 1.5 or 4 < atr_pct <= 5.5:
        score += 10
        reasons.append("Biến động chấp nhận được")
    elif atr_pct <= 6.5:
        score += 5
        reasons.append("Biến động cần thận trọng")

    return min(score, 100), reasons


# =========================================================
# ĐIỂM EARLY BUY
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

    if EARLY_HIGH20_MIN_DISTANCE <= proximity <= EARLY_HIGH20_MAX_DISTANCE:
        score += 18
        reasons.append("Giá sát vùng đỉnh 20 phiên")
    elif 0.93 <= proximity < EARLY_HIGH20_MIN_DISTANCE:
        score += 10
        reasons.append("Giá tiến gần vùng đỉnh")

    if row.VOLUME_DRYUP_RATIO <= 0.85:
        score += 12
        reasons.append("Khối lượng 5 phiên co lại")
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

    return min(score, 100), reasons


# =========================================================
# TỶ LỆ LỜI/LỖ SAU CHI PHÍ
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

    resistance_levels = [
        level for level in candidates
        if np.isfinite(level) and level > price * 1.01
    ]

    if not resistance_levels:
        return None

    target = min(resistance_levels)
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
# PHÂN TÍCH MỘT MÃ - CHỈ SỬ DỤNG D1
# =========================================================

def analyze_stock(df, symbol=None):
    try:
        symbol = str(symbol or "").strip().upper()

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

        # BUY xác nhận theo D1
        if (
            score >= MIN_SCORE
            and 45 <= float(row.RSI) <= 68
            and price > float(row.MA20)
            and extension <= 8
            and plan is not None
            and plan["reward_risk"] >= MIN_REWARD_RISK
        ):
            return {
                **common,
                **plan,
                "signal": "BUY",
                "score": score,
                "confirmed_score": score,
                "early_score": early_score,
                "reasons": reasons,
                "reason": "Đạt điều kiện BUY xác nhận theo D1 và RR sau chi phí.",
            }

        # EARLY BUY theo D1
        early_ok = (
            early_score >= EARLY_MIN_SCORE
            and price >= float(row.MA20) * 0.99
            and 45 <= float(row.RSI) <= 68
            and extension <= EARLY_MAX_EXTENSION_PCT
            and EARLY_HIGH20_MIN_DISTANCE <= proximity <= EARLY_HIGH20_MAX_DISTANCE
            and plan is not None
            and plan["reward_risk"] >= MIN_REWARD_RISK
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
                "reason": "EARLY BUY theo D1; tín hiệu sớm, cần theo dõi thêm.",
            }

        blockers = []

        if score < MIN_SCORE:
            blockers.append(
                f"Điểm kỹ thuật {score}, yêu cầu {MIN_SCORE}."
            )

        if not 45 <= float(row.RSI) <= 68:
            blockers.append(
                f"RSI {row.RSI:.1f} ngoài vùng 45–68."
            )

        if price <= float(row.MA20):
            blockers.append("Giá chưa nằm trên MA20.")

        if extension > 8:
            blockers.append(
                f"Giá cách MA20 {extension:.1f}%."
            )

        if plan is None:
            blockers.append(
                "Không có mục tiêu kháng cự khách quan đạt RR 2:1."
            )

        return {
            **common,
            "signal": "WAIT",
            "score": score,
            "confirmed_score": score,
            "early_score": early_score,
            "reasons": (
                reasons if score >= early_score else early_reasons
            ),
            "blockers": blockers,
            "reason": "Chưa đủ điều kiện mua; tiếp tục theo dõi.",
        }

    except Exception as exc:
        return {
            "symbol": str(symbol or "").strip().upper(),
            "signal": "WAIT",
            "reason": str(exc),
            "data_date": None,
        }


# =========================================================
# QUÉT TOÀN THỊ TRƯỜNG - CHỈ DÙNG D1
# =========================================================

def scan_market(
    max_symbols=None,
    progress_callback=None,
    symbols=None,
):
    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history,
        InsufficientHistoryError,
    )

    if symbols is None:
        symbols = fetch_exchange_symbols()

    symbols = sorted(set(
        str(s).strip().upper()
        for s in symbols
        if s
    ))

    if max_symbols is not None:
        symbols = symbols[:max(0, int(max_symbols))]

    total = len(symbols)

    results = []
    errors = []

    filtered_count = 0
    liquidity_filtered_count = 0
    wait_count = 0

    d1_error_count = 0
    insufficient_history_count = 0

    for index, symbol in enumerate(symbols, start=1):
        try:
            if symbol in MANUALLY_BLOCKED_SYMBOLS:
                filtered_count += 1
                continue

            try:
                history = fetch_stock_history(
                    symbol,
                    length=260,
                )

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

                if "INSUFFICIENT_HISTORY" in message:
                    insufficient_history_count += 1
                    stage = "INSUFFICIENT_HISTORY"
                else:
                    d1_error_count += 1
                    stage = "D1"

                errors.append({
                    "symbol": symbol,
                    "stage": stage,
                    "error": message,
                })
                continue

            result = analyze_stock(
                history,
                symbol=symbol,
            )

            signal = result.get("signal")

            if signal == "FILTERED":
                filtered_count += 1

                if "thanh khoản" in str(result.get("reason", "")).lower():
                    liquidity_filtered_count += 1

            elif signal == "WAIT":
                wait_count += 1

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

    # Sắp xếp BUY trước, EARLY BUY sau, WAIT cuối.
    results.sort(
        key=lambda item: (
            2 if item.get("signal") == "BUY"
            else 1 if item.get("signal") == "EARLY_BUY"
            else 0,
            item.get("score", 0),
            item.get("reward_risk", 0),
        ),
        reverse=True,
    )

    confirmed_count = sum(
        item.get("signal") == "BUY"
        for item in results
    )

    early_count = sum(
        item.get("signal") == "EARLY_BUY"
        for item in results
    )

    return {
        "results": results,
        "scanned": total,
        "buy_count": confirmed_count + early_count,
        "confirmed_buy_count": confirmed_count,
        "early_buy_count": early_count,
        "filtered_count": filtered_count,
        "liquidity_filtered_count": liquidity_filtered_count,
        "wait_count": wait_count,
        "d1_error_count": d1_error_count,
        "insufficient_history_count": insufficient_history_count,
        "d1_candidate_count": confirmed_count + early_count,
        "error_count": len(errors),
        "errors": errors,
    }
