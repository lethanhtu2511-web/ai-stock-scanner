import math
import pandas as pd
import numpy as np

INITIAL_CAPITAL = 10_000_000
RISK_PERCENT = 0.02

MIN_SCORE = 78
MIN_REWARD_RISK = 2.0

EARLY_MIN_SCORE = 70
EARLY_MAX_EXTENSION_PCT = 5.0
EARLY_HIGH20_MIN_DISTANCE = 0.96
EARLY_HIGH20_MAX_DISTANCE = 1.015

MIN_AVG_TURNOVER_20D = 5_000_000_000
MIN_MEDIAN_TURNOVER_20D = 3_000_000_000

LOT_SIZE = 100
BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

EARLY_INITIAL_FRACTION = 0.25
BREAKOUT_BUFFER = 0.005
BREAKOUT_VOLUME_RATIO = 1.2

MANUALLY_BLOCKED_SYMBOLS = {
    "NVB": "Ứng dụng giao dịch hiển thị thuộc diện cảnh báo (09/11/2026)"
}


# =========================================================
# 1. CHUẨN HÓA VÀ TÍNH CHỈ BÁO
# =========================================================

def calculate_indicators(df):
    data = df.copy()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Thiếu cột dữ liệu: {missing}")

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
        & (data["High"] > 0)
        & (data["Low"] > 0)
        & (data["Close"] > 0)
        & (data["Volume"] >= 0)
        & (data["High"] >= data["Low"])
        & (data["High"] >= data["Open"])
        & (data["High"] >= data["Close"])
        & (data["Low"] <= data["Open"])
        & (data["Low"] <= data["Close"])
    )

    data = data.loc[valid].reset_index(drop=True)

    # Không thay đổi tiêu chuẩn 200 phiên.
    if len(data) < 200:
        raise ValueError(
            f"Chỉ có {len(data)} phiên hoạt động; cần tối thiểu 200."
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
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]
    data["MACD_HIST_PREVIOUS"] = data["MACD_HIST"].shift(1)

    data["VOLUME_MA5"] = data["Volume"].rolling(5).mean()
    data["VOLUME_MA20"] = data["Volume"].rolling(20).mean()

    data["VOLUME_RATIO"] = (
        data["Volume"] / data["VOLUME_MA20"].replace(0, np.nan)
    )

    data["VOLUME_DRYUP_RATIO"] = (
        data["VOLUME_MA5"] / data["VOLUME_MA20"].replace(0, np.nan)
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["ATR14"] = true_range.rolling(14).mean()
    data["ATR14_PCT"] = data["ATR14"] / close * 100
    data["ATR14_PREVIOUS_5D"] = data["ATR14"].shift(5)

    # Chỉ dùng dữ liệu các phiên trước, không dùng đỉnh của chính phiên hiện tại.
    data["HIGH20_PREVIOUS"] = data["High"].shift(1).rolling(20).max()
    data["HIGH50_PREVIOUS"] = data["High"].shift(1).rolling(50).max()

    data["RANGE20"] = (
        data["High"].rolling(20).max()
        - data["Low"].rolling(20).min()
    ) / close * 100

    data["RANGE20_PREVIOUS_5D"] = data["RANGE20"].shift(5)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)

    if len(data) == 0:
        raise ValueError("Không còn dữ liệu hợp lệ sau khi tính chỉ báo.")

    return data


# =========================================================
# 2. CHẤM ĐIỂM
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
        reasons.append("Nến tăng, đóng cửa gần mức cao nhất")
    elif row.Close > row.Open and close_position >= 0.55:
        score += 6
        reasons.append("Nến tăng tương đối tốt")

    atr_pct = float(row.ATR14) / max(float(row.Close), 1e-9) * 100

    if 1.5 <= atr_pct <= 4.0:
        score += 15
        reasons.append("Biến động phù hợp lướt sóng")
    elif 1.0 <= atr_pct < 1.5 or 4.0 < atr_pct <= 5.5:
        score += 10
        reasons.append("Biến động chấp nhận được")
    elif atr_pct <= 6.5:
        score += 5
        reasons.append("Biến động cần thận trọng")

    return min(int(score), 100), reasons


def calculate_early_score(row):
    score = 0
    reasons = []

    high20 = float(row.HIGH20_PREVIOUS)
    proximity = float(row.Close) / high20 if high20 > 0 else 0

    ma20_previous = float(row.MA20_PREVIOUS_5D)

    ma20_slope = (
        (float(row.MA20) / ma20_previous - 1) * 100
        if ma20_previous > 0 else -99
    )

    if row.Close >= row.MA20:
        score += 12
        reasons.append("Giá giữ trên MA20")

    if row.MA20 >= row.MA50 * 0.99:
        score += 8
        reasons.append("MA20 áp sát hoặc trên MA50")

    if ma20_slope >= 0:
        score += 8
        reasons.append("MA20 đi ngang hoặc tăng")

    if 0.96 <= proximity <= 1.015:
        score += 18
        reasons.append("Giá sát vùng đỉnh 20 phiên")
    elif 0.93 <= proximity < 0.96:
        score += 10
        reasons.append("Giá đang tiến gần vùng đỉnh")

    if row.VOLUME_DRYUP_RATIO <= 0.85:
        score += 12
        reasons.append("Khối lượng 5 phiên co lại")
    elif row.VOLUME_DRYUP_RATIO <= 1.0:
        score += 8
        reasons.append("Khối lượng tích lũy ổn định")

    if 0.75 <= row.VOLUME_RATIO <= 1.8:
        score += 8
        reasons.append("Khối lượng có thể chấp nhận")

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

    # Không dùng phép so sánh MACD với 98% signal vì sai khi MACD âm.
    if (
        row.MACD > row.MACD_SIGNAL
        or row.MACD_HIST > row.MACD_HIST_PREVIOUS
    ):
        score += 8
        reasons.append("MACD/histogram có dấu hiệu cải thiện")

    candle_range = max(float(row.High - row.Low), 1e-9)
    close_position = (float(row.Close) - float(row.Low)) / candle_range

    if row.Close >= row.Open and close_position >= 0.60:
        score += 8
        reasons.append("Nến giữ giá tốt")

    return min(int(score), 100), reasons


# =========================================================
# 3. QUẢN TRỊ RỦI RO
# =========================================================

def calculate_target_price(price, stop_loss, resistance=None):
    risk_share = (
        price * (1 + BUY_FEE + SLIPPAGE)
        - stop_loss * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    )

    if risk_share <= 0:
        return None

    target = price

    for _ in range(5000):
        reward = (
            target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        if reward >= MIN_REWARD_RISK * risk_share:
            min_target = math.ceil(target / 100) * 100
            break

        target += 100
    else:
        return None

    if (
        resistance is not None
        and np.isfinite(resistance)
        and resistance > price * 1.01
    ):
        resistance_price = math.floor(float(resistance) / 100) * 100

        if resistance_price < min_target:
            return None

        return resistance_price

    return min_target


def _risk_plan(price, row, capital, risk_percent):
    stop = price - 1.5 * float(row.ATR14)

    if stop <= 0 or stop >= price:
        return None

    target = calculate_target_price(
        price,
        stop,
        resistance=float(row.HIGH50_PREVIOUS),
    )

    if target is None:
        return None

    risk_share = (
        price * (1 + BUY_FEE + SLIPPAGE)
        - stop * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    )

    reward_share = (
        target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
        - price * (1 + BUY_FEE + SLIPPAGE)
    )

    rr = reward_share / risk_share if risk_share > 0 else 0
    risk_budget = float(capital) * float(risk_percent)

    quantity_risk = int(risk_budget / risk_share) if risk_share > 0 else 0

    quantity_capital = int(
        capital / (price * (1 + BUY_FEE + SLIPPAGE))
    )

    quantity = min(quantity_risk, quantity_capital)
    quantity = (quantity // LOT_SIZE) * LOT_SIZE

    return {
        "stop_loss": round(stop),
        "target_price": round(target),
        "reward_risk": round(rr, 2),
        "quantity": quantity,
        "capital_used": round(
            quantity * price * (1 + BUY_FEE + SLIPPAGE)
        ),
        "risk_money": round(quantity * risk_share),
        "risk_budget": round(risk_budget),
        "risk_per_share": round(risk_share, 2),
    }


def _early_staged_plan(price, row, plan):
    total_quantity = int(plan["quantity"])

    initial_quantity = (
        int(total_quantity * EARLY_INITIAL_FRACTION) // LOT_SIZE
    ) * LOT_SIZE

    if initial_quantity < LOT_SIZE:
        return None

    high20 = float(row.HIGH20_PREVIOUS)

    trigger = math.ceil(
        high20 * (1 + BREAKOUT_BUFFER) / 100
    ) * 100

    initial_capital = round(
        initial_quantity * price * (1 + BUY_FEE + SLIPPAGE)
    )

    initial_risk = round(
        initial_quantity * plan["risk_per_share"]
    )

    remaining_risk_budget = max(
        0, plan["risk_budget"] - initial_risk
    )

    trigger_risk_per_share = (
        trigger * (1 + BUY_FEE + SLIPPAGE)
        - plan["stop_loss"] * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    )

    add_by_risk = (
        int(remaining_risk_budget / trigger_risk_per_share)
        if trigger_risk_per_share > 0 else 0
    )

    add_by_risk = (add_by_risk // LOT_SIZE) * LOT_SIZE

    planned_remaining = max(0, total_quantity - initial_quantity)
    add_quantity = min(planned_remaining, add_by_risk)

    return {
        "initial_quantity": initial_quantity,
        "add_quantity": add_quantity,
        "breakout_trigger": round(trigger),
        "initial_capital": initial_capital,
        "initial_risk": initial_risk,
        "add_volume_ratio_required": BREAKOUT_VOLUME_RATIO,
        "add_condition": (
            "Chỉ xem xét mua thêm khi nến ngày hoàn tất đóng trên "
            "giá kích hoạt, khối lượng tối thiểu 1,2 lần TB20 và H1 "
            "xác nhận EMA9 > EMA20. Phải tính lại RR và rủi ro theo "
            "giá thực tế trước khi mua."
        ),
    }


# =========================================================
# 4. KIỂM TRA H1
# =========================================================

def _hourly_trend(hourly_df):
    if hourly_df is None or len(hourly_df) < 25:
        return None, None, False, False, "Không đủ 25 nến H1."

    frame = hourly_df.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    close_col = next(
        (
            c for c in frame.columns
            if c in ("close", "closing_price", "price")
        ),
        None,
    )

    if close_col is None:
        return None, None, False, False, "Không có cột giá đóng cửa H1."

    close = pd.to_numeric(frame[close_col], errors="coerce").dropna()

    if len(close) < 25:
        return None, None, False, False, "Không đủ 25 giá H1 hợp lệ."

    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

    if not np.isfinite(ema9) or not np.isfinite(ema20) or ema20 <= 0:
        return None, None, False, False, "EMA H1 không hợp lệ."

    return (
        ema9,
        ema20,
        ema9 > ema20,
        ema9 >= ema20 * 0.995,
        "H1 hợp lệ",
    )


# =========================================================
# 5. PHÂN TÍCH MỘT MÃ
# =========================================================

def analyze_stock(
    df,
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT,
    live_quote=None,
    symbol=None,
    hourly_df=None,
    hourly_error=None,
):
    try:
        if symbol and symbol.upper() in MANUALLY_BLOCKED_SYMBOLS:
            return {
                "signal": "FILTERED",
                "reason": MANUALLY_BLOCKED_SYMBOLS[symbol.upper()],
            }

        data = calculate_indicators(df)
        row = data.iloc[-1].copy()

        daily_close = float(row.Close)
        is_live = live_quote is not None

        if is_live:
            price = float(live_quote.get("last_price", 0))

            if not np.isfinite(price) or price <= 0:
                raise ValueError("Giá bảng điện không hợp lệ.")

            row.Close = price
        else:
            price = daily_close

        data_date = pd.Timestamp(row.Date).strftime("%d/%m/%Y")

        age_days = (
            pd.Timestamp.now().normalize()
            - pd.Timestamp(row.Date).normalize()
        ).days

        if age_days > 7:
            return {
                "signal": "WAIT",
                "reason": f"Dữ liệu D1 đã cũ {age_days} ngày.",
                "data_date": data_date,
                "h1_status": "Chưa kiểm tra",
            }

        avg_turnover = float(row.AVG_TURNOVER_20D)
        median_turnover = float(row.MEDIAN_TURNOVER_20D)

        if (
            avg_turnover < MIN_AVG_TURNOVER_20D
            or median_turnover < MIN_MEDIAN_TURNOVER_20D
        ):
            return {
                "signal": "FILTERED",
                "reason": (
                    f"Thanh khoản không đạt: bình quân "
                    f"{avg_turnover / 1e9:.2f} tỷ/ngày, "
                    f"trung vị {median_turnover / 1e9:.2f} tỷ/ngày."
                ),
                "data_date": data_date,
                "avg_turnover_20d": avg_turnover,
                "median_turnover_20d": median_turnover,
            }

        if not np.isfinite(float(row.ATR14)) or row.ATR14 <= 0:
            raise ValueError("ATR không hợp lệ.")

        ma20 = float(row.MA20)
        extension_pct = (price / ma20 - 1) * 100 if ma20 > 0 else 999

        previous_close = float(data.iloc[-2].Close)

        day_change_pct = (
            (price / previous_close - 1) * 100
            if previous_close > 0 else 0
        )

        if is_live and day_change_pct > 7:
            return {
                "signal": "WAIT",
                "reason": "Giá tăng trên 7% so với phiên trước, không mua đuổi.",
                "data_date": data_date,
            }

        h1_ema9, h1_ema20, h1_bullish, h1_early, h1_status = (
            _hourly_trend(hourly_df)
        )

        if hourly_error and hourly_df is None:
            h1_status = f"Không lấy được H1: {hourly_error}"

        confirmed_score, confirmed_reasons = calculate_score(row)
        early_score, early_reasons = calculate_early_score(row)

        plan = None

        if extension_pct <= 8 and float(row.RSI) <= 70:
            plan = _risk_plan(price, row, capital, risk_percent)

        if plan is not None:
            confirmed_conditions = [
                confirmed_score >= MIN_SCORE,
                45 <= float(row.RSI) <= 70,
                price > ma20,
                plan["reward_risk"] >= MIN_REWARD_RISK,
                plan["quantity"] >= LOT_SIZE,
                plan["risk_money"] <= plan["risk_budget"],
                extension_pct <= 8,
                h1_bullish,
            ]

            if all(confirmed_conditions):
                return {
                    "signal": "BUY",
                    "signal_type": "BUY_CONFIRMED",
                    "score": confirmed_score,
                    "early_score": early_score,
                    "price": round(price),
                    "daily_close": round(daily_close),
                    **plan,
                    "rsi": round(float(row.RSI), 2),
                    "volume_ratio": round(float(row.VOLUME_RATIO), 2),
                    "h1_ema9": round(h1_ema9, 2),
                    "h1_ema20": round(h1_ema20, 2),
                    "h1_confirmed": True,
                    "h1_status": h1_status,
                    "avg_turnover_20d": round(avg_turnover),
                    "median_turnover_20d": round(median_turnover),
                    "reasons": confirmed_reasons,
                    "data_date": data_date,
                    "reason": "Đạt đầy đủ điều kiện BUY xác nhận.",
                    "price_mode": (
                        "Giá bảng điện"
                        if is_live else "Giá đóng cửa nến ngày"
                    ),
                }

        high20 = float(row.HIGH20_PREVIOUS)
        proximity = price / high20 if high20 > 0 else 0

        early_plan = _risk_plan(price, row, capital, risk_percent)

        early_gates = [
            early_score >= EARLY_MIN_SCORE,
            price >= ma20 * 0.99,
            45 <= float(row.RSI) <= 68,
            extension_pct <= EARLY_MAX_EXTENSION_PCT,
            EARLY_HIGH20_MIN_DISTANCE <= proximity <= EARLY_HIGH20_MAX_DISTANCE,
            h1_early,
            early_plan is not None,
        ]

        if early_plan is not None:
            early_gates.extend([
                early_plan["reward_risk"] >= MIN_REWARD_RISK,
                early_plan["quantity"] >= LOT_SIZE,
                early_plan["risk_money"] <= early_plan["risk_budget"],
            ])

        if all(early_gates):
            staged = _early_staged_plan(price, row, early_plan)

            if staged is not None:
                return {
                    "signal": "EARLY_BUY",
                    "signal_type": "EARLY_ENTRY",
                    "score": early_score,
                    "confirmed_score": confirmed_score,
                    "price": round(price),
                    "daily_close": round(daily_close),
                    **early_plan,
                    **staged,
                    "rsi": round(float(row.RSI), 2),
                    "volume_ratio": round(float(row.VOLUME_RATIO), 2),
                    "h1_ema9": round(h1_ema9, 2),
                    "h1_ema20": round(h1_ema20, 2),
                    "h1_confirmed": True,
                    "h1_status": h1_status,
                    "avg_turnover_20d": round(avg_turnover),
                    "median_turnover_20d": round(median_turnover),
                    "reasons": early_reasons,
                    "data_date": data_date,
                    "reason": (
                        "EARLY BUY: chỉ giải ngân thăm dò tối đa 25%; "
                        "chờ breakout được xác nhận mới xem xét mua thêm."
                    ),
                    "price_mode": (
                        "Giá bảng điện"
                        if is_live else "Giá đóng cửa nến ngày"
                    ),
                }

        blockers = []

        if early_score < EARLY_MIN_SCORE:
            blockers.append(
                f"Điểm EARLY BUY {early_score}/{EARLY_MIN_SCORE}"
            )

        if extension_pct > EARLY_MAX_EXTENSION_PCT:
            blockers.append(f"Giá cách MA20 {extension_pct:.1f}%")

        if not (
            EARLY_HIGH20_MIN_DISTANCE
            <= proximity
            <= EARLY_HIGH20_MAX_DISTANCE
        ):
            blockers.append("Chưa ở vùng giá trước breakout")

        if not h1_early:
            blockers.append(f"H1 chưa xác nhận: {h1_status}")

        if early_plan is None or early_plan["reward_risk"] < MIN_REWARD_RISK:
            blockers.append("Chưa đạt RR tối thiểu 2:1")

        return {
            "signal": "WAIT",
            "score": max(confirmed_score, early_score),
            "confirmed_score": confirmed_score,
            "early_score": early_score,
            "price": round(price),
            "daily_close": round(daily_close),
            "rsi": round(float(row.RSI), 2),
            "volume_ratio": round(float(row.VOLUME_RATIO), 2),
            "h1_ema9": h1_ema9,
            "h1_ema20": h1_ema20,
            "h1_confirmed": False,
            "h1_status": h1_status,
            "avg_turnover_20d": round(avg_turnover),
            "median_turnover_20d": round(median_turnover),
            "reasons": (
                early_reasons
                if early_score > confirmed_score
                else confirmed_reasons
            ),
            "blockers": blockers,
            "reason": "Chưa đạt đủ điều kiện mua. Tiếp tục theo dõi.",
            "data_date": data_date,
        }

    except Exception as exc:
        return {
            "signal": "WAIT",
            "reason": str(exc),
            "data_date": None,
        }


# =========================================================
# 6. QUÉT THỊ TRƯỜNG
# =========================================================

def scan_market(
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT,
    max_symbols=None,
    progress_callback=None,
    live_mode=False,
    live_quotes=None,
    symbols=None,
):
    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history,
        fetch_stock_hourly,
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

    if live_mode and live_quotes is None:
        raise RuntimeError(
            "Chưa có dữ liệu bảng giá trực tiếp; không thể chạy live mode."
        )

    results = []
    errors = []

    filtered_count = 0
    wait_count = 0

    d1_error_count = 0
    d1_insufficient_count = 0

    liquidity_filtered_count = 0

    h1_attempted_count = 0
    h1_available_count = 0
    h1_unavailable_count = 0
    h1_kbs_error_count = 0
    h1_vci_fallback_count = 0

    total = len(symbols)

    for index, symbol in enumerate(symbols, start=1):
        try:
            if symbol in MANUALLY_BLOCKED_SYMBOLS:
                filtered_count += 1
                continue

            # -------------------------------------------------
            # BƯỚC 1: LẤY D1
            # -------------------------------------------------
            try:
                history = fetch_stock_history(symbol, length=260)
            except Exception as exc:
                d1_error_count += 1
                message = str(exc)

                if "chỉ có" in message.lower() and "phiên" in message.lower():
                    d1_insufficient_count += 1

                errors.append({
                    "symbol": symbol,
                    "stage": "D1",
                    "error": message,
                })
                continue

            # -------------------------------------------------
            # BƯỚC 2: KIỂM TRA 200 PHIÊN VÀ THANH KHOẢN
            # -------------------------------------------------
            try:
                data = calculate_indicators(history)
                row = data.iloc[-1]

            except Exception as exc:
                d1_error_count += 1
                errors.append({
                    "symbol": symbol,
                    "stage": "D1_PROCESS",
                    "error": str(exc),
                })
                continue

            avg_turnover = float(row.AVG_TURNOVER_20D)
            median_turnover = float(row.MEDIAN_TURNOVER_20D)

            if (
                avg_turnover < MIN_AVG_TURNOVER_20D
                or median_turnover < MIN_MEDIAN_TURNOVER_20D
            ):
                filtered_count += 1
                liquidity_filtered_count += 1
                continue

            # -------------------------------------------------
            # BƯỚC 3: CHỈ LẤY H1 SAU KHI ĐẠT THANH KHOẢN
            # -------------------------------------------------
            h1_attempted_count += 1

            hourly = None
            hourly_error = None

            try:
                hourly = fetch_stock_hourly(
                    symbol,
                    lookback="14d",
                )
                h1_available_count += 1

                source = str(hourly.attrs.get("source", "")).upper()

                if source == "VCI":
                    h1_vci_fallback_count += 1

            except Exception as exc:
                hourly_error = str(exc)
                h1_unavailable_count += 1

                error_text = hourly_error.upper()

                if "KBS" in error_text:
                    h1_kbs_error_count += 1

                errors.append({
                    "symbol": symbol,
                    "stage": "H1",
                    "error": hourly_error,
                })

            # -------------------------------------------------
            # BƯỚC 4: PHÂN TÍCH
            # -------------------------------------------------
            quote = None

            if live_mode:
                quote = live_quotes.get(symbol)

                if quote is None:
                    errors.append({
                        "symbol": symbol,
                        "stage": "QUOTE",
                        "error": "Không có giá trực tiếp cho mã này.",
                    })
                    continue

            result = analyze_stock(
                history,
                capital=capital,
                risk_percent=risk_percent,
                live_quote=quote,
                symbol=symbol,
                hourly_df=hourly,
                hourly_error=hourly_error,
            )

            result["symbol"] = symbol

            if result.get("signal") in ("BUY", "EARLY_BUY"):
                results.append(result)

            elif result.get("signal") == "FILTERED":
                filtered_count += 1

            elif result.get("signal") == "WAIT":
                wait_count += 1

                if result.get("data_date") is None:
                    errors.append({
                        "symbol": symbol,
                        "stage": "ANALYSIS",
                        "error": result.get(
                            "reason",
                            "Không phân tích được.",
                        ),
                    })

        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "stage": "UNKNOWN",
                "error": str(exc),
            })

        finally:
            if progress_callback:
                progress_callback(index, total)

    results.sort(
        key=lambda x: (
            1 if x.get("signal") == "BUY" else 0,
            x.get("score", 0),
            x.get("reward_risk", 0),
            x.get("volume_ratio", 0),
        ),
        reverse=True,
    )

    confirmed_count = sum(
        1 for x in results if x.get("signal") == "BUY"
    )

    early_count = sum(
        1 for x in results if x.get("signal") == "EARLY_BUY"
    )

    return {
        "results": results,

        "scanned": total,

        "buy_count": len(results),
        "confirmed_buy_count": confirmed_count,
        "early_buy_count": early_count,

        "filtered_count": filtered_count,
        "liquidity_filtered_count": liquidity_filtered_count,
        "wait_count": wait_count,

        "d1_error_count": d1_error_count,
        "d1_insufficient_count": d1_insufficient_count,

        "h1_attempted_count": h1_attempted_count,
        "h1_available_count": h1_available_count,
        "h1_unavailable_count": h1_unavailable_count,
        "h1_kbs_error_count": h1_kbs_error_count,
        "h1_vci_fallback_count": h1_vci_fallback_count,

        "error_count": len(errors),
        "errors": errors,
    }
