import math
import pandas as pd
import numpy as np

INITIAL_CAPITAL = 10_000_000
RISK_PERCENT = 0.02

# BUY xác nhận
MIN_SCORE = 78
MIN_REWARD_RISK = 2.0

# EARLY BUY
EARLY_MIN_SCORE = 70
EARLY_MAX_EXTENSION_PCT = 5.0
EARLY_HIGH20_MIN_DISTANCE = 0.96
EARLY_HIGH20_MAX_DISTANCE = 1.015

# Thanh khoản tối thiểu
MIN_AVG_TURNOVER_20D = 5_000_000_000
MIN_MEDIAN_TURNOVER_20D = 3_000_000_000

LOT_SIZE = 100
BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

MANUALLY_BLOCKED_SYMBOLS = {
    "NVB": "Ứng dụng giao dịch hiển thị thuộc diện cảnh báo (09/11/2026)"
}


def calculate_indicators(df):
    data = df.copy()
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Thiếu dữ liệu OHLCV: {missing}")

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

    if len(data) < 200:
        raise ValueError(
            f"Không đủ 200 phiên dữ liệu (hiện có {len(data)})."
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
    data["RSI"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    data["VOLUME_MA5"] = data["Volume"].rolling(5).mean()
    data["VOLUME_MA20"] = data["Volume"].rolling(20).mean()

    data["VOLUME_RATIO"] = (
        data["Volume"] /
        data["VOLUME_MA20"].replace(0, np.nan)
    )

    data["VOLUME_DRYUP_RATIO"] = (
        data["VOLUME_MA5"] /
        data["VOLUME_MA20"].replace(0, np.nan)
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

    # Chỉ sử dụng đỉnh các phiên trước
    data["HIGH20_PREVIOUS"] = (
        data["High"].shift(1).rolling(20).max()
    )

    data["HIGH50_PREVIOUS"] = (
        data["High"].shift(1).rolling(50).max()
    )

    data["RANGE20"] = (
        data["High"].rolling(20).max()
        - data["Low"].rolling(20).min()
    ) / close * 100

    data["RANGE20_PREVIOUS_5D"] = (
        data["RANGE20"].shift(5)
    )

    return (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )


def calculate_score(row):
    """Điểm kỹ thuật BUY xác nhận 0-100."""
    score = 0
    reasons = []

    # Xu hướng
    if row.Close > row.MA20:
        score += 7
        reasons.append("Giá trên MA20")

    if row.MA20 > row.MA50:
        score += 7
        reasons.append("MA20 trên MA50")

    if row.MA50 > row.MA200:
        score += 6
        reasons.append("MA50 trên MA200")

    # Dòng tiền
    if row.VOLUME_RATIO >= 1.5:
        score += 10
        reasons.append("Khối lượng cao hơn TB20 ít nhất 50%")

    elif row.VOLUME_RATIO >= 1.2:
        score += 7
        reasons.append("Khối lượng cải thiện so với TB20")

    if row.TURNOVER > row.AVG_TURNOVER_20D * 1.2:
        score += 10
        reasons.append("Giá trị giao dịch tăng")

    if row.Close > row.Open and row.Volume >= row.VOLUME_MA20:
        score += 5
        reasons.append("Nến tăng có khối lượng xác nhận")

    # Động lượng
    if row.MACD > row.MACD_SIGNAL:
        score += 7
        reasons.append("MACD tích cực")

    if 52 <= row.RSI <= 65:
        score += 8
        reasons.append("RSI trong vùng khỏe")

    elif 48 <= row.RSI < 52 or 65 < row.RSI <= 68:
        score += 5
        reasons.append("RSI chấp nhận được")

    # Breakout
    if row.Close > row.HIGH20_PREVIOUS:
        score += 15
        reasons.append("Vượt đỉnh 20 phiên")

    elif row.Close >= row.HIGH20_PREVIOUS * 0.97:
        score += 8
        reasons.append("Áp sát đỉnh 20 phiên")

    # Chất lượng nến
    candle_range = max(
        float(row.High - row.Low), 1e-9
    )

    close_position = (
        float(row.Close) - float(row.Low)
    ) / candle_range

    if row.Close > row.Open and close_position >= 0.75:
        score += 10
        reasons.append("Nến tăng đóng cửa gần mức cao nhất")

    elif row.Close > row.Open and close_position >= 0.55:
        score += 6
        reasons.append("Nến tăng tương đối tốt")

    # Biến động
    atr_pct = (
        float(row.ATR14) /
        max(float(row.Close), 1e-9) * 100
    )

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
    """Chấm điểm tích lũy và chuẩn bị breakout."""
    score = 0
    reasons = []

    high20 = float(row.HIGH20_PREVIOUS)

    proximity = (
        float(row.Close) / high20
        if high20 > 0 else 0
    )

    ma20_slope = (
        (float(row.MA20) / float(row.MA20_PREVIOUS_5D) - 1) * 100
        if float(row.MA20_PREVIOUS_5D) > 0 else -99
    )

    if row.Close >= row.MA20:
        score += 12
        reasons.append("Giá giữ trên MA20")

    if row.MA20 >= row.MA50 * 0.99:
        score += 8
        reasons.append("MA20 áp sát hoặc nằm trên MA50")

    if ma20_slope >= 0:
        score += 8
        reasons.append("MA20 đi ngang hoặc tăng")

    if EARLY_HIGH20_MIN_DISTANCE <= proximity <= EARLY_HIGH20_MAX_DISTANCE:
        score += 18
        reasons.append("Giá sát đỉnh 20 phiên, chưa bứt quá xa")

    elif 0.93 <= proximity < EARLY_HIGH20_MIN_DISTANCE:
        score += 10
        reasons.append("Giá đang tiến gần đỉnh 20 phiên")

    if row.VOLUME_DRYUP_RATIO <= 0.85:
        score += 12
        reasons.append("Khối lượng 5 phiên co lại")

    elif row.VOLUME_DRYUP_RATIO <= 1.0:
        score += 8
        reasons.append("Khối lượng tích lũy không tăng nóng")

    if 0.75 <= row.VOLUME_RATIO <= 1.8:
        score += 8
        reasons.append("Khối lượng hiện tại ở vùng chấp nhận")

    if row.ATR14_PCT <= 4.5:
        score += 8
        reasons.append("Biến động ATR không quá lớn")

    if row.RANGE20 <= row.RANGE20_PREVIOUS_5D * 1.05:
        score += 8
        reasons.append("Biên độ 20 phiên thu hẹp hoặc ổn định")

    if 48 <= row.RSI <= 65:
        score += 10
        reasons.append("RSI khỏe nhưng chưa quá nóng")

    elif 45 <= row.RSI < 48 or 65 < row.RSI <= 68:
        score += 5
        reasons.append("RSI trong vùng theo dõi")

    if row.MACD >= row.MACD_SIGNAL * 0.98:
        score += 8
        reasons.append("MACD cải thiện")

    candle_range = max(
        float(row.High - row.Low), 1e-9
    )

    close_position = (
        float(row.Close) - float(row.Low)
    ) / candle_range

    if row.Close >= row.Open and close_position >= 0.60:
        score += 8
        reasons.append("Nến giữ giá tốt")

    return min(int(score), 100), reasons


def calculate_target_price(price, stop_loss, resistance=None):
    """Tính mục tiêu tối thiểu 2R sau phí."""
    risk_per_share = (
        price * (1 + BUY_FEE + SLIPPAGE)
        - stop_loss * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    )

    if risk_per_share <= 0:
        return None

    target = price
    min_target = None

    for _ in range(5000):
        reward = (
            target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        if reward >= MIN_REWARD_RISK * risk_per_share:
            min_target = math.ceil(target / 100) * 100
            break

        target += 100

    if min_target is None:
        return None

    if (
        resistance is not None
        and np.isfinite(resistance)
        and resistance > price * 1.01
    ):
        resistance_price = (
            math.floor(float(resistance) / 100) * 100
        )

        if resistance_price < min_target:
            return None

        return resistance_price

    return min_target


def _result_filtered(reason, row=None):
    result = {
        "signal": "FILTERED",
        "reason": reason,
    }

    if row is not None:
        try:
            result["data_date"] = pd.Timestamp(
                row.Date
            ).strftime("%d/%m/%Y")
        except Exception:
            pass

    return result


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

    quantity_risk = (
        int(risk_budget / risk_share)
        if risk_share > 0 else 0
    )

    quantity_capital = int(
        capital / (price * (1 + BUY_FEE + SLIPPAGE))
    )

    quantity = min(quantity_risk, quantity_capital)
    quantity = (quantity // LOT_SIZE) * LOT_SIZE

    capital_used = (
        quantity * price * (1 + BUY_FEE + SLIPPAGE)
    )

    risk_money = quantity * risk_share

    return {
        "stop_loss": round(stop),
        "target_price": round(target),
        "reward_risk": round(rr, 2),
        "quantity": quantity,
        "capital_used": round(capital_used),
        "risk_money": round(risk_money),
        "risk_budget": round(risk_budget),
    }


def _hourly_trend(hourly_df):
    """Kiểm tra xu hướng EMA9/EMA20 H1."""
    if hourly_df is None or len(hourly_df) < 25:
        return None, None, False, False

    frame = hourly_df.copy()

    close_col = next(
        (
            c for c in frame.columns
            if str(c).lower() in (
                "close", "closing_price", "price"
            )
        ),
        None,
    )

    if close_col is None:
        return None, None, False, False

    close = pd.to_numeric(
        frame[close_col], errors="coerce"
    ).dropna()

    if len(close) < 25:
        return None, None, False, False

    ema9 = float(
        close.ewm(span=9, adjust=False).mean().iloc[-1]
    )

    ema20 = float(
        close.ewm(span=20, adjust=False).mean().iloc[-1]
    )

    if (
        not np.isfinite(ema9)
        or not np.isfinite(ema20)
        or ema20 <= 0
    ):
        return None, None, False, False

    return (
        ema9,
        ema20,
        ema9 > ema20,
        ema9 >= ema20 * 0.995,
    )


def analyze_stock(
    df,
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT,
    live_quote=None,
    symbol=None,
    hourly_df=None,
):
    try:
        if symbol and symbol.upper() in MANUALLY_BLOCKED_SYMBOLS:
            return _result_filtered(
                f"{symbol.upper()} bị chặn: "
                f"{MANUALLY_BLOCKED_SYMBOLS[symbol.upper()]}"
            )

        if hourly_df is None and symbol:
            try:
                from data_provider import fetch_stock_hourly
                hourly_df = fetch_stock_hourly(
                    symbol, lookback="14d"
                )
            except Exception:
                hourly_df = None

        h1_ema9, h1_ema20, h1_bullish, h1_early = (
            _hourly_trend(hourly_df)
        )

        data = calculate_indicators(df)
        row = data.iloc[-1].copy()

        daily_close = float(row.Close)
        is_live = live_quote is not None

        avg_turnover = float(row.AVG_TURNOVER_20D)
        median_turnover = float(row.MEDIAN_TURNOVER_20D)

        if (
            not np.isfinite(avg_turnover)
            or not np.isfinite(median_turnover)
            or avg_turnover < MIN_AVG_TURNOVER_20D
            or median_turnover < MIN_MEDIAN_TURNOVER_20D
        ):
            return _result_filtered(
                "Thanh khoản không đạt: "
                f"GTGD bình quân 20 phiên "
                f"{avg_turnover / 1e9:.2f} tỷ/ngày "
                f"(yêu cầu ≥ "
                f"{MIN_AVG_TURNOVER_20D / 1e9:.0f} tỷ), "
                f"trung vị {median_turnover / 1e9:.2f} tỷ/ngày "
                f"(yêu cầu ≥ "
                f"{MIN_MEDIAN_TURNOVER_20D / 1e9:.0f} tỷ).",
                row,
            )

        if is_live:
            price = float(live_quote.get("last_price", 0))

            if not np.isfinite(price) or price <= 0:
                raise ValueError("Giá bảng điện không hợp lệ.")

            reference = live_quote.get("reference_price")

            if (
                reference is not None
                and pd.notna(reference)
                and float(reference) > 0
            ):
                if abs(price / float(reference) - 1) > 0.20:
                    raise ValueError(
                        "Giá bảng điện lệch bất thường so với giá tham chiếu."
                    )

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
                "reason": (
                    f"Dữ liệu nến đã cũ {age_days} ngày; "
                    "không phát tín hiệu mua."
                ),
                "data_date": data_date,
            }

        if not np.isfinite(float(row.ATR14)) or float(row.ATR14) <= 0:
            raise ValueError("ATR không hợp lệ.")

        ma20 = float(row.MA20)

        extension_pct = (
            (price / ma20 - 1) * 100
            if ma20 > 0 else 999
        )

        previous_close = float(data.iloc[-2].Close)

        day_change_pct = (
            (price / previous_close - 1) * 100
            if previous_close > 0 else 0
        )

        if is_live and day_change_pct > 7:
            return {
                "signal": "WAIT",
                "reason": (
                    f"Giá tăng {day_change_pct:.1f}% "
                    "so với đóng cửa trước, không mua đuổi."
                ),
                "data_date": data_date,
            }

        confirmed_score, confirmed_reasons = calculate_score(row)
        early_score, early_reasons = calculate_early_score(row)

        # BUY xác nhận
        plan = None

        if extension_pct <= 8 and float(row.RSI) <= 70:
            plan = _risk_plan(
                price, row, capital, risk_percent
            )

            if plan is not None:
                confirmed_conditions = [
                    confirmed_score >= MIN_SCORE,
                    45 <= float(row.RSI) <= 70,
                    price > ma20,
                    plan["reward_risk"] >= MIN_REWARD_RISK,
                    plan["quantity"] >= LOT_SIZE,
                    plan["risk_money"] <= plan["risk_budget"],
                    avg_turnover >= MIN_AVG_TURNOVER_20D,
                    median_turnover >= MIN_MEDIAN_TURNOVER_20D,
                    extension_pct <= 8,
                    h1_bullish,
                ]

                if all(confirmed_conditions):
                    result = {
                        "signal": "BUY",
                        "signal_type": "BUY_CONFIRMED",
                        "score": confirmed_score,
                        "early_score": early_score,
                        "price": round(price),
                        "daily_close": round(daily_close),
                        **plan,
                        "rsi": round(float(row.RSI), 2),
                        "volume_ratio": round(
                            float(row.VOLUME_RATIO), 2
                        ),
                        "h1_ema9": (
                            round(h1_ema9, 4)
                            if h1_ema9 is not None else None
                        ),
                        "h1_ema20": (
                            round(h1_ema20, 4)
                            if h1_ema20 is not None else None
                        ),
                        "h1_confirmed": h1_bullish,
                        "avg_turnover_20d": round(avg_turnover),
                        "median_turnover_20d": round(median_turnover),
                        "reasons": confirmed_reasons,
                        "data_date": data_date,
                        "message": (
                            "BUY xác nhận: đạt bộ lọc kỹ thuật, "
                            "thanh khoản và quản trị rủi ro."
                        ),
                        "reason": "Đạt đầy đủ điều kiện BUY xác nhận.",
                        "price_mode": (
                            "Giá khớp gần nhất từ bảng giá"
                            if is_live else "Giá đóng cửa nến ngày"
                        ),
                    }

                    if is_live:
                        result["reference_price"] = (
                            live_quote.get("reference_price")
                        )
                        result["exchange"] = (
                            live_quote.get("exchange")
                        )

                    return result

        # EARLY BUY
        high20 = float(row.HIGH20_PREVIOUS)

        proximity = (
            price / high20 if high20 > 0 else 0
        )

        early_hard_gates = [
            early_score >= EARLY_MIN_SCORE,
            price >= ma20 * 0.99,
            45 <= float(row.RSI) <= 68,
            extension_pct <= EARLY_MAX_EXTENSION_PCT,
            EARLY_HIGH20_MIN_DISTANCE <= proximity
            <= EARLY_HIGH20_MAX_DISTANCE,
            h1_early,
        ]

        early_plan = _risk_plan(
            price, row, capital, risk_percent
        )

        if early_plan is not None:
            early_hard_gates.extend([
                early_plan["reward_risk"] >= MIN_REWARD_RISK,
                early_plan["quantity"] >= LOT_SIZE,
                early_plan["risk_money"] <= early_plan["risk_budget"],
            ])
        else:
            early_hard_gates.append(False)

        if all(early_hard_gates):
            result = {
                "signal": "EARLY_BUY",
                "signal_type": "EARLY_ENTRY",
                "score": early_score,
                "confirmed_score": confirmed_score,
                "price": round(price),
                "daily_close": round(daily_close),
                **early_plan,
                "rsi": round(float(row.RSI), 2),
                "volume_ratio": round(
                    float(row.VOLUME_RATIO), 2
                ),
                "h1_ema9": (
                    round(h1_ema9, 4)
                    if h1_ema9 is not None else None
                ),
                "h1_ema20": (
                    round(h1_ema20, 4)
                    if h1_ema20 is not None else None
                ),
                "h1_confirmed": h1_early,
                "avg_turnover_20d": round(avg_turnover),
                "median_turnover_20d": round(median_turnover),
                "reasons": early_reasons,
                "data_date": data_date,
                "message": (
                    "EARLY BUY: cổ phiếu đang tích lũy sát vùng "
                    "breakout; giải ngân thăm dò, rủi ro cao hơn "
                    "BUY xác nhận."
                ),
                "reason": (
                    "Đạt bộ lọc mua sớm và quản trị rủi ro; "
                    "cần xác nhận thêm khi breakout."
                ),
                "price_mode": (
                    "Giá khớp gần nhất từ bảng giá"
                    if is_live else "Giá đóng cửa nến ngày"
                ),
            }

            if is_live:
                result["reference_price"] = (
                    live_quote.get("reference_price")
                )
                result["exchange"] = live_quote.get("exchange")

            return result

        blockers = []

        if early_score < EARLY_MIN_SCORE:
            blockers.append(
                f"Điểm mua sớm {early_score}/{EARLY_MIN_SCORE}"
            )

        if extension_pct > EARLY_MAX_EXTENSION_PCT:
            blockers.append(
                f"Giá cách MA20 {extension_pct:.1f}%"
            )

        if not (
            EARLY_HIGH20_MIN_DISTANCE
            <= proximity
            <= EARLY_HIGH20_MAX_DISTANCE
        ):
            blockers.append(
                "Chưa ở vùng giá phù hợp trước breakout"
            )

        if plan is None:
            blockers.append(
                "Chưa đạt RR tối thiểu 2:1 "
                "hoặc khối lượng mua tối thiểu"
            )

        return {
            "signal": "WAIT",
            "score": max(confirmed_score, early_score),
            "confirmed_score": confirmed_score,
            "early_score": early_score,
            "price": round(price),
            "daily_close": round(daily_close),
            "rsi": round(float(row.RSI), 2),
            "volume_ratio": round(float(row.VOLUME_RATIO), 2),
            "avg_turnover_20d": round(avg_turnover),
            "median_turnover_20d": round(median_turnover),
            "reasons": (
                early_reasons
                if early_score > confirmed_score
                else confirmed_reasons
            ),
            "blockers": blockers,
            "reason": (
                "Chưa đạt đầy đủ điều kiện mua; tiếp tục theo dõi, "
                "không mua chỉ dựa vào điểm số."
            ),
            "message": "Chưa có tín hiệu mua đạt chuẩn.",
            "data_date": data_date,
        }

    except Exception as exc:
        return {
            "signal": "WAIT",
            "reason": str(exc),
            "data_date": None,
        }


def scan_market(
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT,
    max_symbols=None,
    progress_callback=None,
    live_mode=False,
    live_quotes=None,
):
    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history,
    )

    try:
        from data_provider import fetch_stock_hourly
    except ImportError:
        fetch_stock_hourly = None

    symbols = fetch_exchange_symbols()

    if max_symbols is not None:
        symbols = symbols[:max(1, int(max_symbols))]

    if live_mode and live_quotes is None:
        raise RuntimeError(
            "Đang trong khung giờ giao dịch nhưng chưa có bảng giá."
        )

    results = []
    errors = []
    filtered_count = 0
    total = len(symbols)

    for index, symbol in enumerate(symbols, start=1):
        try:
            if symbol.upper() in MANUALLY_BLOCKED_SYMBOLS:
                filtered_count += 1
                continue

            history = fetch_stock_history(
                symbol, length=260
            )

            quote = None

            if live_mode:
                quote = live_quotes.get(symbol)

                if quote is None:
                    raise ValueError(
                        "Không nhận được giá bảng điện cho mã này."
                    )

            hourly = None

            if fetch_stock_hourly is not None:
                try:
                    hourly = fetch_stock_hourly(
                        symbol, lookback="14d"
                    )
                except Exception:
                    hourly = None

            result = analyze_stock(
                history,
                capital=capital,
                risk_percent=risk_percent,
                live_quote=quote,
                symbol=symbol,
                hourly_df=hourly,
            )

            result["symbol"] = symbol

            if result.get("signal") in ("BUY", "EARLY_BUY"):
                results.append(result)

            elif result.get("signal") == "FILTERED":
                filtered_count += 1

            elif result.get("data_date") is None:
                errors.append({
                    "symbol": symbol,
                    "error": result.get(
                        "reason",
                        "Không phân tích được dữ liệu."
                    ),
                })

        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "error": str(exc),
            })

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

    early_count = sum(
        1 for x in results
        if x.get("signal") == "EARLY_BUY"
    )

    confirmed_count = sum(
        1 for x in results
        if x.get("signal") == "BUY"
    )

    return {
        "results": results,
        "scanned": total,
        "buy_count": len(results),
        "confirmed_buy_count": confirmed_count,
        "early_buy_count": early_count,
        "filtered_count": filtered_count,
        "error_count": len(errors),
        "errors": errors,
    }
