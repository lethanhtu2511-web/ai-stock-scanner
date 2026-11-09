import math

import pandas as pd
import numpy as np


MIN_SCORE = 78
MIN_REWARD_RISK = 2.0

MIN_AVG_TURNOVER_20D = 5_000_000_000
MIN_MEDIAN_TURNOVER_20D = 3_000_000_000

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002

MANUALLY_BLOCKED_SYMBOLS = {
    "NVB": (
        "Ứng dụng giao dịch hiển thị thuộc diện cảnh báo "
        "(09/11/2026)"
    )
}


def calculate_indicators(df):
    data = df.copy()

    required = [
        "Date", "Open", "High", "Low", "Close", "Volume"
    ]

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
            f"INSUFFICIENT_HISTORY: Chỉ có {len(data)} phiên, "
            "yêu cầu tối thiểu 200 phiên."
        )

    data["TURNOVER"] = data["Close"] * data["Volume"]
    data["AVG_TURNOVER_20D"] = data["TURNOVER"].rolling(20).mean()
    data["MEDIAN_TURNOVER_20D"] = data["TURNOVER"].rolling(20).median()

    close = data["Close"]

    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()
    data["MA200"] = close.rolling(200).mean()

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

    data["VOLUME_MA20"] = data["Volume"].rolling(20).mean()
    data["VOLUME_RATIO"] = (
        data["Volume"] / data["VOLUME_MA20"].replace(0, np.nan)
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    data["ATR14"] = true_range.rolling(14).mean()
    data["HIGH20_PREVIOUS"] = (
        data["High"].shift(1).rolling(20).max()
    )
    data["HIGH50_PREVIOUS"] = (
        data["High"].shift(1).rolling(50).max()
    )

    return data.dropna().reset_index(drop=True)


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
        reasons.append("Khối lượng cải thiện so với TB20")

    if row.TURNOVER > row.AVG_TURNOVER_20D * 1.2:
        score += 10
        reasons.append("Giá trị giao dịch tăng so với TB20")

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
        reasons.append("Áp sát vùng đỉnh 20 phiên")

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
        reasons.append("Biến động ở mức chấp nhận được")
    elif atr_pct <= 6.5:
        score += 5
        reasons.append("Biến động cần thận trọng")

    return min(int(score), 100), reasons


def calculate_target_price(price, stop_loss, resistance=None):
    risk_per_share = (
        price * (1 + BUY_FEE + SLIPPAGE)
        - stop_loss * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    )

    if risk_per_share <= 0:
        return None

    target = price

    for _ in range(5000):
        reward = (
            target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        if reward >= MIN_REWARD_RISK * risk_per_share:
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


def _result_filtered(reason, row=None):
    result = {
        "signal": "FILTERED",
        "reason": reason
    }

    if row is not None:
        try:
            result["data_date"] = pd.Timestamp(row.Date).strftime("%d/%m/%Y")
        except Exception:
            pass

    return result


def analyze_stock(df, symbol=None):
    try:
        if symbol and symbol.upper() in MANUALLY_BLOCKED_SYMBOLS:
            return _result_filtered(
                f"{symbol.upper()} bị chặn: "
                f"{MANUALLY_BLOCKED_SYMBOLS[symbol.upper()]}"
            )

        data = calculate_indicators(df)
        row = data.iloc[-1].copy()

        price = float(row.Close)
        avg_turnover = float(row.AVG_TURNOVER_20D)
        median_turnover = float(row.MEDIAN_TURNOVER_20D)

        data_date = pd.Timestamp(row.Date).strftime("%d/%m/%Y")

        if (
            not np.isfinite(avg_turnover)
            or not np.isfinite(median_turnover)
            or avg_turnover < MIN_AVG_TURNOVER_20D
            or median_turnover < MIN_MEDIAN_TURNOVER_20D
        ):
            return _result_filtered(
                "Thanh khoản không đạt: "
                f"GTGD bình quân {avg_turnover / 1e9:.2f} tỷ/ngày "
                "(yêu cầu ≥ 5 tỷ), "
                f"trung vị {median_turnover / 1e9:.2f} tỷ/ngày "
                "(yêu cầu ≥ 3 tỷ).",
                row
            )

        today = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").date()
        row_date = pd.Timestamp(row.Date).date()
        age_days = (today - row_date).days

        if age_days > 7:
            return {
                "signal": "WAIT",
                "reason": (
                    f"Dữ liệu nến đã cũ {age_days} ngày; "
                    "không phát tín hiệu mua."
                ),
                "data_date": data_date
            }

        if not np.isfinite(float(row.ATR14)) or float(row.ATR14) <= 0:
            raise ValueError("ATR không hợp lệ.")

        ma20 = float(row.MA20)
        extension_pct = (price / ma20 - 1) * 100 if ma20 > 0 else 999

        if extension_pct > 8:
            return {
                "signal": "WAIT",
                "reason": f"Giá cách MA20 {extension_pct:.1f}%, không mua đuổi.",
                "data_date": data_date
            }

        if float(row.RSI) > 70:
            return {
                "signal": "WAIT",
                "reason": "RSI trên 70: cổ phiếu đang quá nóng, chờ điều chỉnh.",
                "data_date": data_date
            }

        score, reasons = calculate_score(row)

        stop = price - 1.5 * float(row.ATR14)

        if stop <= 0 or stop >= price:
            raise ValueError("Giá cắt lỗ không hợp lệ.")

        target = calculate_target_price(
            price,
            stop,
            resistance=float(row.HIGH50_PREVIOUS)
        )

        if target is None:
            return {
                "signal": "WAIT",
                "reason": (
                    "Kháng cự gần không đủ dư địa "
                    "để đạt tỷ lệ lợi nhuận/rủi ro tối thiểu 2:1."
                ),
                "data_date": data_date
            }

        risk_share = (
            price * (1 + BUY_FEE + SLIPPAGE)
            - stop * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
        )

        reward_share = (
            target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        rr = reward_share / risk_share if risk_share > 0 else 0

        conditions = [
            score >= MIN_SCORE,
            45 <= float(row.RSI) <= 70,
            price > ma20,
            rr >= MIN_REWARD_RISK,
            avg_turnover >= MIN_AVG_TURNOVER_20D,
            median_turnover >= MIN_MEDIAN_TURNOVER_20D,
            extension_pct <= 8
        ]

        signal = "BUY" if all(conditions) else "WAIT"

        reason = (
            "Đạt điều kiện kỹ thuật, thanh khoản "
            "và tỷ lệ lợi nhuận/rủi ro."
            if signal == "BUY"
            else "Chưa đạt đầy đủ điều kiện mua."
        )

        return {
            "signal": signal,
            "score": score,
            "price": round(price),
            "daily_close": round(price),
            "stop_loss": round(stop),
            "target_price": round(target),
            "reward_risk": round(rr, 2),
            "rsi": round(float(row.RSI), 2),
            "volume_ratio": round(float(row.VOLUME_RATIO), 2),
            "avg_turnover_20d": round(avg_turnover),
            "median_turnover_20d": round(median_turnover),
            "reasons": reasons,
            "data_date": data_date,
            "message": reason,
            "reason": reason,
            "price_mode": "Giá đóng cửa nến ngày"
        }

    except Exception as exc:
        return {
            "signal": "WAIT",
            "reason": str(exc),
            "data_date": None
        }


def scan_market(max_symbols=None, progress_callback=None):
    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history
    )

    symbols = fetch_exchange_symbols()

    if max_symbols is not None:
        symbols = symbols[:max(1, int(max_symbols))]

    results = []
    errors = []
    filtered_count = 0
    insufficient_history_count = 0

    total = len(symbols)

    for index, symbol in enumerate(symbols, start=1):
        try:
            if symbol.upper() in MANUALLY_BLOCKED_SYMBOLS:
                filtered_count += 1
            else:
                history = fetch_stock_history(symbol, length=260)

                result = analyze_stock(
                    history,
                    symbol=symbol
                )

                result["symbol"] = symbol

                if result.get("signal") == "BUY":
                    results.append(result)

                elif result.get("signal") == "FILTERED":
                    filtered_count += 1

                elif result.get("data_date") is None:
                    error_message = result.get("reason", "")

                    if "INSUFFICIENT_HISTORY" in error_message:
                        insufficient_history_count += 1
                    else:
                        errors.append({
                            "symbol": symbol,
                            "error": error_message
                        })

        except Exception as exc:
            error_message = str(exc)

            if "200 phiên" in error_message or "INSUFFICIENT_HISTORY" in error_message:
                insufficient_history_count += 1
            else:
                errors.append({
                    "symbol": symbol,
                    "error": error_message
                })

        if progress_callback:
            progress_callback(index, total)

    results.sort(
        key=lambda x: (
            x.get("score", 0),
            x.get("reward_risk", 0),
            x.get("volume_ratio", 0)
        ),
        reverse=True
    )

    return {
        "results": results,
        "scanned": total,
        "buy_count": len(results),
        "filtered_count": filtered_count,
        "insufficient_history_count": insufficient_history_count,
        "error_count": len(errors),
        "errors": errors
    }
