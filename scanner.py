import math
import pandas as pd
import numpy as np


INITIAL_CAPITAL = 10_000_000
RISK_PERCENT = 0.02

MIN_SCORE = 78
MIN_REWARD_RISK = 2.0

LOT_SIZE = 100

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002


def calculate_indicators(df):

    data = df.copy()

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if any(c not in data.columns for c in required):
        raise ValueError(
            "Thiếu dữ liệu OHLCV."
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    for col in required[1:]:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = (
        data.dropna(subset=required)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(data) < 200:
        raise ValueError(
            "Không đủ 200 phiên dữ liệu."
        )

    close = data["Close"]

    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()
    data["MA200"] = close.rolling(200).mean()

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

    data["RSI"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["VOLUME_MA20"] = (
        data["Volume"].rolling(20).mean()
    )

    data["VOLUME_RATIO"] = (
        data["Volume"]
        / data["VOLUME_MA20"].replace(0, np.nan)
    )

    previous_close = close.shift(1)

    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    data["ATR14"] = tr.rolling(14).mean()

    data["HIGH20_PREVIOUS"] = (
        data["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    return data.dropna().reset_index(drop=True)


def calculate_score(row):

    score = 0
    reasons = []

    conditions = [
        (
            row.Close > row.MA20,
            10,
            "Giá trên MA20"
        ),
        (
            row.MA20 > row.MA50,
            10,
            "MA20 trên MA50"
        ),
        (
            row.MA50 > row.MA200,
            10,
            "MA50 trên MA200"
        ),
        (
            row.MACD > row.MACD_SIGNAL,
            15,
            "MACD tích cực"
        ),
        (
            row.Close > row.HIGH20_PREVIOUS,
            15,
            "Breakout đỉnh 20 phiên"
        )
    ]

    for passed, points, label in conditions:

        if passed:
            score += points
            reasons.append(label)

    if 50 <= row.RSI <= 65:

        score += 15
        reasons.append("RSI vùng tích cực")

    elif 45 <= row.RSI < 50 or 65 < row.RSI <= 70:

        score += 8
        reasons.append("RSI chấp nhận được")

    if row.VOLUME_RATIO >= 1.5:

        score += 15
        reasons.append("Thanh khoản tăng mạnh")

    elif row.VOLUME_RATIO >= 1.2:

        score += 10
        reasons.append("Thanh khoản cải thiện")

    return min(round(score / 90 * 100), 100), reasons


def calculate_target_price(price, stop_loss):

    risk = (
        price * (1 + BUY_FEE + SLIPPAGE)
        - stop_loss * (
            1 - SELL_FEE - SELL_TAX - SLIPPAGE
        )
    )

    if risk <= 0:
        return None

    target = price

    for _ in range(3000):

        reward = (
            target * (
                1 - SELL_FEE - SELL_TAX - SLIPPAGE
            )
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        if reward >= MIN_REWARD_RISK * risk:

            return math.ceil(target / 100) * 100

        target += 100

    return None


def analyze_stock(
    df,
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT
):

    try:

        data = calculate_indicators(df)

        row = data.iloc[-1]

        price = float(row.Close)
        atr = float(row.ATR14)

        score, reasons = calculate_score(row)

        data_date = row.Date.strftime("%d/%m/%Y")

        age = (
            pd.Timestamp.now().normalize()
            - row.Date.normalize()
        ).days

        if age > 10:

            return {
                "signal": "WAIT",
                "reason": "Dữ liệu quá cũ.",
                "data_date": data_date
            }

        stop = price - 1.5 * atr

        if stop <= 0:
            raise ValueError(
                "Giá cắt lỗ không hợp lệ."
            )

        target = calculate_target_price(
            price,
            stop
        )

        if target is None:
            raise ValueError(
                "Không tính được giá mục tiêu."
            )

        risk_share = (
            price * (1 + BUY_FEE + SLIPPAGE)
            - stop * (
                1 - SELL_FEE - SELL_TAX - SLIPPAGE
            )
        )

        reward_share = (
            target * (
                1 - SELL_FEE - SELL_TAX - SLIPPAGE
            )
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        rr = (
            reward_share / risk_share
            if risk_share > 0 else 0
        )

        risk_budget = capital * risk_percent

        quantity_risk = (
            int(risk_budget / risk_share)
            if risk_share > 0 else 0
        )

        quantity_capital = int(
            capital / (
                price * (1 + BUY_FEE + SLIPPAGE)
            )
        )

        quantity = min(
            quantity_risk,
            quantity_capital
        )

        quantity = (
            quantity // LOT_SIZE
        ) * LOT_SIZE

        capital_used = (
            quantity * price * (1 + BUY_FEE + SLIPPAGE)
        )

        risk_money = quantity * risk_share

        conditions = [
            score >= MIN_SCORE,
            45 <= float(row.RSI) <= 70,
            price > float(row.MA20),
            rr >= MIN_REWARD_RISK,
            quantity >= LOT_SIZE,
            risk_money <= risk_budget
        ]

        signal = (
            "BUY"
            if all(conditions)
            else "WAIT"
        )

        reason = (
            "Đạt các điều kiện kỹ thuật và quản trị rủi ro."
            if signal == "BUY"
            else "Chưa đạt đầy đủ điều kiện mua."
        )

        return {
            "signal": signal,
            "score": score,
            "price": round(price),
            "stop_loss": round(stop),
            "target_price": round(target),
            "quantity": quantity,
            "capital_used": round(capital_used),
            "risk_money": round(risk_money),
            "reward_risk": round(rr, 2),
            "rsi": round(float(row.RSI), 2),
            "volume_ratio": round(
                float(row.VOLUME_RATIO), 2
            ),
            "reasons": reasons,
            "data_date": data_date,
            "message": reason,
            "reason": reason
        }

    except Exception as exc:

        return {
            "signal": "WAIT",
            "reason": str(exc),
            "data_date": None
        }


def scan_market(
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT,
    max_symbols=None,
    progress_callback=None
):

    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history
    )

    symbols = fetch_exchange_symbols()

    if max_symbols is not None:

        symbols = symbols[
            :max(1, int(max_symbols))
        ]

    results = []
    errors = []

    total = len(symbols)

    for index, symbol in enumerate(
        symbols,
        start=1
    ):

        try:

            history = fetch_stock_history(
                symbol,
                length=260
            )

            result = analyze_stock(
                history,
                capital=capital,
                risk_percent=risk_percent
            )

            result["symbol"] = symbol

            if result.get("signal") == "BUY":

                results.append(result)

            elif (
                result.get("reason")
                and result.get("data_date") is None
            ):

                errors.append({
                    "symbol": symbol,
                    "error": result["reason"]
                })

        except Exception as exc:

            errors.append({
                "symbol": symbol,
                "error": str(exc)
            })

        if progress_callback:

            progress_callback(
                index,
                total
            )

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
        "error_count": len(errors),
        "errors": errors
    }
