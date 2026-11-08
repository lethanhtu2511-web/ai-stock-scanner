
import math
import pandas as pd
import numpy as np

# =========================
# CẤU HÌNH GIAO DỊCH
# =========================

INITIAL_CAPITAL = 10_000_000
RISK_PERCENT = 0.02
MIN_SCORE = 85
MIN_REWARD_RISK = 2.0
LOT_SIZE = 100

BUY_FEE = 0.0015
SELL_FEE = 0.0015
SELL_TAX = 0.001
SLIPPAGE = 0.002


# =========================
# TÍNH CHỈ BÁO KỸ THUẬT
# =========================

def calculate_indicators(df):
    data = df.copy()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]

    if not all(col in data.columns for col in required):
        raise ValueError("Thiếu dữ liệu OHLCV.")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=required)
    data = data.sort_values("Date").reset_index(drop=True)

    if len(data) < 200:
        raise ValueError("Không đủ dữ liệu để phân tích.")

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
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["VOLUME_MA20"] = data["Volume"].rolling(20).mean()

    data["VOLUME_RATIO"] = (
        data["Volume"] /
        data["VOLUME_MA20"].replace(0, np.nan)
    )

    previous_close = close.shift(1)

    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - previous_close).abs()
    tr3 = (data["Low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    data["ATR14"] = true_range.rolling(14).mean()

    data["HIGH20_PREVIOUS"] = (
        data["High"].shift(1).rolling(20).max()
    )

    return data.dropna().reset_index(drop=True)


# =========================
# CHẤM ĐIỂM KỸ THUẬT
# =========================

def calculate_score(row):
    score = 0
    reasons = []

    close = row["Close"]
    ma20 = row["MA20"]
    ma50 = row["MA50"]
    ma200 = row["MA200"]
    rsi = row["RSI"]
    macd = row["MACD"]
    macd_signal = row["MACD_SIGNAL"]
    volume_ratio = row["VOLUME_RATIO"]
    high20 = row["HIGH20_PREVIOUS"]

    if close > ma20:
        score += 10
        reasons.append("Giá trên MA20")

    if ma20 > ma50:
        score += 10
        reasons.append("MA20 trên MA50")

    if ma50 > ma200:
        score += 10
        reasons.append("MA50 trên MA200")

    if 50 <= rsi <= 65:
        score += 15
        reasons.append("RSI trong vùng tích cực")
    elif 65 < rsi <= 70:
        score += 10
        reasons.append("RSI khá tích cực")

    if macd > macd_signal:
        score += 15
        reasons.append("MACD cho tín hiệu tăng")

    if volume_ratio >= 1.5:
        score += 15
        reasons.append("Thanh khoản tăng mạnh")
    elif volume_ratio >= 1.2:
        score += 10
        reasons.append("Thanh khoản cải thiện")

    if close > high20:
        score += 15
        reasons.append("Vượt đỉnh 20 phiên")
    elif close >= ma20 * 0.97:
        score += 10
        reasons.append("Giá gần đường MA20")

    # Điểm gốc tối đa 90, quy đổi về thang 100.
    score = round(score / 90 * 100)

    return min(score, 100), reasons


# =========================
# TÍNH CHI PHÍ GIAO DỊCH
# =========================

def calculate_trade_costs(buy_price, sell_price, quantity):
    buy_value = buy_price * quantity
    sell_value = sell_price * quantity

    buy_cost = buy_value * (BUY_FEE + SLIPPAGE)

    sell_cost = sell_value * (
        SELL_FEE + SELL_TAX + SLIPPAGE
    )

    return buy_cost + sell_cost


# =========================
# TÍNH GIÁ MỤC TIÊU
# =========================

def calculate_target_price(price, stop_loss):
    risk_per_share = price - stop_loss

    if risk_per_share <= 0:
        return None

    # Tìm giá mục tiêu đảm bảo tỷ lệ lợi nhuận/rủi ro
    # sau khi tính phí, thuế và trượt giá ước tính.
    low = price
    high = price + risk_per_share * 10

    for _ in range(100):
        target = (low + high) / 2

        profit = (
            target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        risk = (
            price * (1 + BUY_FEE + SLIPPAGE)
            - stop_loss * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
        )

        if risk <= 0:
            return None

        if profit / risk >= MIN_REWARD_RISK:
            high = target
        else:
            low = target

    return math.ceil(high / 100) * 100


# =========================
# PHÂN TÍCH MỘT CỔ PHIẾU
# =========================

def analyze_stock(
    df,
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT
):
    try:
        if df is None or df.empty:
            return {
                "signal": "WAIT",
                "reason": "Không có dữ liệu.",
                "data_date": None
            }

        data = calculate_indicators(df)
        row = data.iloc[-1]

        price = float(row["Close"])
        atr = float(row["ATR14"])
        rsi = float(row["RSI"])
        volume_ratio = float(row["VOLUME_RATIO"])

        data_date = row["Date"].strftime("%d/%m/%Y")

        if price <= 0 or atr <= 0:
            raise ValueError("Giá hoặc ATR không hợp lệ.")

        # Không sử dụng dữ liệu quá cũ.
        if (pd.Timestamp.now().normalize() - row["Date"].normalize()).days > 10:
            return {
                "signal": "WAIT",
                "reason": "Dữ liệu quá cũ.",
                "data_date": data_date
            }

        score, reasons = calculate_score(row)

        stop_loss = price - 1.5 * atr

        if stop_loss <= 0:
            raise ValueError("Giá cắt lỗ không hợp lệ.")

        target_price = calculate_target_price(
            price,
            stop_loss
        )

        if target_price is None:
            raise ValueError("Không tính được giá mục tiêu.")

        # Rủi ro dự kiến trên mỗi cổ phiếu.
        risk_per_share = (
            price * (1 + BUY_FEE + SLIPPAGE)
            - stop_loss * (
                1 - SELL_FEE - SELL_TAX - SLIPPAGE
            )
        )

        reward_per_share = (
            target_price * (
                1 - SELL_FEE - SELL_TAX - SLIPPAGE
            )
            - price * (1 + BUY_FEE + SLIPPAGE)
        )

        if risk_per_share <= 0:
            raise ValueError("Rủi ro giao dịch không hợp lệ.")

        reward_risk = reward_per_share / risk_per_share

        risk_budget = capital * risk_percent

        quantity_by_risk = int(
            risk_budget / risk_per_share
        )

        quantity_by_capital = int(
            capital / (
                price * (1 + BUY_FEE + SLIPPAGE)
            )
        )

        quantity = min(
            quantity_by_risk,
            quantity_by_capital
        )

        quantity = (
            quantity // LOT_SIZE
        ) * LOT_SIZE

        capital_used = (
            quantity * price * (1 + BUY_FEE + SLIPPAGE)
        )

        risk_money = quantity * risk_per_share

        # Điều kiện mua.
        conditions = [
            score >= MIN_SCORE,
            rsi <= 75,
            quantity >= LOT_SIZE,
            price > row["MA20"],
            reward_risk >= MIN_REWARD_RISK,
            risk_money <= risk_budget
        ]

        signal = "BUY" if all(conditions) else "WAIT"

        if signal == "BUY":
            message = "Đạt các điều kiện mua theo bộ lọc."
        else:
            message = "Chưa đạt đầy đủ điều kiện mua."

        return {
            "signal": signal,
            "score": score,
            "price": round(price),
            "stop_loss": round(stop_loss),
            "target_price": round(target_price),
            "quantity": quantity,
            "capital_used": round(capital_used),
            "risk_money": round(risk_money),
            "reward_risk": round(reward_risk, 2),
            "rsi": round(rsi, 2),
            "volume_ratio": round(volume_ratio, 2),
            "reasons": reasons,
            "data_date": data_date,
            "message": message,
            "reason": message
        }

    except Exception as error:
        return {
            "signal": "WAIT",
            "reason": str(error),
            "data_date": None
        }


# =========================
# QUÉT DANH SÁCH CỔ PHIẾU
# =========================

def scan_market(
    capital=INITIAL_CAPITAL,
    risk_percent=RISK_PERCENT,
    max_symbols=30,
    progress_callback=None
):
    """
    Quét cổ phiếu HOSE và HNX.

    Mặc định tối đa 30 mã/lượt để hạn chế
    số lần gọi API.

    progress_callback (nếu có):
        callback(số_mã_đã_quét, tổng_số_mã)
    """

    from data_provider import (
        fetch_exchange_symbols,
        fetch_stock_history
    )

    symbols = fetch_exchange_symbols()

    if not symbols:
        raise RuntimeError("Không tìm thấy danh sách cổ phiếu.")

    if max_symbols is not None:
        max_symbols = max(1, int(max_symbols))
        symbols = symbols[:max_symbols]

    results = []
    errors = []

    total = len(symbols)

    for index, symbol in enumerate(symbols, start=1):
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

        except Exception as error:
            errors.append({
                "symbol": symbol,
                "error": str(error)
            })

        if progress_callback is not None:
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
        "error_count": len(errors),
        "errors": errors
    }
