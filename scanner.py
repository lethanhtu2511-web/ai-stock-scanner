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

# Cổ phiếu phải còn ít nhất 3% dư địa tới kháng cự,
# trừ trường hợp đã breakout được xác nhận.
MIN_ROOM_TO_RESISTANCE_PCT = 3.0
BREAKOUT_BUFFER_PCT = 0.3
MIN_BREAKOUT_VOLUME_RATIO = 1.5

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
    data["RSI"] = (100 - 100 / (1 + rs)).fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
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
    close_position = (
        float(row.Close) - float(row.Low)
    ) / candle_range

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


def _net_reward_risk(entry, stop, target):
    cost_buy = entry * (1 + BUY_FEE + SLIPPAGE)
    net_sell_stop = stop * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)
    net_sell_target = target * (1 - SELL_FEE - SELL_TAX - SLIPPAGE)

    risk = cost_buy - net_sell_stop
    reward = net_sell_target - cost_buy

    if risk <= 0:
        return 0.0

    return reward / risk


def _minimum_target_for_rr(price, stop):
    """
    Tính mục tiêu tối thiểu để đạt RR sau chi phí 2:1.
    Làm tròn theo bước giá 100 đồng.
    """

    target = price

    for _ in range(10000):
        rr = _net_reward_risk(price, stop, target)

        if rr >= MIN_REWARD_RISK:
            return math.ceil(target / 100) * 100

        target += 100

    return None


def _find_nearest_resistance(data, price):
    """
    Tìm đỉnh cục bộ trong 60 phiên trước phiên hiện tại.
    Không dùng đỉnh của nến hiện tại để tránh tự xác nhận breakout.
    """

    history = data.iloc[:-1].tail(60).reset_index(drop=True)

    if len(history) < 20:
        return None

    highs = history["High"].astype(float).to_numpy()
    pivot_levels = []

    for i in range(2, len(highs) - 2):
        window = highs[i - 2:i + 3]

        if highs[i] >= max(window):
            level = float(highs[i])

            if level > price * 1.005:
                pivot_levels.append(level)

    if not pivot_levels:
        recent_high = float(history["High"].tail(20).max())

        if recent_high > price * 1.005:
            return recent_high

        return None

    # Gộp các đỉnh gần nhau thành một vùng kháng cự.
    pivot_levels.sort()
    clusters = []

    for level in pivot_levels:
        if not clusters:
            clusters.append([level])
            continue

        center = float(np.mean(clusters[-1]))

        if abs(level / center - 1) <= 0.01:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    levels = [float(np.mean(group)) for group in clusters]
    above_price = [level for level in levels if level > price * 1.005]

    return min(above_price) if above_price else None


def _hourly_confirmation(hourly_df):
    if hourly_df is None or len(hourly_df) < 20:
        return {
            "ok": False,
            "status": "Không đủ dữ liệu H1",
            "reason": "Cần tối thiểu 20 nến H1 hoàn tất."
        }

    h = hourly_df.copy()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]

    if any(c not in h.columns for c in required):
        return {
            "ok": False,
            "status": "Dữ liệu H1 không hợp lệ",
            "reason": "Thiếu cột OHLCV."
        }

    h["Date"] = pd.to_datetime(h["Date"], errors="coerce")

    for col in required[1:]:
        h[col] = pd.to_numeric(h[col], errors="coerce")

    h = (
        h.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if len(h) < 20:
        return {
            "ok": False,
            "status": "Không đủ dữ liệu H1",
            "reason": "Cần tối thiểu 20 nến H1 hợp lệ."
        }

    h["EMA9"] = h["Close"].ewm(span=9, adjust=False).mean()
    h["EMA20"] = h["Close"].ewm(span=20, adjust=False).mean()

    last = h.iloc[-1]

    ok = bool(
        last["EMA9"] > last["EMA20"]
        and last["Close"] >= last["EMA9"]
    )

    return {
        "ok": ok,
        "status": "H1 xác nhận tăng" if ok else "H1 chưa xác nhận tăng",
        "reason": (
            "EMA9 trên EMA20 và giá H1 trên EMA9"
            if ok
            else "EMA9 chưa vượt EMA20 hoặc giá H1 dưới EMA9"
        ),
        "last_hour": pd.Timestamp(last["Date"]).strftime("%d/%m %H:%M"),
        "ema9": round(float(last["EMA9"])),
        "ema20": round(float(last["EMA20"]))
    }


def _result_filtered(reason, row=None):
    result = {
        "signal": "FILTERED",
        "reason": reason
    }

    if row is not None:
        try:
            result["data_date"] = pd.Timestamp(
                row.Date
            ).strftime("%d/%m/%Y")
        except Exception:
            pass

    return result


def analyze_stock(df, symbol=None, hourly_df=None):
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

        today = pd.Timestamp.now(
            tz="Asia/Ho_Chi_Minh"
        ).date()

        row_date = pd.Timestamp(row.Date).date()
        age_days = (today - row_date).days

        if age_days > 7:
            return {
                "signal": "WAIT",
                "reason": (
                    f"Dữ liệu nến ngày đã cũ {age_days} ngày; "
                    "không phát tín hiệu mua."
                ),
                "data_date": data_date
            }

        atr = float(row.ATR14)

        if not np.isfinite(atr) or atr <= 0:
            raise ValueError("ATR không hợp lệ.")

        ma20 = float(row.MA20)
        extension_pct = (
            (price / ma20 - 1) * 100
            if ma20 > 0 else 999
        )

        score, reasons = calculate_score(row)

        resistance = _find_nearest_resistance(data, price)

        resistance_gap_pct = (
            (resistance / price - 1) * 100
            if resistance is not None else None
        )

        breakout = bool(
            resistance is not None
            and price > resistance * (1 + BREAKOUT_BUFFER_PCT / 100)
            and float(row.VOLUME_RATIO) >= MIN_BREAKOUT_VOLUME_RATIO
            and float(row.Close) > float(row.Open)
        )

        stop = price - 1.5 * atr

        if stop <= 0 or stop >= price:
            raise ValueError("Giá cắt lỗ không hợp lệ.")

        min_target = _minimum_target_for_rr(price, stop)

        target = min_target

        # Nếu còn kháng cự phía trên và chưa breakout,
        # lấy kháng cự làm mục tiêu thực tế, không đặt mục tiêu xuyên qua.
        if (
            resistance is not None
            and not breakout
            and resistance > price
        ):
            resistance_target = math.floor(resistance / 100) * 100

            if resistance_target > price:
                target = resistance_target

        rr = (
            _net_reward_risk(price, stop, target)
            if target is not None and target > price
            else 0.0
        )

        h1 = _hourly_confirmation(hourly_df)

        blockers = []

        if score < MIN_SCORE:
            blockers.append(
                f"Điểm kỹ thuật {score} dưới ngưỡng {MIN_SCORE}"
            )

        if not (45 <= float(row.RSI) <= 68):
            blockers.append(
                f"RSI {float(row.RSI):.1f} ngoài vùng mua ưu tiên 45–68"
            )

        if price <= ma20:
            blockers.append("Giá chưa nằm trên MA20")

        if extension_pct > 8:
            blockers.append(
                f"Giá cách MA20 {extension_pct:.1f}%, có nguy cơ mua đuổi"
            )

        if (
            resistance is not None
            and not breakout
            and resistance_gap_pct < MIN_ROOM_TO_RESISTANCE_PCT
        ):
            blockers.append(
                f"Chỉ còn {resistance_gap_pct:.1f}% "
                f"tới kháng cự gần {resistance:,.0f}"
            )

        if target is None or rr < MIN_REWARD_RISK:
            blockers.append(
                "Kháng cự gần làm RR sau chi phí thấp hơn 2:1"
            )

        if not h1["ok"]:
            blockers.append(
                "H1 chưa xác nhận: " + h1["reason"]
            )

        if breakout:
            reasons.append(
                "Breakout D1: đóng cửa vượt kháng cự "
                "với khối lượng xác nhận"
            )

        if resistance is not None:
            reasons.append(
                f"Kháng cự gần: {resistance:,.0f}"
            )

        if h1["ok"]:
            reasons.append(
                "H1 xác nhận xu hướng tăng bằng EMA9/EMA20"
            )

        signal = "BUY" if not blockers else "WAIT"

        reason = (
            "Đạt đầy đủ điều kiện D1, H1, kháng cự "
            "và RR sau chi phí."
            if signal == "BUY"
            else "; ".join(blockers)
        )

        return {
            "signal": signal,
            "score": score,
            "price": round(price),
            "daily_close": round(price),
            "stop_loss": round(stop),
            "target_price": round(target) if target else None,
            "reward_risk": round(rr, 2),
            "rsi": round(float(row.RSI), 2),
            "volume_ratio": round(float(row.VOLUME_RATIO), 2),
            "avg_turnover_20d": round(avg_turnover),
            "median_turnover_20d": round(median_turnover),
            "resistance": round(resistance) if resistance else None,
            "resistance_gap_pct": (
                round(resistance_gap_pct, 2)
                if resistance_gap_pct is not None else None
            ),
            "breakout_confirmed": breakout,
            "h1_status": h1["status"],
            "h1_last_bar": h1.get("last_hour"),
            "h1_ema9": h1.get("ema9"),
            "h1_ema20": h1.get("ema20"),
            "reasons": reasons,
            "blockers": blockers,
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
        fetch_stock_history,
        fetch_stock_hourly
    )

    symbols = fetch_exchange_symbols()

    if max_symbols is not None:
        symbols = symbols[:max(1, int(max_symbols))]

    results = []
    errors = []
    filtered_count = 0
    insufficient_history_count = 0
    hourly_unavailable_count = 0

    total = len(symbols)

    for index, symbol in enumerate(symbols, start=1):
        try:
            if symbol.upper() in MANUALLY_BLOCKED_SYMBOLS:
                filtered_count += 1
            else:
                history = fetch_stock_history(symbol, length=260)

                hourly = None

                try:
                    hourly = fetch_stock_hourly(symbol)
                except Exception as exc:
                    hourly_unavailable_count += 1
                    hourly_error = str(exc)

                result = analyze_stock(
                    history,
                    symbol=symbol,
                    hourly_df=hourly
                )

                result["symbol"] = symbol

                if hourly is None:
                    result["h1_status"] = "Không lấy được dữ liệu H1"
                    result["h1_error"] = hourly_error

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

            if (
                "200 phiên" in error_message
                or "INSUFFICIENT_HISTORY" in error_message
            ):
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
        "hourly_unavailable_count": hourly_unavailable_count,
        "error_count": len(errors),
        "errors": errors
    }
