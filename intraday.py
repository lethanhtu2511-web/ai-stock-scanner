
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from vnstock import Market


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Cache dữ liệu nến 5 phút
INTRADAY_CACHE_SECONDS = 45
INTRADAY_CACHE = {}

MIN_5M_CANDLES = 60
MAX_CANDLES = 150


def _normalize_symbol(symbol):
    symbol = str(symbol or "").strip().upper()

    if not symbol.isalnum() or not 3 <= len(symbol) <= 10:
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    return symbol


def _parse_datetime(value):
    ts = pd.Timestamp(value)

    if pd.isna(ts):
        return pd.NaT

    if ts.tzinfo is None:
        return ts.tz_localize(VN_TZ)

    return ts.tz_convert(VN_TZ)


def _standardize(raw):
    if raw is None:
        raise ValueError("Nguồn dữ liệu không trả về dữ liệu.")

    df = pd.DataFrame(raw).copy()

    if df.empty:
        raise ValueError("Không có dữ liệu nến 5 phút.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if str(x))
            for col in df.columns
        ]

    df.columns = [str(c).strip().lower() for c in df.columns]

    rename = {
        "time": "date",
        "timestamp": "date",
        "trading_date": "date",
        "tradingdate": "date",
        "vol": "volume",
        "total_volume": "volume",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "closing_price": "close",
    }

    df = df.rename(columns=rename)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Dữ liệu 5 phút thiếu cột: {missing}")

    df = df[required].copy()

    df["date"] = df["date"].apply(_parse_datetime)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    valid = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    return df.loc[valid].reset_index(drop=True)


def _remove_incomplete_candle(df):
    """
    Chỉ giữ các nến 5 phút đã hoàn tất.
    Timestamp được hiểu là thời điểm bắt đầu nến.
    """
    if df.empty:
        return df

    now = pd.Timestamp.now(tz=VN_TZ)
    completed_before = now.floor("5min")

    result = df.loc[
        df["date"] + pd.Timedelta(minutes=5) <= completed_before
    ].copy()

    return result.reset_index(drop=True)


def _normalize_price_unit(df, symbol):
    """
    Đối chiếu giá 5 phút với giá D1 gần nhất để xác định
    dữ liệu đang ở đơn vị đồng hay nghìn đồng.
    """
    from data_provider import fetch_stock_history

    daily = fetch_stock_history(symbol, length=220)

    if daily is None or daily.empty:
        raise ValueError("Không có giá D1 để đối chiếu đơn vị giá.")

    daily_close = float(daily.iloc[-1]["Close"])
    raw_close = float(df.iloc[-1]["close"])

    if daily_close <= 0 or raw_close <= 0:
        raise ValueError("Giá đối chiếu không hợp lệ.")

    ratio = raw_close / daily_close

    if 0.0005 <= ratio <= 0.005:
        multiplier = 1000.0
    elif 0.80 <= ratio <= 1.20:
        multiplier = 1.0
    else:
        raise ValueError(
            "Không xác định được đơn vị giá nến 5 phút "
            f"(tỷ lệ so với D1: {ratio:.5f})."
        )

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * multiplier

    last_close = float(df.iloc[-1]["close"])
    difference = abs(last_close / daily_close - 1)

    if difference > 0.15:
        raise ValueError(
            f"Giá 5 phút lệch {difference * 100:.1f}% so với D1. "
            "Dữ liệu có thể sai đơn vị hoặc không đồng bộ."
        )

    df.attrs["price_multiplier"] = multiplier

    return df


def fetch_intraday_5m(symbol, force_refresh=False):
    symbol = _normalize_symbol(symbol)
    now = time.monotonic()

    cached = INTRADAY_CACHE.get(symbol)

    if (
        not force_refresh
        and cached
        and now - cached["time"] < INTRADAY_CACHE_SECONDS
    ):
        result = cached["data"].copy()
        result.attrs.update(cached["data"].attrs)
        return result

    errors = []

    try:
        # Vnstock 6: chọn nguồn VCI ngay khi khởi tạo Market.
        raw = (
            Market("vci")
            .equity(symbol)
            .ohlcv(
                length=MAX_CANDLES,
                interval="5m",
            )
        )

        df = _standardize(raw)
        df = _remove_incomplete_candle(df)

        if df.empty:
            raise ValueError("Không có nến 5 phút đã hoàn tất.")

        df = _normalize_price_unit(df, symbol)

        if len(df) < MIN_5M_CANDLES:
            raise ValueError(
                f"Chỉ có {len(df)} nến hoàn tất; "
                f"cần tối thiểu {MIN_5M_CANDLES} nến."
            )

        now_vn = pd.Timestamp.now(tz=VN_TZ)
        last_time = pd.Timestamp(df.iloc[-1]["date"])

        age_minutes = (
            now_vn - last_time
        ).total_seconds() / 60

        if age_minutes > 60:
            current_time = now_vn.time()

            in_session = (
                (
                    current_time
                    >= datetime.strptime("09:00", "%H:%M").time()
                    and current_time
                    <= datetime.strptime("11:30", "%H:%M").time()
                )
                or
                (
                    current_time
                    >= datetime.strptime("13:00", "%H:%M").time()
                    and current_time
                    <= datetime.strptime("14:45", "%H:%M").time()
                )
            )

            if in_session:
                raise ValueError(
                    f"Nến gần nhất đã cũ {age_minutes:.0f} phút. "
                    "Không phát tín hiệu từ dữ liệu trễ."
                )

        df = df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        ).reset_index(drop=True)

        df.attrs["source"] = "VCI"
        df.attrs["last_date"] = last_time.strftime("%d/%m/%Y %H:%M")
        df.attrs["price_multiplier"] = df.attrs.get(
            "price_multiplier", 1.0
        )

        INTRADAY_CACHE[symbol] = {
            "time": time.monotonic(),
            "data": df.copy(),
        }

        return df

    except Exception as exc:
        errors.append(str(exc))

    raise RuntimeError(
        f"{symbol}: không lấy được dữ liệu 5 phút từ VCI. "
        + " | ".join(errors)
    )


def analyze_intraday_5m(df, symbol=None):
    if df is None or len(df) < MIN_5M_CANDLES:
        raise ValueError("Không đủ dữ liệu để phân tích 5 phút.")

    data = df.copy()

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Thiếu cột dữ liệu: {missing}")

    for col in required[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = (
        data.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if len(data) < MIN_5M_CANDLES:
        raise ValueError("Không đủ nến hợp lệ sau khi chuẩn hóa.")

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # EMA 9 và EMA 20
    data["EMA9"] = close.ewm(span=9, adjust=False).mean()
    data["EMA20"] = close.ewm(span=20, adjust=False).mean()

    # RSI 14
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

    data["RSI14"] = (100 - 100 / (1 + rs)).fillna(50)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]

    # VWAP tính riêng từng ngày
    typical_price = (high + low + close) / 3

    data["DATE_ONLY"] = pd.to_datetime(data["Date"]).dt.date
    data["TP_VOLUME"] = typical_price * volume

    data["VWAP"] = (
        data.groupby("DATE_ONLY")["TP_VOLUME"].cumsum()
        / data.groupby("DATE_ONLY")["Volume"].cumsum().replace(0, np.nan)
    )

    # ATR và khối lượng
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["ATR14"] = true_range.rolling(14).mean()

    data["VOL_MA20"] = volume.rolling(20).mean()

    data["VOL_RATIO"] = (
        volume / data["VOL_MA20"].replace(0, np.nan)
    )

    data["HIGH20_PREVIOUS"] = high.shift(1).rolling(20).max()
    data["LOW20_PREVIOUS"] = low.shift(1).rolling(20).min()

    data = data.replace([np.inf, -np.inf], np.nan)

    last = data.iloc[-1]
    previous = data.iloc[-2]

    price = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema20 = float(last["EMA20"])
    rsi = float(last["RSI14"])

    vwap = (
        float(last["VWAP"])
        if pd.notna(last["VWAP"])
        else np.nan
    )

    macd = float(last["MACD"])
    macd_signal = float(last["MACD_SIGNAL"])

    volume_ratio = (
        float(last["VOL_RATIO"])
        if pd.notna(last["VOL_RATIO"])
        else 0.0
    )

    bullish_trend = (
        ema9 > ema20
        and price > ema9
        and ema9 >= float(previous["EMA9"])
    )

    vwap_ok = np.isfinite(vwap) and price >= vwap
    rsi_ok = 50 <= rsi <= 68
    macd_ok = macd > macd_signal
    volume_ok = volume_ratio >= 1.2

    score = 0
    reasons = []
    blockers = []

    if ema9 > ema20:
        score += 20
        reasons.append("EMA9 nằm trên EMA20.")

    if price > ema9:
        score += 15
        reasons.append("Giá đóng cửa trên EMA9.")

    if ema9 >= float(previous["EMA9"]):
        score += 10
        reasons.append("EMA9 đang đi lên.")

    if vwap_ok:
        score += 15
        reasons.append("Giá đang trên VWAP.")

    if rsi_ok:
        score += 10
        reasons.append("RSI nằm trong vùng tích cực 50–68.")
    elif rsi > 68:
        blockers.append("RSI cao, có nguy cơ mua đuổi.")

    if macd_ok:
        score += 15
        reasons.append("MACD nằm trên đường tín hiệu.")

    if volume_ok:
        score += 15
        reasons.append("Khối lượng cao hơn trung bình 20 nến.")

    if price <= ema20:
        blockers.append("Giá chưa vượt EMA20.")

    if not vwap_ok:
        blockers.append("Giá chưa vượt VWAP.")

    if not macd_ok:
        blockers.append("MACD chưa xác nhận động lượng tăng.")

    if not volume_ok:
        blockers.append("Khối lượng chưa xác nhận.")

    # Tránh mua đuổi khi nến biến động quá lớn
    atr = (
        float(last["ATR14"])
        if pd.notna(last["ATR14"])
        else 0.0
    )

    candle_range = float(last["High"] - last["Low"])

    if atr > 0 and candle_range > atr * 2.5:
        blockers.append(
            "Nến biến động quá lớn so với ATR; tránh mua đuổi."
        )

    if rsi >= 75:
        blockers.append("RSI quá mua.")

    if (
        bullish_trend
        and vwap_ok
        and rsi_ok
        and macd_ok
        and volume_ok
    ):
        signal = "TÍN HIỆU TĂNG"

        explanation = (
            "Các điều kiện xu hướng, VWAP, RSI, MACD và khối lượng "
            "đang đồng thuận trên khung 5 phút. Đây vẫn là tín hiệu "
            "ngắn hạn trong phiên, không thay thế xác nhận D1."
        )

    elif score >= 50:
        signal = "THEO DÕI"

        explanation = (
            "Có một số yếu tố tích cực nhưng chưa đủ đồng thuận "
            "để xem là tín hiệu tăng."
        )

    else:
        signal = "CHỜ"

        explanation = (
            "Chưa có lợi thế kỹ thuật rõ ràng trên khung 5 phút."
        )

    if blockers and signal == "TÍN HIỆU TĂNG":
        if any(
            "quá mua" in x
            or "quá lớn" in x
            or "mua đuổi" in x
            for x in blockers
        ):
            signal = "THEO DÕI"

            explanation = (
                "Có tín hiệu tăng nhưng xuất hiện rủi ro mua đuổi. "
                "Chờ giá điều chỉnh hoặc xác nhận lại."
            )

    support = (
        float(last["LOW20_PREVIOUS"])
        if pd.notna(last["LOW20_PREVIOUS"])
        else np.nan
    )

    resistance = (
        float(last["HIGH20_PREVIOUS"])
        if pd.notna(last["HIGH20_PREVIOUS"])
        else np.nan
    )

    stop = price - 1.5 * atr if atr > 0 else np.nan

    return {
        "symbol": symbol,
        "signal": signal,
        "explanation": explanation,
        "score": int(score),
        "price": price,
        "timestamp": pd.Timestamp(last["Date"]),
        "ema9": ema9,
        "ema20": ema20,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "vwap": vwap,
        "volume_ratio": volume_ratio,
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "stop_reference": stop,
        "reasons": reasons,
        "blockers": blockers,
        "candles": data,
    }
