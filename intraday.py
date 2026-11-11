
import json
import time
from datetime import datetime, date
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Cập nhật dữ liệu mỗi 45 giây
INTRADAY_CACHE_SECONDS = 45

# Một phiên có tối đa khoảng 51 nến 5 phút.
# Cần tối thiểu 20 nến để tính các chỉ báo.
MIN_5M_CANDLES = 20

PAGE_SIZE = 5000
MAX_PAGES = 60

KBS_URL = (
    "https://kbbuddywts.kbsec.com.vn/"
    "iis-server/investment/trade/history/"
)

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
}

HTTP_TIMEOUT = 15.0

# Lưu giao dịch trong bộ nhớ để không phải tải lại
# toàn bộ lịch sử giao dịch sau mỗi 45 giây.
TRADE_CACHE = {}


# =====================================================
# CHUẨN HÓA MÃ CỔ PHIẾU
# =====================================================

def _normalize_symbol(symbol):
    symbol = str(symbol or "").strip().upper()

    if not symbol.isalnum() or not 3 <= len(symbol) <= 10:
        raise ValueError("Mã cổ phiếu không hợp lệ.")

    return symbol


# =====================================================
# LẤY VÀ ĐỌC DỮ LIỆU KHỚP LỆNH KBS
# =====================================================

def _fetch_page(symbol, page):
    url = KBS_URL + symbol

    try:
        response = httpx.get(
            url,
            params={
                "page": page,
                "limit": PAGE_SIZE,
            },
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT,
        )

        response.raise_for_status()

        payload = response.json()

    except Exception as exc:
        raise RuntimeError(
            f"KBS không phản hồi ở trang {page}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("KBS trả về dữ liệu không đúng định dạng.")

    rows = payload.get("data")

    if not isinstance(rows, list):
        raise RuntimeError("KBS không trả về danh sách giao dịch.")

    return rows


def _parse_trade_rows(rows):
    parsed = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        try:
            session_text = str(row.get("TD", "")).strip()

            session_date = datetime.strptime(
                session_text,
                "%d/%m/%Y",
            ).date()

            trade_time = str(row.get("FT", "")).strip()

            if not trade_time:
                continue

            trade_time = trade_time[:8]

            parsed_time = datetime.strptime(
                trade_time,
                "%H:%M:%S",
            ).time()

            timestamp = datetime.combine(
                session_date,
                parsed_time,
            ).replace(tzinfo=VN_TZ)

            price = float(row.get("FMP"))
            volume = int(row.get("FV"))
            accumulated_volume = int(row.get("AVO"))

            if price <= 0 or volume <= 0:
                continue

            parsed.append({
                "Date": timestamp,
                "Price": price,
                "Volume": volume,
                "Accumulated": accumulated_volume,
            })

        except (TypeError, ValueError, OverflowError):
            continue

    return parsed


# =====================================================
# TẢI TOÀN BỘ GIAO DỊCH KHI KHỞI TẠO
# =====================================================

def _fetch_all_trades(symbol):
    all_rows = []

    for page in range(1, MAX_PAGES + 1):
        rows = _fetch_page(symbol, page)

        parsed = _parse_trade_rows(rows)

        all_rows.extend(parsed)

        if len(rows) < PAGE_SIZE:
            break

        # Tránh gửi yêu cầu liên tục quá nhanh.
        time.sleep(0.15)

    else:
        raise RuntimeError(
            "KBS vượt giới hạn số trang cho phép. "
            "Không sử dụng dữ liệu chưa đầy đủ."
        )

    if not all_rows:
        raise RuntimeError(
            f"KBS chưa có dữ liệu giao dịch cho {symbol}."
        )

    df = pd.DataFrame(all_rows)

    df = (
        df.sort_values("Accumulated")
        .drop_duplicates(
            subset=["Accumulated"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Kiểm tra tính liên tục của khối lượng tích lũy.
    # Nếu dữ liệu thiếu hoặc trùng giao dịch thì từ chối.
    expected = df["Volume"].cumsum()

    actual = df["Accumulated"]

    if not np.array_equal(
        expected.to_numpy(),
        actual.to_numpy(),
    ):
        raise RuntimeError(
            "Dữ liệu giao dịch KBS không liên tục. "
            "Không sử dụng dữ liệu thiếu để tạo tín hiệu."
        )

    return df


# =====================================================
# CẬP NHẬT GIAO DỊCH MỚI
# =====================================================

def _refresh_latest_trades(symbol, cached):
    rows = _fetch_page(symbol, 1)

    new_trades = _parse_trade_rows(rows)

    if not new_trades:
        return cached["trades"].copy()

    new_df = pd.DataFrame(new_trades)

    old_df = cached["trades"].copy()

    combined = pd.concat(
        [old_df, new_df],
        ignore_index=True,
    )

    combined = (
        combined.sort_values("Accumulated")
        .drop_duplicates(
            subset=["Accumulated"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Nếu dữ liệu mới có ngày giao dịch khác,
    # không trộn hai phiên với nhau.
    latest_session = combined["Date"].dt.date.max()

    combined = combined.loc[
        combined["Date"].dt.date == latest_session
    ].copy()

    return combined.reset_index(drop=True)


# =====================================================
# CHUYỂN GIAO DỊCH THÀNH NẾN 5 PHÚT
# =====================================================

def _trades_to_candles(trades):
    if trades is None or trades.empty:
        raise ValueError("Không có giao dịch để tạo nến.")

    data = trades.copy()

    data = data.sort_values("Date").reset_index(drop=True)

    # Giá giao dịch KBS đã ở đơn vị đồng.
    # ATC lúc 14:45 được đưa vào nến 14:40-14:45.
    atc_mask = (
        (data["Date"].dt.hour == 14)
        & (data["Date"].dt.minute == 45)
    )

    data.loc[atc_mask, "Date"] = (
        data.loc[atc_mask, "Date"]
        - pd.Timedelta(minutes=5)
    )

    data["CandleTime"] = data["Date"].dt.floor("5min")

    candles = (
        data.groupby("CandleTime")
        .agg(
            Open=("Price", "first"),
            High=("Price", "max"),
            Low=("Price", "min"),
            Close=("Price", "last"),
            Volume=("Volume", "sum"),
        )
        .reset_index()
        .rename(columns={"CandleTime": "Date"})
    )

    # Chỉ sử dụng nến đã hoàn tất.
    now = pd.Timestamp.now(tz=VN_TZ)
    completed_before = now.floor("5min")

    candles = candles.loc[
        candles["Date"] + pd.Timedelta(minutes=5)
        <= completed_before
    ].copy()

    candles = (
        candles.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if candles.empty:
        raise ValueError(
            "Chưa có nến 5 phút hoàn tất."
        )

    candles.attrs["source"] = "KBS"
    candles.attrs["last_date"] = (
        candles.iloc[-1]["Date"].strftime(
            "%d/%m/%Y %H:%M"
        )
    )

    return candles


# =====================================================
# HÀM LẤY DỮ LIỆU 5 PHÚT
# =====================================================

def fetch_intraday_5m(symbol, force_refresh=False):
    symbol = _normalize_symbol(symbol)

    now = time.monotonic()

    cached = TRADE_CACHE.get(symbol)

    # Dùng cache nếu chưa hết 45 giây.
    if (
        not force_refresh
        and cached
        and now - cached["time"] < INTRADAY_CACHE_SECONDS
    ):
        candles = _trades_to_candles(
            cached["trades"]
        )

        if len(candles) >= MIN_5M_CANDLES:
            return candles

    # Lần đầu tải dữ liệu hoặc đổi phiên:
    # tải toàn bộ giao dịch để dựng lịch sử nến.
    if cached is None:
        trades = _fetch_all_trades(symbol)

    else:
        latest_date = (
            cached["trades"]["Date"].dt.date.max()
        )

        current_latest = _refresh_latest_trades(
            symbol,
            cached,
        )

        new_date = current_latest["Date"].dt.date.max()

        if new_date != latest_date:
            # Phiên mới: tải lại đầy đủ giao dịch phiên mới.
            trades = _fetch_all_trades(symbol)
        else:
            trades = current_latest

    if trades.empty:
        raise RuntimeError(
            f"Không có giao dịch hợp lệ của {symbol}."
        )

    TRADE_CACHE[symbol] = {
        "time": time.monotonic(),
        "trades": trades.copy(),
    }

    candles = _trades_to_candles(trades)

    if len(candles) < MIN_5M_CANDLES:
        raise RuntimeError(
            f"{symbol} hiện có {len(candles)} nến 5 phút "
            f"đã hoàn tất. Cần tối thiểu "
            f"{MIN_5M_CANDLES} nến để phân tích. "
            "Hãy chờ thêm dữ liệu trong phiên."
        )

    # Kiểm tra độ trễ dữ liệu trong phiên.
    now_vn = pd.Timestamp.now(tz=VN_TZ)
    last_time = pd.Timestamp(candles.iloc[-1]["Date"])

    age_minutes = (
        now_vn - last_time
    ).total_seconds() / 60

    current_time = now_vn.time()

    in_session = (
        (
            current_time >= datetime.strptime(
                "09:00", "%H:%M"
            ).time()
            and current_time <= datetime.strptime(
                "11:30", "%H:%M"
            ).time()
        )
        or
        (
            current_time >= datetime.strptime(
                "13:00", "%H:%M"
            ).time()
            and current_time <= datetime.strptime(
                "14:45", "%H:%M"
            ).time()
        )
    )

    if in_session and age_minutes > 60:
        raise RuntimeError(
            f"Dữ liệu nến đã cũ {age_minutes:.0f} phút. "
            "Không phát tín hiệu từ dữ liệu trễ."
        )

    candles.attrs["source"] = "KBS"
    candles.attrs["last_date"] = last_time.strftime(
        "%d/%m/%Y %H:%M"
    )

    return candles


# =====================================================
# PHÂN TÍCH KỸ THUẬT 5 PHÚT
# =====================================================

def analyze_intraday_5m(df, symbol=None):
    if df is None or len(df) < MIN_5M_CANDLES:
        raise ValueError(
            "Không đủ dữ liệu để phân tích 5 phút."
        )

    data = df.copy()

    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu cột dữ liệu: {missing}"
        )

    for col in required[1:]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    data = (
        data.dropna(subset=required)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if len(data) < MIN_5M_CANDLES:
        raise ValueError(
            "Không đủ nến hợp lệ sau khi chuẩn hóa."
        )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # EMA 9 và EMA 20
    data["EMA9"] = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    data["EMA20"] = close.ewm(
        span=20,
        adjust=False,
    ).mean()

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

    data["RSI14"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    data["MACD_HIST"] = (
        data["MACD"] - data["MACD_SIGNAL"]
    )

    # VWAP tính riêng từng ngày
    typical_price = (
        high + low + close
    ) / 3

    data["DATE_ONLY"] = (
        pd.to_datetime(data["Date"]).dt.date
    )

    data["TP_VOLUME"] = (
        typical_price * volume
    )

    data["VWAP"] = (
        data.groupby("DATE_ONLY")["TP_VOLUME"].cumsum()
        /
        data.groupby("DATE_ONLY")["Volume"]
        .cumsum()
        .replace(0, np.nan)
    )

    # ATR
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

    # Khối lượng
    data["VOL_MA20"] = volume.rolling(20).mean()

    data["VOL_RATIO"] = (
        volume
        / data["VOL_MA20"].replace(0, np.nan)
    )

    # Hỗ trợ và kháng cự
    data["HIGH20_PREVIOUS"] = (
        high.shift(1).rolling(20).max()
    )

    data["LOW20_PREVIOUS"] = (
        low.shift(1).rolling(20).min()
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

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

    vwap_ok = (
        np.isfinite(vwap)
        and price >= vwap
    )

    rsi_ok = 50 <= rsi <= 68
    macd_ok = macd > macd_signal
    volume_ok = volume_ratio >= 1.2

    score = 0
    reasons = []
    blockers = []

    if ema9 > ema20:
        score += 20
        reasons.append(
            "EMA9 nằm trên EMA20."
        )

    if price > ema9:
        score += 15
        reasons.append(
            "Giá đóng cửa trên EMA9."
        )

    if ema9 >= float(previous["EMA9"]):
        score += 10
        reasons.append(
            "EMA9 đang đi lên."
        )

    if vwap_ok:
        score += 15
        reasons.append(
            "Giá đang trên VWAP."
        )

    if rsi_ok:
        score += 10
        reasons.append(
            "RSI nằm trong vùng tích cực 50–68."
        )
    elif rsi > 68:
        blockers.append(
            "RSI cao, có nguy cơ mua đuổi."
        )

    if macd_ok:
        score += 15
        reasons.append(
            "MACD nằm trên đường tín hiệu."
        )

    if volume_ok:
        score += 15
        reasons.append(
            "Khối lượng cao hơn trung bình 20 nến."
        )

    if price <= ema20:
        blockers.append(
            "Giá chưa vượt EMA20."
        )

    if not vwap_ok:
        blockers.append(
            "Giá chưa vượt VWAP."
        )

    if not macd_ok:
        blockers.append(
            "MACD chưa xác nhận động lượng tăng."
        )

    if not volume_ok:
        blockers.append(
            "Khối lượng chưa xác nhận."
        )

    # Tránh mua đuổi
    atr = (
        float(last["ATR14"])
        if pd.notna(last["ATR14"])
        else 0.0
    )

    candle_range = float(
        last["High"] - last["Low"]
    )

    if atr > 0 and candle_range > atr * 2.5:
        blockers.append(
            "Nến biến động quá lớn so với ATR; tránh mua đuổi."
        )

    if rsi >= 75:
        blockers.append(
            "RSI quá mua."
        )

    if (
        bullish_trend
        and vwap_ok
        and rsi_ok
        and macd_ok
        and volume_ok
    ):
        signal = "TÍN HIỆU TĂNG"

        explanation = (
            "Các điều kiện xu hướng, VWAP, RSI, MACD "
            "và khối lượng đang đồng thuận trên khung 5 phút. "
            "Đây là tín hiệu ngắn hạn trong phiên, "
            "không thay thế xác nhận D1."
        )

    elif score >= 50:
        signal = "THEO DÕI"

        explanation = (
            "Có một số yếu tố tích cực nhưng chưa đủ "
            "đồng thuận để xem là tín hiệu tăng."
        )

    else:
        signal = "CHỜ"

        explanation = (
            "Chưa có lợi thế kỹ thuật rõ ràng "
            "trên khung 5 phút."
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
                "Có tín hiệu tăng nhưng xuất hiện rủi ro "
                "mua đuổi. Chờ giá điều chỉnh hoặc xác nhận lại."
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

    stop = (
        price - 1.5 * atr
        if atr > 0
        else np.nan
    )

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
