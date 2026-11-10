def _hourly_trend(hourly_df):
    """
    Tính EMA9, EMA20 từ dữ liệu H1.
    Chấp nhận nhiều cách đặt tên cột giá đóng cửa.
    Cần tối thiểu 20 nến hợp lệ để tính EMA20.
    """
    if hourly_df is None or not isinstance(hourly_df, pd.DataFrame):
        return None, None, False, False

    if hourly_df.empty:
        return None, None, False, False

    frame = hourly_df.copy()

    # Chuẩn hóa tên cột
    frame.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in frame.columns
    ]

    close_aliases = [
        "close",
        "closing_price",
        "price",
        "close_price",
        "match_price",
        "last_price",
        "gia_dong_cua",
    ]

    close_col = next(
        (col for col in close_aliases if col in frame.columns),
        None
    )

    if close_col is None:
        return None, None, False, False

    close = pd.to_numeric(
        frame[close_col],
        errors="coerce"
    )

    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close = close[close > 0]

    if len(close) < 20:
        return None, None, False, False

    ema9_series = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema20_series = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema9 = float(ema9_series.iloc[-1])
    ema20 = float(ema20_series.iloc[-1])

    if (
        not np.isfinite(ema9)
        or not np.isfinite(ema20)
        or ema20 <= 0
    ):
        return None, None, False, False

    h1_bullish = ema9 > ema20
    h1_early = ema9 >= ema20 * 0.995

    return ema9, ema20, h1_bullish, h1_early
