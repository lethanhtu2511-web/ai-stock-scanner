
import streamlit as st
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock, scan_market
from data_provider import (
    fetch_stock_history,
    fetch_stock_hourly,
    fetch_exchange_symbols,
    get_hourly_source_status,
)
from intraday import fetch_intraday_5m, analyze_intraday_5m


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide",
)

if "market_scan_output" not in st.session_state:
    st.session_state.market_scan_output = None

if "market_scan_meta" not in st.session_state:
    st.session_state.market_scan_meta = None

if "symbol_list" not in st.session_state:
    st.session_state.symbol_list = []

if "symbol_stats" not in st.session_state:
    st.session_state.symbol_stats = {}


# =====================================================
# TIÊU ĐỀ
# =====================================================

st.title("📈 AI STOCK SCANNER")

st.caption(
    "Sàng lọc cổ phiếu HOSE và HNX cho chiến lược ngắn hạn 2–5 phiên."
)

st.info(
    "Bot chỉ phân tích và đưa ra tín hiệu tham khảo, "
    "không tự động đặt lệnh."
)

st.caption(
    "BUY xác nhận: điểm tối thiểu 78, RSI 45–68, "
    "D1 và H1 đồng thuận, RR sau chi phí tối thiểu 2:1."
)

st.caption(
    "Thanh khoản tối thiểu: bình quân 20 phiên 5 tỷ/ngày, "
    "trung vị 20 phiên 3 tỷ/ngày. Lịch sử tối thiểu 200 phiên."
)

st.caption(
    "EARLY BUY là tín hiệu theo dõi sớm, không phải BUY xác nhận."
)


# =====================================================
# HÀM HỖ TRỢ
# =====================================================

def valid_number(value):
    try:
        return value is not None and pd.notna(value)
    except (TypeError, ValueError):
        return False


def fmt_price(value):
    if not valid_number(value):
        return "—"
    return f"{float(value):,.0f} đ"


def fmt_num(value, digits=2):
    if not valid_number(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_turnover(value):
    if not valid_number(value):
        return "—"
    return f"{float(value) / 1e9:.2f} tỷ/ngày"


def render_signal(signal):
    if signal == "BUY":
        st.success("🟢 BUY XÁC NHẬN")
    elif signal == "EARLY_BUY":
        st.warning("🟡 EARLY BUY – TÍN HIỆU SỚM")
    elif signal == "FILTERED":
        st.warning("⚪ MÃ BỊ LỌC")
    else:
        st.info("⚪ WAIT – TIẾP TỤC THEO DÕI")


def make_results_dataframe(results):
    df = pd.DataFrame(results)

    if df.empty:
        return df

    columns = [
        "signal",
        "symbol",
        "score",
        "confirmed_score",
        "early_score",
        "price",
        "stop_loss",
        "target_price",
        "reward_risk",
        "rsi",
        "volume_ratio",
        "h1_ema9",
        "h1_ema20",
        "h1_status",
        "avg_turnover_20d",
        "median_turnover_20d",
        "data_date",
        "reason",
    ]

    df = df[[c for c in columns if c in df.columns]]

    return df.rename(
        columns={
            "signal": "Tín hiệu",
            "symbol": "Mã",
            "score": "Điểm tín hiệu",
            "confirmed_score": "Điểm BUY xác nhận",
            "early_score": "Điểm EARLY BUY",
            "price": "Giá phân tích",
            "stop_loss": "Cắt lỗ tham khảo",
            "target_price": "Mục tiêu tham khảo",
            "reward_risk": "RR sau chi phí",
            "rsi": "RSI",
            "volume_ratio": "KL/TB20",
            "h1_ema9": "EMA9 H1",
            "h1_ema20": "EMA20 H1",
            "h1_status": "Trạng thái H1",
            "avg_turnover_20d": "GTGD TB20",
            "median_turnover_20d": "GTGD trung vị",
            "data_date": "Ngày dữ liệu",
            "reason": "Nhận định",
        }
    )


def render_technical_details(result):
    st.subheader("Thông số kỹ thuật")

    c1, c2, c3 = st.columns(3)
    c1.metric("Giá phân tích", fmt_price(result.get("price")))
    c2.metric("Điểm tín hiệu", fmt_num(result.get("score"), 0))
    c3.metric("RSI", fmt_num(result.get("rsi")))

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Điểm BUY xác nhận",
        fmt_num(result.get("confirmed_score"), 0),
    )
    c2.metric(
        "Điểm EARLY BUY",
        fmt_num(result.get("early_score"), 0),
    )
    c3.metric(
        "Khối lượng/TB20",
        fmt_num(result.get("volume_ratio")),
    )

    st.subheader("Thanh khoản")

    c1, c2 = st.columns(2)
    c1.metric(
        "GTGD bình quân 20 phiên",
        fmt_turnover(result.get("avg_turnover_20d")),
    )
    c2.metric(
        "GTGD trung vị 20 phiên",
        fmt_turnover(result.get("median_turnover_20d")),
    )

    st.subheader("Kế hoạch giá và tỷ lệ RR")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cắt lỗ tham khảo", fmt_price(result.get("stop_loss")))
    c2.metric("Mục tiêu tham khảo", fmt_price(result.get("target_price")))
    c3.metric("RR sau chi phí", fmt_num(result.get("reward_risk")) + ":1")

    st.caption(
        "Giá mục tiêu dựa trên vùng kháng cự kỹ thuật thực tế. "
        "Đây là mức tham khảo, không bảo đảm lợi nhuận."
    )

    st.subheader("Xác nhận xu hướng H1")

    c1, c2, c3 = st.columns(3)
    c1.metric("EMA9 H1", fmt_price(result.get("h1_ema9")))
    c2.metric("EMA20 H1", fmt_price(result.get("h1_ema20")))
    c3.metric(
        "H1 xác nhận",
        "Đạt" if result.get("h1_confirmed") else "Chưa đạt",
    )

    st.write(result.get("h1_status", "Chưa có thông tin H1."))

    reasons = result.get("reasons", [])
    if reasons:
        st.subheader("Các yếu tố kỹ thuật")
        for item in reasons:
            st.write("• " + str(item))

    blockers = result.get("blockers", [])
    if blockers:
        st.subheader("Điều kiện chưa đạt")
        for item in blockers:
            st.write("• " + str(item))


def render_hourly_source_status():
    status = get_hourly_source_status()

    st.subheader("Trạng thái nguồn dữ liệu H1")

    for source in ("KBS", "VCI"):
        item = status.get(source, {})
        if item.get("blocked"):
            minutes = max(1, item.get("remaining_seconds", 0) // 60)
            st.warning(
                f"{source}: đang tạm khóa do giới hạn truy vấn. "
                f"Thời gian chờ dự kiến còn {minutes} phút."
            )
        else:
            st.success(f"{source}: chưa ghi nhận trạng thái khóa hạn mức.")


def render_intraday_result(result, data):
    st.subheader(f"Phân tích trong phiên: {result['symbol']}")

    signal = result["signal"]

    if signal == "TÍN HIỆU TĂNG":
        st.success("🟢 TÍN HIỆU TĂNG – KHUNG 5 PHÚT")
    elif signal == "THEO DÕI":
        st.warning("🟡 THEO DÕI – CHƯA ĐỦ ĐIỀU KIỆN")
    else:
        st.info("⚪ CHỜ – CHƯA CÓ TÍN HIỆU RÕ")

    st.write(result["explanation"])

    st.caption(
        "Nến hoàn tất gần nhất: "
        + pd.Timestamp(result["timestamp"]).strftime("%H:%M %d/%m/%Y")
        + " | Nguồn dữ liệu: VCI"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá gần nhất", fmt_price(result["price"]))
    c2.metric("Điểm kỹ thuật", f"{result['score']}/100")
    c3.metric("RSI 14", fmt_num(result["rsi"]))
    c4.metric("KL/TB20 nến", fmt_num(result["volume_ratio"]))

    st.subheader("Xu hướng và động lượng")

    c1, c2, c3 = st.columns(3)
    c1.metric("EMA9", fmt_price(result["ema9"]))
    c2.metric("EMA20", fmt_price(result["ema20"]))
    c3.metric("VWAP", fmt_price(result["vwap"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("MACD", fmt_num(result["macd"], 4))
    c2.metric("MACD Signal", fmt_num(result["macd_signal"], 4))
    c3.metric("ATR14", fmt_price(result["atr"]))

    st.subheader("Vùng giá tham khảo")

    c1, c2, c3 = st.columns(3)
    c1.metric("Hỗ trợ gần", fmt_price(result["support"]))
    c2.metric("Kháng cự gần", fmt_price(result["resistance"]))
    c3.metric("Mốc rủi ro ATR", fmt_price(result["stop_reference"]))

    if result["reasons"]:
        st.subheader("Yếu tố tích cực")
        for item in result["reasons"]:
            st.write("• " + item)

    if result["blockers"]:
        st.subheader("Rủi ro / điều kiện chưa đạt")
        for item in result["blockers"]:
            st.write("• " + item)

    st.subheader("Biểu đồ nến 5 phút")

    chart_df = data.tail(60).copy()
    chart_df = chart_df.set_index("Date")

    st.line_chart(
        chart_df[["Close", "EMA9", "EMA20", "VWAP"]],
        use_container_width=True,
    )

    with st.expander("Xem 30 nến 5 phút gần nhất"):
        view = data.tail(30).copy()
        view["Date"] = pd.to_datetime(view["Date"]).dt.strftime(
            "%H:%M %d/%m"
        )
        st.dataframe(
            view[[
                "Date", "Open", "High", "Low", "Close", "Volume"
            ]],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Lưu ý: tín hiệu 5 phút chỉ phản ánh biến động trong phiên. "
        "Không thay thế điều kiện BUY xác nhận D1 + H1 của chiến lược 2–5 phiên."
    )


# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Phân tích một mã",
        "Quét toàn thị trường",
        "Phân tích trong phiên 5 phút",
    ]
)


# =====================================================
# TAB 1: PHÂN TÍCH MỘT MÃ D1 + H1
# =====================================================

with tab1:
    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10,
        key="daily_symbol",
    ).strip().upper()

    if st.button(
        "PHÂN TÍCH CỔ PHIẾU",
        type="primary",
        use_container_width=True,
    ):
        if not symbol:
            st.error("Vui lòng nhập mã cổ phiếu.")
        else:
            try:
                with st.spinner("Đang lấy dữ liệu D1 và H1..."):
                    history = fetch_stock_history(symbol, length=260)

                    hourly = None
                    hourly_error = None

                    try:
                        hourly = fetch_stock_hourly(
                            symbol,
                            lookback="14d",
                        )
                    except Exception as exc:
                        hourly_error = str(exc)

                    result = analyze_stock(
                        history,
                        symbol=symbol,
                        hourly_df=hourly,
                        hourly_error=hourly_error,
                    )

                st.subheader(f"Kết quả phân tích: {symbol}")
                st.caption(
                    "Ngày dữ liệu D1: "
                    + pd.Timestamp(history.iloc[-1]["Date"]).strftime("%d/%m/%Y")
                )

                render_signal(result.get("signal", "WAIT"))
                st.write(result.get("reason", ""))

                render_technical_details(result)

                if hourly is not None:
                    st.success(
                        "Đã lấy được H1. Nguồn: "
                        + str(hourly.attrs.get("source", "Không xác định"))
                    )
                elif hourly_error:
                    st.warning(
                        "Không lấy được H1. Bot không xác nhận BUY khi "
                        "thiếu dữ liệu H1. Chi tiết: " + hourly_error
                    )

                with st.expander("Xem 10 phiên D1 gần nhất"):
                    st.dataframe(
                        history.tail(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                if hourly is not None:
                    with st.expander("Xem 20 nến H1 gần nhất"):
                        st.dataframe(
                            hourly.tail(20),
                            use_container_width=True,
                            hide_index=True,
                        )

            except Exception as exc:
                st.error(f"Không phân tích được {symbol}: {exc}")

    render_hourly_source_status()


# =====================================================
# TAB 2: QUÉT TOÀN THỊ TRƯỜNG
# =====================================================

with tab2:
    st.subheader("Quét toàn bộ mã cổ phiếu HOSE và HNX")

    if st.button(
        "LẤY DANH SÁCH MÃ HOSE + HNX",
        use_container_width=True,
        key="get_symbols",
    ):
        try:
            symbols = fetch_exchange_symbols(force_refresh=True)
            st.session_state.symbol_list = symbols
            st.session_state.symbol_stats = fetch_exchange_symbols.last_stats
        except Exception as exc:
            st.error(f"Không lấy được danh sách mã: {exc}")

    symbols = st.session_state.symbol_list
    stats = st.session_state.symbol_stats

    if symbols:
        st.write("**Thống kê danh sách mã:**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HOSE", stats.get("hose_count", "—"))
        c2.metric("HNX", stats.get("hnx_count", "—"))
        c3.metric("Sau loại trùng", len(symbols))
        c4.metric("Mã trùng", stats.get("duplicate_count", "—"))

        st.caption("Nguồn danh sách: " + str(stats.get("source", "KBS Listing")))

        with st.expander("Xem danh sách mã được lấy"):
            st.write(", ".join(symbols))

        if st.button(
            "QUÉT TOÀN BỘ DANH SÁCH",
            type="primary",
            use_container_width=True,
            key="scan_all_market",
        ):
            progress = st.progress(0)
            status_box = st.empty()

            def update_progress(done, total):
                progress.progress(done / total if total else 1.0)
                status_box.write(f"Đang phân tích {done}/{total} mã...")

            try:
                with st.spinner(
                    "Đang lấy dữ liệu D1, lọc thanh khoản "
                    "và truy vấn H1 có kiểm soát..."
                ):
                    output = scan_market(
                        symbols=symbols,
                        progress_callback=update_progress,
                    )

                st.session_state.market_scan_output = output
                st.session_state.market_scan_meta = {
                    "symbols_total": len(symbols),
                    "completed_at": datetime.now(VN_TZ).strftime(
                        "%H:%M:%S %d/%m/%Y"
                    ),
                }

                progress.progress(1.0)
                status_box.success("Đã hoàn tất quét.")

            except Exception as exc:
                st.error(f"Không thể quét thị trường: {exc}")

    output = st.session_state.market_scan_output
    meta = st.session_state.market_scan_meta

    if output is not None and meta is not None:
        st.divider()

        st.caption(
            f"Kết quả lúc {meta['completed_at']} — "
            f"{meta['symbols_total']} mã trong danh sách."
        )

        st.subheader("Tổng kết quét")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Đã xử lý", output.get("scanned", 0))
        c2.metric("BUY xác nhận", output.get("confirmed_buy_count", 0))
        c3.metric("EARLY BUY", output.get("early_buy_count", 0))
        c4.metric("WAIT", output.get("wait_count", 0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Bị lọc", output.get("filtered_count", 0))
        c2.metric(
            "Lọc do thanh khoản",
            output.get("liquidity_filtered_count", 0),
        )
        c3.metric("Lỗi D1 thực sự", output.get("d1_error_count", 0))

        st.metric(
            "Thiếu lịch sử dưới 200 phiên",
            output.get("insufficient_history_count", 0),
        )

        st.subheader("Thống kê dữ liệu H1")

        c1, c2, c3 = st.columns(3)
        c1.metric("Đã thử lấy H1", output.get("h1_attempted_count", 0))
        c2.metric("Lấy được H1", output.get("h1_available_count", 0))
        c3.metric("Không lấy được H1", output.get("h1_unavailable_count", 0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Lỗi KBS", output.get("h1_kbs_error_count", 0))
        c2.metric("VCI dự phòng thành công", output.get("h1_vci_fallback_count", 0))
        c3.metric("Lỗi giới hạn API", output.get("h1_rate_limited_count", 0))

        st.metric(
            "Mã chưa truy vấn H1 do hai nguồn hết hạn mức",
            output.get("h1_skipped_rate_limit_count", 0),
        )

        st.caption(
            "H1 chỉ được truy vấn sau khi mã đạt điều kiện D1, "
            "đủ 200 phiên và đạt tiêu chuẩn thanh khoản."
        )

        st.metric("Tổng lỗi ghi nhận", output.get("error_count", 0))

        render_hourly_source_status()

        results = output.get("results", [])

        if results:
            st.subheader("Danh sách tín hiệu")

            display = make_results_dataframe(results)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
            )

            confirmed = [
                x for x in results if x.get("signal") == "BUY"
            ]
            early = [
                x for x in results if x.get("signal") == "EARLY_BUY"
            ]

            if confirmed:
                st.subheader("🟢 BUY xác nhận")
                confirmed_df = make_results_dataframe(confirmed)
                st.dataframe(
                    confirmed_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "TẢI DANH SÁCH BUY",
                    data=confirmed_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="buy_xac_nhan.csv",
                    mime="text/csv",
                )

            if early:
                st.subheader("🟡 EARLY BUY")
                early_df = make_results_dataframe(early)
                st.dataframe(
                    early_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "TẢI DANH SÁCH EARLY BUY",
                    data=early_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="early_buy.csv",
                    mime="text/csv",
                )

            st.download_button(
                "TẢI TOÀN BỘ TÍN HIỆU",
                data=display.to_csv(index=False).encode("utf-8-sig"),
                file_name="tat_ca_tin_hieu.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "Không có mã nào đạt BUY hoặc EARLY BUY. "
                "Không hạ tiêu chuẩn chỉ để tạo thêm tín hiệu."
            )

        errors = output.get("errors", [])

        if errors:
            error_df = pd.DataFrame(errors)

            with st.expander(f"Chi tiết lỗi dữ liệu ({len(errors)} mã)"):
                st.dataframe(
                    error_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "TẢI DANH SÁCH LỖI",
                    data=error_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="loi_du_lieu.csv",
                    mime="text/csv",
                )


# =====================================================
# TAB 3: PHÂN TÍCH TRONG PHIÊN 5 PHÚT
# =====================================================

with tab3:
    st.subheader("Phân tích cổ phiếu trực tiếp trong phiên")

    st.caption(
        "Phân tích một mã bằng nến 5 phút đã hoàn tất. "
        "Nguồn dữ liệu VCI."
    )

    intraday_symbol = st.text_input(
        "Nhập mã cổ phiếu cần phân tích 5 phút",
        value="FPT",
        max_chars=10,
        key="intraday_symbol",
    ).strip().upper()

    c1, c2 = st.columns([1, 1])

    with c1:
        analyze_button = st.button(
            "PHÂN TÍCH 5 PHÚT",
            type="primary",
            use_container_width=True,
            key="run_intraday",
        )

    with c2:
        refresh_button = st.button(
            "CẬP NHẬT DỮ LIỆU",
            use_container_width=True,
            key="refresh_intraday",
        )

    if analyze_button or refresh_button:
        if not intraday_symbol:
            st.error("Vui lòng nhập mã cổ phiếu.")
        else:
            try:
                with st.spinner(
                    f"Đang lấy dữ liệu nến 5 phút của {intraday_symbol}..."
                ):
                    candles = fetch_intraday_5m(
                        intraday_symbol,
                        force_refresh=refresh_button,
                    )

                    intraday_result = analyze_intraday_5m(
                        candles,
                        symbol=intraday_symbol,
                    )

                render_intraday_result(
                    intraday_result,
                    intraday_result["candles"],
                )

            except Exception as exc:
                st.error(
                    f"Không phân tích được {intraday_symbol}: {exc}"
                )

    st.divider()

    st.caption(
        "Tự cập nhật: hiện chưa bật tự động làm mới. "
        "Bạn có thể bấm CẬP NHẬT DỮ LIỆU để lấy nến mới nhất."
    )

    st.caption(
        "Không sử dụng tín hiệu 5 phút làm căn cứ duy nhất để mua. "
        "Đối chiếu xu hướng D1/H1, thanh khoản, vùng giá và rủi ro "
        "trước khi ra quyết định."
    )


# =====================================================
# CHÂN TRANG
# =====================================================

st.divider()

st.caption(
    "Thời điểm hệ thống: "
    + datetime.now(VN_TZ).strftime("%H:%M:%S %d/%m/%Y")
)

st.caption(
    "Dữ liệu D1: vn-market-data. "
    "Dữ liệu H1: Vnstock KBS, dự phòng VCI. "
    "Dữ liệu 5 phút: Vnstock VCI. "
    "Bot chỉ phân tích, không tự động đặt lệnh."
)
