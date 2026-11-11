
import streamlit as st
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock, scan_market
from data_provider import fetch_exchange_symbols
from intraday import fetch_intraday_5m, analyze_intraday_5m


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


# =====================================================
# CẤU HÌNH
# =====================================================

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide",
)


# =====================================================
# SESSION STATE
# =====================================================

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
    "Phân tích cổ phiếu HOSE và HNX cho chiến lược ngắn hạn."
)

st.info(
    "Phân tích mã đơn lẻ: nến 5 phút cập nhật trong phiên. "
    "Quét toàn thị trường: dữ liệu D1."
)

st.caption(
    "Bot chỉ phân tích và đưa ra tín hiệu tham khảo, "
    "không tự động đặt lệnh."
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
            "avg_turnover_20d": "GTGD TB20",
            "median_turnover_20d": "GTGD trung vị",
            "data_date": "Ngày dữ liệu",
            "reason": "Nhận định",
        }
    )


# =====================================================
# PHÂN TÍCH 5 PHÚT
# =====================================================

def render_intraday_analysis(symbol):

    st.subheader("Phân tích cổ phiếu theo thời gian thực")

    st.caption(
        "Khung phân tích: 5 phút | Nguồn dữ liệu: KBS | "
        "Tự cập nhật mỗi 45 giây."
    )

    st.warning(
        "Tín hiệu 5 phút phục vụ giao dịch trong phiên, "
        "không đồng nghĩa với khuyến nghị mua chắc chắn."
    )

    try:
        with st.spinner(
            f"Đang lấy dữ liệu nến 5 phút của {symbol}..."
        ):

            candles = fetch_intraday_5m(
                symbol,
                force_refresh=True,
            )

            result = analyze_intraday_5m(
                candles,
                symbol=symbol,
            )

        st.divider()

        st.subheader(f"Kết quả phân tích: {symbol}")

        last_time = result.get("timestamp")

        if last_time is not None:
            st.caption(
                "Thời điểm nến gần nhất: "
                + pd.Timestamp(last_time).strftime(
                    "%H:%M %d/%m/%Y"
                )
            )

        st.caption(
            "Nguồn dữ liệu: "
            + str(candles.attrs.get("source", "KBS"))
        )

        signal = result["signal"]

        if signal == "TÍN HIỆU TĂNG":
            st.success("🟢 TÍN HIỆU TĂNG")

        elif signal == "THEO DÕI":
            st.warning("🟡 THEO DÕI – CHƯA ĐỦ ĐIỀU KIỆN")

        else:
            st.info("⚪ CHỜ – CHƯA CÓ TÍN HIỆU RÕ")

        st.write(result["explanation"])

        # -----------------------------
        # CHỈ SỐ CHÍNH
        # -----------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Giá gần nhất",
            fmt_price(result["price"]),
        )

        c2.metric(
            "Điểm kỹ thuật",
            f'{result["score"]}/100',
        )

        c3.metric(
            "RSI 14",
            fmt_num(result["rsi"]),
        )

        c4.metric(
            "KL/TB20 nến",
            fmt_num(result["volume_ratio"]),
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "EMA9",
            fmt_price(result["ema9"]),
        )

        c2.metric(
            "EMA20",
            fmt_price(result["ema20"]),
        )

        c3.metric(
            "VWAP",
            fmt_price(result["vwap"]),
        )

        c4.metric(
            "MACD",
            fmt_num(result["macd"], 4),
        )

        # -----------------------------
        # HỖ TRỢ / KHÁNG CỰ / CẮT LỖ
        # -----------------------------

        st.subheader("Các vùng giá tham khảo")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Hỗ trợ",
            fmt_price(result["support"]),
        )

        c2.metric(
            "Kháng cự",
            fmt_price(result["resistance"]),
        )

        c3.metric(
            "Cắt lỗ tham khảo",
            fmt_price(result["stop_reference"]),
        )

        # -----------------------------
        # BIỂU ĐỒ
        # -----------------------------

        st.subheader("Biểu đồ giá 5 phút và đường kỹ thuật")

        chart_data = result["candles"].copy()

        chart_columns = [
            "Close",
            "EMA9",
            "EMA20",
            "VWAP",
        ]

        chart_data = chart_data.set_index("Date")

        st.line_chart(
            chart_data[chart_columns],
            height=400,
        )

        # -----------------------------
        # YẾU TỐ TÍCH CỰC
        # -----------------------------

        if result["reasons"]:
            st.subheader("Yếu tố tích cực")

            for reason in result["reasons"]:
                st.write("✅ " + reason)

        # -----------------------------
        # CẢNH BÁO
        # -----------------------------

        if result["blockers"]:
            st.subheader("Điều kiện chưa đạt / cảnh báo")

            for blocker in result["blockers"]:
                st.write("⚠️ " + blocker)

        # -----------------------------
        # DỮ LIỆU NẾN
        # -----------------------------

        with st.expander("Xem dữ liệu 5 phút gần nhất"):

            display_columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "EMA9",
                "EMA20",
                "RSI14",
                "VWAP",
                "MACD",
            ]

            available = [
                c for c in display_columns
                if c in chart_data.columns
            ]

            st.dataframe(
                chart_data[available].tail(30),
                use_container_width=True,
            )

    except Exception as exc:

        st.error(
            f"Không phân tích được {symbol}: {exc}"
        )

        st.caption(
            "Nếu đang ngoài phiên, dữ liệu có thể không có nến mới. "
            "Nếu đang trong phiên mà dữ liệu quá cũ, bot sẽ không "
            "phát tín hiệu để tránh sử dụng dữ liệu trễ."
        )


# =====================================================
# TỰ ĐỘNG CẬP NHẬT PHÂN TÍCH 5 PHÚT
# =====================================================

@st.fragment(run_every="45s")
def intraday_panel(symbol):

    if not symbol:
        st.warning("Vui lòng nhập mã cổ phiếu.")
        return

    render_intraday_analysis(symbol)


# =====================================================
# HIỂN THỊ KẾT QUẢ QUÉT THỊ TRƯỜNG D1
# =====================================================

def render_market_results(output, meta):

    if output is None or meta is None:
        return

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
    c3.metric("Lỗi D1", output.get("d1_error_count", 0))

    st.metric(
        "Thiếu lịch sử dưới 200 phiên",
        output.get("insufficient_history_count", 0),
    )

    st.metric(
        "Tổng lỗi ghi nhận",
        output.get("error_count", 0),
    )

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
            x for x in results
            if x.get("signal") == "BUY"
        ]

        early = [
            x for x in results
            if x.get("signal") == "EARLY_BUY"
        ]

        waiting = [
            x for x in results
            if x.get("signal") == "WAIT"
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
                data=confirmed_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
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
                data=early_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="early_buy.csv",
                mime="text/csv",
            )

        if waiting:

            with st.expander(
                f"Xem các mã WAIT ({len(waiting)} mã)"
            ):

                waiting_df = make_results_dataframe(waiting)

                st.dataframe(
                    waiting_df,
                    use_container_width=True,
                    hide_index=True,
                )

        st.download_button(
            "TẢI TOÀN BỘ KẾT QUẢ",
            data=display.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="ket_qua_quet_thi_truong.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "Không có mã nào đạt BUY hoặc EARLY BUY. "
            "Không hạ tiêu chuẩn để tạo thêm tín hiệu."
        )

    errors = output.get("errors", [])

    if errors:

        error_df = pd.DataFrame(errors)

        with st.expander(
            f"Chi tiết lỗi dữ liệu ({len(errors)} mã)"
        ):

            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "TẢI DANH SÁCH LỖI",
                data=error_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="loi_du_lieu.csv",
                mime="text/csv",
            )


# =====================================================
# GIAO DIỆN 2 CHỨC NĂNG
# =====================================================

tab1, tab2 = st.tabs(
    [
        "Phân tích một mã – 5 phút",
        "Quét toàn thị trường – D1",
    ]
)


# =====================================================
# TAB 1: PHÂN TÍCH 5 PHÚT
# =====================================================

with tab1:

    st.subheader("Phân tích cổ phiếu đơn lẻ")

    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10,
        key="intraday_symbol",
    ).strip().upper()

    intraday_panel(symbol)


# =====================================================
# TAB 2: QUÉT TOÀN THỊ TRƯỜNG D1
# =====================================================

with tab2:

    st.subheader(
        "Quét toàn bộ mã cổ phiếu HOSE và HNX"
    )

    st.caption(
        "Chức năng này giữ nguyên thuật toán D1, "
        "không sử dụng dữ liệu 5 phút."
    )

    if st.button(
        "LẤY DANH SÁCH MÃ HOSE + HNX",
        use_container_width=True,
        key="get_symbols",
    ):

        try:

            symbols = fetch_exchange_symbols(
                force_refresh=True
            )

            st.session_state.symbol_list = symbols

            st.session_state.symbol_stats = (
                fetch_exchange_symbols.last_stats
            )

            st.success(
                f"Đã lấy được {len(symbols)} mã cổ phiếu."
            )

        except Exception as exc:

            st.error(
                f"Không lấy được danh sách mã: {exc}"
            )

    symbols = st.session_state.symbol_list
    stats = st.session_state.symbol_stats

    if symbols:

        st.write("**Thống kê danh sách mã:**")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("HOSE", stats.get("hose_count", "—"))
        c2.metric("HNX", stats.get("hnx_count", "—"))
        c3.metric("Sau loại trùng", len(symbols))
        c4.metric("Mã trùng", stats.get("duplicate_count", "—"))

        st.caption(
            "Nguồn danh sách: "
            + str(
                stats.get(
                    "source",
                    "Vietcap public listing",
                )
            )
        )

        with st.expander("Xem danh sách mã"):
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
                progress.progress(
                    done / total if total else 1.0
                )

                status_box.write(
                    f"Đang phân tích {done}/{total} mã..."
                )

            try:

                with st.spinner(
                    "Đang lấy dữ liệu D1 và phân tích kỹ thuật..."
                ):

                    output = scan_market(
                        symbols=symbols,
                        progress_callback=update_progress,
                    )

                st.session_state.market_scan_output = output

                st.session_state.market_scan_meta = {
                    "symbols_total": len(symbols),
                    "completed_at": datetime.now(
                        VN_TZ
                    ).strftime("%H:%M:%S %d/%m/%Y"),
                }

                progress.progress(1.0)

                status_box.success(
                    "Đã hoàn tất quét thị trường."
                )

            except Exception as exc:

                st.error(
                    f"Không thể quét thị trường: {exc}"
                )

    render_market_results(
        st.session_state.market_scan_output,
        st.session_state.market_scan_meta,
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
    "Phân tích đơn lẻ: nến 5 phút từ KBS. "
    "Quét thị trường: D1 từ vn-market-data. "
    "Không tự động đặt lệnh."
)
