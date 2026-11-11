
import streamlit as st
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock, scan_market
from data_provider import (
    fetch_stock_history,
    fetch_exchange_symbols,
)


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


# =====================================================
# CẤU HÌNH TRANG
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
    "Sàng lọc cổ phiếu HOSE và HNX cho chiến lược ngắn hạn 2–5 phiên."
)

st.info(
    "Bot chỉ phân tích và đưa ra tín hiệu tham khảo, "
    "không tự động đặt lệnh."
)

st.caption(
    "BUY xác nhận: điểm tối thiểu 78, RSI 45–68, "
    "xu hướng và động lượng được đánh giá trên D1, "
    "RR sau chi phí tối thiểu 2:1."
)

st.caption(
    "EARLY BUY: điểm tối thiểu 70, "
    "tín hiệu theo dõi sớm, chưa phải BUY xác nhận."
)

st.caption(
    "Thanh khoản tối thiểu: bình quân 20 phiên 5 tỷ/ngày, "
    "trung vị 20 phiên 3 tỷ/ngày. "
    "Lịch sử tối thiểu 200 phiên."
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


def render_technical_details(result):
    st.subheader("Thông số kỹ thuật D1")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Giá phân tích",
        fmt_price(result.get("price")),
    )

    c2.metric(
        "Điểm tín hiệu",
        fmt_num(result.get("score"), 0),
    )

    c3.metric(
        "RSI",
        fmt_num(result.get("rsi")),
    )

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

    c1.metric(
        "Cắt lỗ tham khảo",
        fmt_price(result.get("stop_loss")),
    )

    c2.metric(
        "Mục tiêu tham khảo",
        fmt_price(result.get("target_price")),
    )

    c3.metric(
        "RR sau chi phí",
        fmt_num(result.get("reward_risk")) + ":1",
    )

    st.caption(
        "Giá mục tiêu dựa trên vùng kháng cự kỹ thuật thực tế. "
        "Đây là mức tham khảo, không bảo đảm lợi nhuận."
    )

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


# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(
    [
        "Phân tích một mã D1",
        "Quét toàn thị trường D1",
    ]
)


# =====================================================
# TAB 1: PHÂN TÍCH MỘT MÃ
# =====================================================

with tab1:

    st.subheader("Phân tích cổ phiếu đơn lẻ")

    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10,
        key="daily_symbol",
    ).strip().upper()

    analyze_button = st.button(
        "PHÂN TÍCH CỔ PHIẾU",
        type="primary",
        use_container_width=True,
        key="analyze_daily",
    )

    if analyze_button:

        if not symbol:
            st.error("Vui lòng nhập mã cổ phiếu.")

        else:

            try:
                with st.spinner(
                    f"Đang lấy dữ liệu D1 của {symbol}..."
                ):

                    history = fetch_stock_history(
                        symbol,
                        length=260,
                    )

                    result = analyze_stock(
                        history,
                        symbol=symbol,
                    )

                st.divider()

                st.subheader(
                    f"Kết quả phân tích: {symbol}"
                )

                st.caption(
                    "Ngày dữ liệu D1: "
                    + pd.Timestamp(
                        history.iloc[-1]["Date"]
                    ).strftime("%d/%m/%Y")
                )

                render_signal(
                    result.get("signal", "WAIT")
                )

                st.write(
                    result.get("reason", "")
                )

                render_technical_details(result)

                with st.expander(
                    "Xem 20 phiên D1 gần nhất"
                ):

                    st.dataframe(
                        history.tail(20),
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as exc:

                st.error(
                    f"Không phân tích được {symbol}: {exc}"
                )


# =====================================================
# TAB 2: QUÉT TOÀN THỊ TRƯỜNG
# =====================================================

with tab2:

    st.subheader(
        "Quét toàn bộ mã cổ phiếu HOSE và HNX"
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

        c1.metric(
            "HOSE",
            stats.get("hose_count", "—"),
        )

        c2.metric(
            "HNX",
            stats.get("hnx_count", "—"),
        )

        c3.metric(
            "Sau loại trùng",
            len(symbols),
        )

        c4.metric(
            "Mã trùng",
            stats.get("duplicate_count", "—"),
        )

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

        c1.metric(
            "Đã xử lý",
            output.get("scanned", 0),
        )

        c2.metric(
            "BUY xác nhận",
            output.get("confirmed_buy_count", 0),
        )

        c3.metric(
            "EARLY BUY",
            output.get("early_buy_count", 0),
        )

        c4.metric(
            "WAIT",
            output.get("wait_count", 0),
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Bị lọc",
            output.get("filtered_count", 0),
        )

        c2.metric(
            "Lọc do thanh khoản",
            output.get("liquidity_filtered_count", 0),
        )

        c3.metric(
            "Lỗi D1 thực sự",
            output.get("d1_error_count", 0),
        )

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

                confirmed_df = make_results_dataframe(
                    confirmed
                )

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

                early_df = make_results_dataframe(
                    early
                )

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

                    waiting_df = make_results_dataframe(
                        waiting
                    )

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
                "Không hạ tiêu chuẩn chỉ để tạo thêm tín hiệu."
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
# CHÂN TRANG
# =====================================================

st.divider()

st.caption(
    "Thời điểm hệ thống: "
    + datetime.now(VN_TZ).strftime("%H:%M:%S %d/%m/%Y")
)

st.caption(
    "Dữ liệu phân tích: D1 từ vn-market-data. "
    "Bot chỉ phân tích, không tự động đặt lệnh."
)
