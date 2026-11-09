
import streamlit as st
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock, scan_market

from data_provider import (
    fetch_stock_history,
    fetch_stock_hourly,
    fetch_exchange_symbols
)


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


# =====================================================
# CẤU HÌNH TRANG
# =====================================================

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)


if "market_scan_output" not in st.session_state:
    st.session_state["market_scan_output"] = None

if "market_scan_meta" not in st.session_state:
    st.session_state["market_scan_meta"] = None


st.title("📈 AI STOCK SCANNER")

st.caption(
    "Sàng lọc cổ phiếu HOSE/HNX cho chiến lược ngắn hạn 2–5 phiên."
)

st.info(
    "Bot ưu tiên chất lượng tín hiệu hơn số lượng. "
    "Không yêu cầu nhập vốn, không tính số lượng mua "
    "và không tự động đặt lệnh."
)

st.caption(
    "Bộ lọc thanh khoản: GTGD bình quân 20 phiên "
    "tối thiểu 5 tỷ/ngày và trung vị tối thiểu 3 tỷ/ngày."
)

st.caption(
    "Yêu cầu tối thiểu 200 phiên D1, điểm kỹ thuật từ 78, "
    "RSI 45–68, xác nhận H1 và tỷ lệ lợi nhuận/rủi ro "
    "sau chi phí tối thiểu 2:1."
)


tab1, tab2 = st.tabs([
    "Phân tích một mã",
    "Quét toàn thị trường"
])


# =====================================================
# HÀM HỖ TRỢ
# =====================================================

def format_price(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.0f} đ"


def format_number(value, digits=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def show_optional_metric(value):
    if value is None or pd.isna(value):
        return "—"
    return str(value)


# =====================================================
# TAB 1: PHÂN TÍCH MỘT MÃ
# =====================================================

with tab1:

    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10
    ).strip().upper()

    if st.button(
        "PHÂN TÍCH CỔ PHIẾU",
        type="primary",
        use_container_width=True
    ):

        if not symbol:
            st.error("Vui lòng nhập mã cổ phiếu.")

        else:
            try:
                with st.spinner(
                    "Đang lấy dữ liệu D1, H1 và VN-Index..."
                ):

                    # Dữ liệu D1
                    history = fetch_stock_history(
                        symbol,
                        length=260
                    )

                    if history is None or history.empty:
                        raise RuntimeError(
                            "Không có dữ liệu D1 của cổ phiếu."
                        )

                    # Dữ liệu H1
                    hourly = None
                    hourly_error = None

                    try:
                        hourly = fetch_stock_hourly(symbol)
                    except Exception as exc:
                        hourly_error = str(exc)

                    # Dữ liệu VN-Index
                    benchmark = None
                    benchmark_error = None

                    try:
                        benchmark = fetch_stock_history(
                            "VNINDEX",
                            length=260
                        )

                        if benchmark is None or benchmark.empty:
                            benchmark_error = (
                                "Nguồn dữ liệu trả về VN-Index rỗng."
                            )
                            benchmark = None

                    except Exception as exc:
                        benchmark_error = str(exc)

                    # Phân tích cổ phiếu
                    result = analyze_stock(
                        history,
                        symbol=symbol,
                        hourly_df=hourly,
                        benchmark_df=benchmark
                    )

                last_date = pd.Timestamp(
                    history.iloc[-1]["Date"]
                ).strftime("%d/%m/%Y")

                st.subheader(
                    f"Kết quả phân tích: {symbol}"
                )

                st.info(
                    f"Dữ liệu D1 đóng cửa gần nhất: {last_date}. "
                    "Chỉ sử dụng nến đã hoàn tất."
                )

                # Tín hiệu
                signal = result.get("signal")

                if signal == "BUY":
                    st.success("🟢 TÍN HIỆU BUY")

                elif signal == "FILTERED":
                    st.warning("⚪ MÃ BỊ LOẠI KHỎI DANH SÁCH")

                else:
                    st.warning("🟡 TÍN HIỆU WAIT")

                st.write(result.get("reason", ""))

                # Bối cảnh thị trường
                st.subheader("Bối cảnh thị trường")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Xu hướng VN-Index",
                    show_optional_metric(
                        result.get("market_regime")
                    )
                )

                c2.metric(
                    "Sức mạnh tương đối so với VN-Index",
                    (
                        f"{format_number(result.get('relative_strength'))}%"
                        if result.get("relative_strength") is not None
                        else "—"
                    )
                )

                if benchmark_error:
                    st.warning(
                        "Chưa lấy được dữ liệu VN-Index: "
                        + benchmark_error
                    )

                # Chỉ số kỹ thuật
                st.subheader("Chỉ số kỹ thuật")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá đóng cửa",
                    format_price(result.get("price"))
                )

                c2.metric(
                    "Điểm kỹ thuật",
                    f"{result.get('score', 0)}/100"
                )

                c3.metric(
                    "RSI",
                    format_number(result.get("rsi"))
                )

                # Cắt lỗ, mục tiêu, RR
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá cắt lỗ",
                    format_price(result.get("stop_loss"))
                )

                c2.metric(
                    "Giá mục tiêu",
                    format_price(result.get("target_price"))
                )

                c3.metric(
                    "Lợi nhuận/rủi ro",
                    (
                        f"{format_number(result.get('reward_risk'))}:1"
                        if result.get("reward_risk") is not None
                        else "—"
                    )
                )

                # Thanh khoản
                st.subheader("Thanh khoản")

                c1, c2 = st.columns(2)

                avg_turnover = result.get(
                    "avg_turnover_20d"
                )

                median_turnover = result.get(
                    "median_turnover_20d"
                )

                c1.metric(
                    "GTGD bình quân 20 phiên",
                    (
                        f"{avg_turnover / 1e9:.2f} tỷ/ngày"
                        if avg_turnover is not None
                        else "—"
                    )
                )

                c2.metric(
                    "GTGD trung vị 20 phiên",
                    (
                        f"{median_turnover / 1e9:.2f} tỷ/ngày"
                        if median_turnover is not None
                        else "—"
                    )
                )

                # Kháng cự và breakout
                st.subheader("Kháng cự và điểm mua")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Kháng cự gần",
                    format_price(result.get("resistance"))
                )

                c2.metric(
                    "Dư địa tới kháng cự",
                    (
                        f"{format_number(result.get('resistance_gap_pct'))}%"
                        if result.get("resistance_gap_pct") is not None
                        else "—"
                    )
                )

                c3.metric(
                    "Breakout D1",
                    (
                        "Đã xác nhận"
                        if result.get("breakout_confirmed")
                        else "Chưa xác nhận"
                    )
                )

                # Xác nhận H1
                st.subheader("Xác nhận xu hướng H1")

                st.write(
                    "**Trạng thái:**",
                    result.get(
                        "h1_status",
                        "Không có dữ liệu"
                    )
                )

                if result.get("h1_last_bar"):
                    st.write(
                        "**Nến H1 gần nhất:**",
                        result.get("h1_last_bar")
                    )

                c1, c2 = st.columns(2)

                c1.metric(
                    "EMA9 H1",
                    format_price(result.get("h1_ema9"))
                )

                c2.metric(
                    "EMA20 H1",
                    format_price(result.get("h1_ema20"))
                )

                if hourly is not None:
                    st.success(
                        "Đã lấy được dữ liệu H1. "
                        f"Nguồn: {hourly.attrs.get('source', 'Không xác định')}."
                    )

                if hourly_error:
                    st.warning(
                        "Không lấy được dữ liệu H1. "
                        "Không xác nhận BUY khi thiếu H1. "
                        f"Chi tiết: {hourly_error}"
                    )

                # Các yếu tố kỹ thuật
                st.subheader("Các yếu tố kỹ thuật")

                reasons = result.get("reasons", [])

                if reasons:
                    for reason in reasons:
                        st.write("• " + str(reason))
                else:
                    st.write("Chưa có thông tin.")

                # Điều kiện chưa đạt
                blockers = result.get("blockers", [])

                if blockers:
                    st.subheader("Điều kiện chưa đạt")

                    for blocker in blockers:
                        st.write("• " + str(blocker))

                st.caption(
                    "D1: MA20, MA50, MA200, RSI, MACD, ATR, "
                    "khối lượng và vùng kháng cự. "
                    "H1: EMA9 và EMA20 trên nến một giờ đã hoàn tất."
                )

                # Dữ liệu D1
                with st.expander("Xem 10 phiên D1 gần nhất"):
                    st.dataframe(
                        history.tail(10),
                        use_container_width=True,
                        hide_index=True
                    )

                # Dữ liệu H1
                if hourly is not None:
                    with st.expander("Xem 20 nến H1 gần nhất"):
                        st.dataframe(
                            hourly.tail(20),
                            use_container_width=True,
                            hide_index=True
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

    st.info(
        "Bot sử dụng dữ liệu D1 đã hoàn tất và H1 "
        "để xác nhận điểm mua. Không sử dụng bảng giá trực tiếp."
    )

    st.warning(
        "Mã có dưới 200 phiên dữ liệu sẽ được thống kê riêng, "
        "không tính chung với lỗi truy vấn dữ liệu."
    )

    if st.button(
        "QUÉT TOÀN BỘ HOSE + HNX",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)
        status_box = st.empty()

        try:
            with st.spinner(
                "Đang lấy danh sách cổ phiếu HOSE và HNX..."
            ):
                symbols = fetch_exchange_symbols()

            if not symbols:
                raise RuntimeError(
                    "Danh sách cổ phiếu đang rỗng."
                )

            st.write(
                f"**Danh sách cổ phiếu:** {len(symbols)} mã."
            )

            def update_progress(done, total):
                progress.progress(
                    done / total if total else 1.0
                )

                status_box.write(
                    f"Đang phân tích {done}/{total} mã..."
                )

            with st.spinner(
                "Đang phân tích D1, H1 và các điều kiện kỹ thuật..."
            ):
                output = scan_market(
                    max_symbols=len(symbols),
                    progress_callback=update_progress
                )

            st.session_state["market_scan_output"] = output

            st.session_state["market_scan_meta"] = {
                "symbols_total": len(symbols),
                "completed_at": datetime.now(
                    VN_TZ
                ).strftime("%H:%M:%S %d/%m/%Y")
            }

        except Exception as exc:
            st.error(
                f"Không thể quét toàn thị trường: {exc}"
            )

    output = st.session_state.get(
        "market_scan_output"
    )

    meta = st.session_state.get(
        "market_scan_meta"
    )

    if output is not None and meta is not None:

        st.divider()

        st.caption(
            f"Kết quả lần quét lúc {meta['completed_at']} "
            f"({meta['symbols_total']} mã)."
        )

        # Tổng quan
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Đã xử lý",
            f"{output.get('scanned', 0)} mã"
        )

        c2.metric(
            "Tín hiệu BUY",
            output.get("buy_count", 0)
        )

        c3.metric(
            "Lỗi dữ liệu thực sự",
            output.get("error_count", 0)
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Thiếu 200 phiên - bị loại",
            output.get("insufficient_history_count", 0)
        )

        c2.metric(
            "Loại do thanh khoản/điều kiện",
            output.get("filtered_count", 0)
        )

        c3.metric(
            "Không lấy được H1",
            output.get("hourly_unavailable_count", 0)
        )

        # Lỗi VN-Index
        market_error = output.get("market_regime_error")

        if market_error:
            st.warning(
                "Lượt quét chưa lấy được VN-Index. "
                "Các điều kiện liên quan đến thị trường "
                "không thể được xác nhận. "
                f"Chi tiết: {market_error}"
            )

        # Danh sách BUY
        if output.get("results"):

            display = pd.DataFrame(
                output["results"]
            )

            columns = [
                "symbol",
                "score",
                "price",
                "stop_loss",
                "target_price",
                "reward_risk",
                "resistance",
                "resistance_gap_pct",
                "breakout_confirmed",
                "h1_status",
                "rsi",
                "volume_ratio",
                "avg_turnover_20d",
                "median_turnover_20d",
                "data_date"
            ]

            display = display[
                [c for c in columns if c in display.columns]
            ]

            display = display.rename(columns={
                "symbol": "Mã",
                "score": "Điểm",
                "price": "Giá đóng cửa",
                "stop_loss": "Cắt lỗ",
                "target_price": "Mục tiêu",
                "reward_risk": "LN/RR",
                "resistance": "Kháng cự gần",
                "resistance_gap_pct": "Cách kháng cự (%)",
                "breakout_confirmed": "Breakout D1",
                "h1_status": "Xác nhận H1",
                "rsi": "RSI",
                "volume_ratio": "KL/TB20",
                "avg_turnover_20d": "GTGD TB20 (đồng/ngày)",
                "median_turnover_20d": "GTGD trung vị 20 phiên",
                "data_date": "Ngày dữ liệu"
            })

            st.subheader("Danh sách cổ phiếu BUY")

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "TẢI DANH SÁCH TÍN HIỆU BUY",
                data=display.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="co_phieu_tin_hieu_buy.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.info(
                "Không có mã nào đạt đủ điều kiện BUY. "
                "Không có tín hiệu cũng là kết quả hợp lệ."
            )

        # Danh sách lỗi
        errors = output.get("errors", [])

        if errors:

            error_df = pd.DataFrame(errors)

            with st.expander(
                f"Xem chi tiết lỗi dữ liệu ({len(errors)} mã)"
            ):

                st.dataframe(
                    error_df.head(20),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "TẢI TOÀN BỘ LỖI DỮ LIỆU",
                    data=error_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="loi_du_lieu_co_phieu.csv",
                    mime="text/csv",
                    use_container_width=True
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
    "Bot chỉ đưa ra tín hiệu tham khảo dựa trên dữ liệu lịch sử. "
    "Tín hiệu không bảo đảm lợi nhuận và không thay thế "
    "việc kiểm tra thị trường trước khi giao dịch."
)

st.caption(
    "Nguồn D1: DNSE/Vietcap thông qua vn-market-data. "
    "Nguồn H1: KBS ưu tiên, VCI dự phòng thông qua Vnstock."
)
