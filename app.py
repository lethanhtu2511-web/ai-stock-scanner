
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
    "Bot chỉ đánh giá tín hiệu cổ phiếu. "
    "Không yêu cầu nhập vốn, không tính số lượng mua "
    "và không tự động đặt lệnh."
)

st.caption(
    "Bộ lọc thanh khoản: GTGD bình quân 20 phiên "
    "tối thiểu 5 tỷ/ngày và trung vị tối thiểu 3 tỷ/ngày."
)

st.caption(
    "Yêu cầu tối thiểu 200 phiên dữ liệu D1. "
    "Tín hiệu BUY cần thêm xác nhận H1, kháng cự "
    "và tỷ lệ lợi nhuận/rủi ro sau chi phí tối thiểu 2:1."
)


tab1, tab2 = st.tabs([
    "Phân tích một mã",
    "Quét toàn thị trường"
])


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
                    "Đang lấy dữ liệu D1 và H1..."
                ):

                    history = fetch_stock_history(
                        symbol,
                        length=260
                    )

                    hourly = None
                    hourly_error = None

                    try:
                        hourly = fetch_stock_hourly(symbol)
                    except Exception as exc:
                        hourly_error = str(exc)

                    result = analyze_stock(
                        history,
                        symbol=symbol,
                        hourly_df=hourly
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

                if result.get("signal") == "BUY":
                    st.success("🟢 TÍN HIỆU BUY")

                elif result.get("signal") == "FILTERED":
                    st.warning("⚪ MÃ BỊ LOẠI KHỎI DANH SÁCH")

                else:
                    st.warning("🟡 TÍN HIỆU WAIT")

                st.write(result.get("reason", ""))

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá đóng cửa",
                    f"{result.get('price', 0):,.0f} đ"
                )

                c2.metric(
                    "Điểm kỹ thuật",
                    f"{result.get('score', 0)}/100"
                )

                c3.metric(
                    "RSI",
                    result.get("rsi", "—")
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá cắt lỗ",
                    (
                        f"{result.get('stop_loss'):,.0f} đ"
                        if result.get("stop_loss") is not None
                        else "—"
                    )
                )

                c2.metric(
                    "Giá mục tiêu",
                    (
                        f"{result.get('target_price'):,.0f} đ"
                        if result.get("target_price") is not None
                        else "—"
                    )
                )

                c3.metric(
                    "Lợi nhuận/rủi ro",
                    (
                        f"{result.get('reward_risk')}:1"
                        if result.get("reward_risk") is not None
                        else "—"
                    )
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "GTGD bình quân 20 phiên",
                    (
                        f"{result.get('avg_turnover_20d', 0) / 1e9:.2f} "
                        "tỷ/ngày"
                    )
                )

                c2.metric(
                    "GTGD trung vị 20 phiên",
                    (
                        f"{result.get('median_turnover_20d', 0) / 1e9:.2f} "
                        "tỷ/ngày"
                    )
                )

                st.subheader("Kháng cự và điểm mua")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Kháng cự gần",
                    (
                        f"{result.get('resistance'):,.0f} đ"
                        if result.get('resistance') is not None
                        else "Chưa xác định"
                    )
                )

                c2.metric(
                    "Dư địa tới kháng cự",
                    (
                        f"{result.get('resistance_gap_pct'):.2f}%"
                        if result.get('resistance_gap_pct') is not None
                        else "—"
                    )
                )

                c3.metric(
                    "Breakout D1",
                    (
                        "Đã xác nhận"
                        if result.get('breakout_confirmed')
                        else "Chưa xác nhận"
                    )
                )

                st.subheader("Xác nhận xu hướng H1")

                st.write(
                    "**Trạng thái:**",
                    result.get("h1_status", "Không có dữ liệu")
                )

                if result.get("h1_last_bar"):
                    st.write(
                        "**Nến H1 gần nhất:**",
                        result.get("h1_last_bar")
                    )

                c1, c2 = st.columns(2)

                c1.metric(
                    "EMA9 H1",
                    (
                        f"{result.get('h1_ema9'):,.0f} đ"
                        if result.get('h1_ema9') is not None
                        else "—"
                    )
                )

                c2.metric(
                    "EMA20 H1",
                    (
                        f"{result.get('h1_ema20'):,.0f} đ"
                        if result.get('h1_ema20') is not None
                        else "—"
                    )
                )

                if hourly is not None:
                    st.success(
                        "Đã lấy được dữ liệu H1. "
                        f"Nguồn: {hourly.attrs.get('source', 'Không xác định')}."
                    )

                if hourly_error:
                    st.warning(
                        "Không lấy được dữ liệu H1. "
                        "Bot không xác nhận BUY khi thiếu H1. "
                        f"Chi tiết: {hourly_error}"
                    )

                st.subheader("Các yếu tố kỹ thuật")

                for reason in result.get("reasons", []):
                    st.write("• " + reason)

                blockers = result.get("blockers", [])

                if blockers:
                    st.subheader("Điều kiện chưa đạt")

                    for blocker in blockers:
                        st.write("• " + blocker)

                st.caption(
                    "D1: MA20, MA50, MA200, RSI, MACD, ATR, "
                    "khối lượng và vùng kháng cự. "
                    "H1: EMA9 và EMA20 trên nến một giờ đã hoàn tất."
                )

                with st.expander("Xem 10 phiên D1 gần nhất"):
                    st.dataframe(
                        history.tail(10),
                        use_container_width=True,
                        hide_index=True
                    )

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
        "Bot sử dụng dữ liệu nến D1 đã hoàn tất và nến H1 "
        "để xác nhận điểm mua. Không sử dụng bảng giá trực tiếp."
    )

    st.warning(
        "Mã có dưới 200 phiên dữ liệu sẽ bị loại riêng, "
        "không được tính là lỗi truy vấn."
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
                "Đang phân tích D1, H1 và vùng kháng cự..."
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
    "Giá mục tiêu được tính theo tỷ lệ lợi nhuận/rủi ro "
    "sau chi phí và có xét vùng kháng cự. "
    "Đây là mức tham khảo, không bảo đảm lợi nhuận. "
    "Ứng dụng không tự động đặt lệnh mua/bán."
)

st.caption(
    "Nguồn D1: DNSE/Vietcap thông qua vn-market-data. "
    "Nguồn H1: KBS ưu tiên, VCI dự phòng thông qua Vnstock."
)
