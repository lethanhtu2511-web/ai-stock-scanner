import streamlit as st
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import (
    analyze_stock,
    scan_market,
    INITIAL_CAPITAL
)

from data_provider import (
    fetch_stock_history,
    fetch_exchange_symbols
)


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)


# Lưu kết quả quét trong phiên Streamlit.
if "market_scan_output" not in st.session_state:
    st.session_state["market_scan_output"] = None

if "market_scan_meta" not in st.session_state:
    st.session_state["market_scan_meta"] = None


st.title("📈 AI STOCK SCANNER")

st.caption(
    "Sàng lọc cổ phiếu HOSE/HNX cho chiến lược ngắn hạn 2–5 phiên."
)


# =====================================================
# QUẢN TRỊ VỐN
# =====================================================

with st.sidebar:

    st.header("Quản trị vốn")

    capital = st.number_input(
        "Vốn giao dịch (đồng)",
        min_value=1_000_000,
        value=INITIAL_CAPITAL,
        step=1_000_000
    )

    risk_pct = st.number_input(
        "Rủi ro tối đa mỗi lệnh (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    ) / 100

    st.caption(
        "Mặc định: vốn 10 triệu, rủi ro 2%."
    )

    st.divider()

    st.caption(
        "Dữ liệu phân tích: nến ngày OHLCV. "
        "Không sử dụng bảng giá trực tiếp trong phiên."
    )

    st.caption(
        "Lọc thanh khoản: GTGD bình quân 20 phiên "
        "tối thiểu 5 tỷ/ngày và trung vị tối thiểu 3 tỷ/ngày."
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

            st.error(
                "Vui lòng nhập mã cổ phiếu."
            )

        else:

            try:

                with st.spinner(
                    "Đang lấy dữ liệu nến ngày..."
                ):

                    history = fetch_stock_history(
                        symbol,
                        length=260
                    )

                    result = analyze_stock(
                        history,
                        capital=capital,
                        risk_percent=risk_pct,
                        symbol=symbol
                    )

                last_date = pd.Timestamp(
                    history.iloc[-1]["Date"]
                ).strftime("%d/%m/%Y")

                st.subheader(
                    f"Kết quả phân tích: {symbol}"
                )

                st.info(
                    f"Dữ liệu đóng cửa gần nhất: {last_date}. "
                    "Chỉ sử dụng nến ngày đã hoàn tất."
                )

                if result.get("signal") == "BUY":

                    st.success(
                        "🟢 TÍN HIỆU BUY"
                    )

                elif result.get("signal") == "FILTERED":

                    st.warning(
                        "⚪ MÃ BỊ LOẠI KHỎI DANH SÁCH"
                    )

                else:

                    st.warning(
                        "🟡 TÍN HIỆU WAIT"
                    )

                st.write(
                    result.get("reason", "")
                )

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
                    f"{result.get('stop_loss', 0):,.0f} đ"
                )

                c2.metric(
                    "Giá mục tiêu",
                    f"{result.get('target_price', 0):,.0f} đ"
                )

                c3.metric(
                    "Lợi nhuận/rủi ro",
                    f"{result.get('reward_risk', 0)}:1"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Khối lượng dự kiến",
                    f"{result.get('quantity', 0):,} cp"
                )

                c2.metric(
                    "Vốn sử dụng",
                    f"{result.get('capital_used', 0):,.0f} đ"
                )

                c3.metric(
                    "Rủi ro dự kiến",
                    f"{result.get('risk_money', 0):,.0f} đ"
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "GTGD bình quân 20 phiên",
                    (
                        f"{result.get('avg_turnover_20d', 0) / 1e9:.2f} "
                        "tỷ đồng/ngày"
                    )
                )

                c2.metric(
                    "GTGD trung vị 20 phiên",
                    (
                        f"{result.get('median_turnover_20d', 0) / 1e9:.2f} "
                        "tỷ đồng/ngày"
                    )
                )

                st.subheader(
                    "Các yếu tố kỹ thuật"
                )

                for reason in result.get("reasons", []):

                    st.write(
                        "• " + reason
                    )

                st.caption(
                    "Chỉ báo MA, RSI, MACD, ATR và khối lượng "
                    "đều được tính từ dữ liệu nến ngày."
                )

                with st.expander(
                    "Xem 10 phiên gần nhất"
                ):

                    st.dataframe(
                        history.tail(10),
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
        "Bot chỉ sử dụng dữ liệu nến ngày đã hoàn tất. "
        "Không truy vấn bảng giá KBS trong phiên."
    )

    st.warning(
        "Nếu quét trước 15:30, nến của ngày hiện tại "
        "sẽ được bỏ qua để tránh sử dụng dữ liệu chưa hoàn tất. "
        "Sau 15:30, bot sử dụng nến mới nhất nếu nguồn đã cập nhật."
    )

    if st.button(
        "QUÉT TOÀN BỘ HOSE + HNX",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)
        status_box = st.empty()

        try:

            # Lấy danh sách mã.

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

            # Quét dữ liệu nến ngày.

            def update_progress(done, total):

                progress.progress(
                    done / total if total else 1.0
                )

                status_box.write(
                    f"Đang phân tích {done}/{total} mã..."
                )

            with st.spinner(
                "Đang phân tích dữ liệu nến ngày của toàn thị trường..."
            ):

                output = scan_market(
                    capital=capital,
                    risk_percent=risk_pct,
                    max_symbols=len(symbols),
                    progress_callback=update_progress
                )

            # Lưu kết quả.

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

    # Hiển thị kết quả đã lưu.

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
            "Lỗi dữ liệu",
            output.get("error_count", 0)
        )

        st.caption(
            "Số mã bị loại do thanh khoản hoặc điều kiện: "
            f"{output.get('filtered_count', 0)}"
        )

        # Bảng BUY.

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
                "rsi",
                "volume_ratio",
                "avg_turnover_20d",
                "median_turnover_20d",
                "quantity",
                "capital_used",
                "risk_money",
                "data_date"
            ]

            display = display[
                [
                    c for c in columns
                    if c in display.columns
                ]
            ]

            display = display.rename(columns={
                "symbol": "Mã",
                "score": "Điểm",
                "price": "Giá đóng cửa",
                "stop_loss": "Cắt lỗ",
                "target_price": "Mục tiêu",
                "reward_risk": "LN/RR",
                "rsi": "RSI",
                "volume_ratio": "KL/TB20",
                "avg_turnover_20d": "GTGD TB20 (đồng/ngày)",
                "median_turnover_20d": "GTGD trung vị 20 phiên",
                "quantity": "Số lượng",
                "capital_used": "Vốn sử dụng",
                "risk_money": "Rủi ro dự kiến",
                "data_date": "Ngày dữ liệu"
            })

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True
            )

            csv_data = display.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "TẢI DANH SÁCH TÍN HIỆU BUY",
                data=csv_data,
                file_name="co_phieu_tin_hieu_buy.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "Không có mã nào đạt đủ điều kiện BUY. "
                "Không có tín hiệu cũng là kết quả hợp lệ."
            )

        # Lỗi dữ liệu.

        errors = output.get("errors", [])

        st.caption(
            f"Số mã lỗi dữ liệu: {len(errors)}"
        )

        if errors:

            error_df = pd.DataFrame(errors)

            with st.expander(
                f"Xem chi tiết lỗi dữ liệu ({len(errors)} mã)"
            ):

                st.write(
                    "Hiển thị tối đa 20 lỗi đầu tiên."
                )

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
    + datetime.now(VN_TZ).strftime(
        "%H:%M:%S %d/%m/%Y"
    )
)

st.caption(
    "Giá mục tiêu được tính theo tỷ lệ lợi nhuận/rủi ro "
    "sau chi phí, không phải dự báo chắc chắn. "
    "Ứng dụng không tự động đặt lệnh mua/bán."
)

st.caption(
    "Nguồn dữ liệu lịch sử nến ngày: DNSE/Vietcap "
    "thông qua vn-market-data. "
    "Bot không sử dụng bảng giá trực tiếp."
)
