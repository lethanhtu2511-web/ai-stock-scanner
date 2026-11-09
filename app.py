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
    fetch_stock_quote,
    fetch_exchange_symbols,
    get_market_status,
    fetch_market_quotes
)


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)


# Lưu kết quả trong phiên Streamlit.
if "market_scan_output" not in st.session_state:
    st.session_state["market_scan_output"] = None

if "market_scan_meta" not in st.session_state:
    st.session_state["market_scan_meta"] = None


st.title("📈 AI STOCK SCANNER")

st.caption(
    "Sàng lọc cổ phiếu HOSE/HNX cho chiến lược ngắn hạn 2–5 phiên."
)


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
        "Lịch sử nến ngày: DNSE/Vietcap. "
        "Bảng giá trong phiên: VNStock KBS."
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
                    "Đang kiểm tra phiên và lấy dữ liệu..."
                ):

                    market_status = get_market_status()

                    history = fetch_stock_history(
                        symbol,
                        length=260
                    )

                    live_quote = None

                    if market_status["is_open"]:

                        quotes = fetch_market_quotes(
                            [symbol]
                        )

                        live_quote = quotes.get(symbol)

                        if live_quote is None:

                            raise RuntimeError(
                                "Đang trong khung giờ giao dịch "
                                "nhưng không lấy được giá bảng điện."
                            )

                        result = analyze_stock(
                            history,
                            capital,
                            risk_pct,
                            live_quote=live_quote,
                            symbol=symbol
                        )

                        price_label = "Giá bảng điện"
                        current_price = live_quote["last_price"]

                    else:

                        daily_quote = fetch_stock_quote(
                            symbol
                        )

                        result = analyze_stock(
                            history,
                            capital,
                            risk_pct,
                            symbol=symbol
                        )

                        price_label = "Giá đóng cửa nến ngày"
                        current_price = daily_quote["last_price"]

                st.subheader(
                    f"Kết quả: {symbol}"
                )

                if market_status["is_open"]:

                    st.success(
                        "Đã truy vấn bảng giá trong khung giờ giao dịch."
                    )

                    st.caption(
                        f"Nguồn: VNStock KBS | "
                        f"Thời điểm truy vấn: "
                        f"{market_status['checked_at']}"
                    )

                    st.warning(
                        "Bảng giá KBS không cung cấp timestamp riêng "
                        "cho từng mã. Thời điểm truy vấn không đồng nghĩa "
                        "với thời điểm khớp lệnh của mã này."
                    )

                else:

                    st.info(
                        "Ngoài khung giờ giao dịch. "
                        "Đang sử dụng nến ngày gần nhất."
                    )

                    st.caption(
                        f"Trạng thái: {market_status['status']}"
                    )

                if result.get("signal") == "BUY":

                    st.success("🟢 TÍN HIỆU BUY")

                elif result.get("signal") == "FILTERED":

                    st.warning("⚪ MÃ BỊ LOẠI KHỎI DANH SÁCH")

                else:

                    st.warning("🟡 TÍN HIỆU WAIT")

                st.write(
                    result.get("reason", "")
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    price_label,
                    f"{current_price:,.0f} đ"
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

                if result.get("avg_turnover_20d") is not None:

                    st.metric(
                        "GTGD bình quân 20 phiên",
                        f"{result['avg_turnover_20d'] / 1e9:.2f} tỷ đồng/ngày"
                    )

                if result.get("median_turnover_20d") is not None:

                    st.metric(
                        "GTGD trung vị 20 phiên",
                        f"{result['median_turnover_20d'] / 1e9:.2f} tỷ đồng/ngày"
                    )

                st.subheader("Yếu tố kỹ thuật")

                for reason in result.get("reasons", []):

                    st.write("• " + reason)

                st.caption(
                    "RSI, MACD, MA, ATR và khối lượng trung bình "
                    "được tính từ nến ngày. Trong khung giờ giao dịch, "
                    "giá bảng điện được dùng để đánh giá mức giá hiện tại."
                )

                with st.expander("10 phiên gần nhất"):

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
        "Bot tự kiểm tra khung giờ và lịch nghỉ đã cấu hình. "
        "Trong khung giờ giao dịch sẽ truy vấn bảng giá KBS; "
        "ngoài phiên sẽ sử dụng nến ngày gần nhất."
    )

    st.warning(
        "Lịch nghỉ đã cấu hình cho năm 2026. "
        "Bảng giá KBS không có timestamp riêng từng mã; "
        "thời điểm truy vấn không phải thời điểm khớp lệnh riêng."
    )

    if st.button(
        "QUÉT TOÀN BỘ HOSE + HNX",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)
        status_box = st.empty()

        try:

            # BƯỚC 1: Kiểm tra phiên giao dịch.

            with st.spinner(
                "Đang kiểm tra phiên giao dịch..."
            ):

                market_status = get_market_status()

            st.write(
                f"**Trạng thái:** {market_status['status']}"
            )

            st.caption(
                f"Thời điểm kiểm tra: "
                f"{market_status['checked_at']}"
            )

            st.caption(
                market_status.get("calendar_note", "")
            )

            live_mode = market_status["is_open"]

            # BƯỚC 2: Lấy danh sách mã.

            with st.spinner(
                "Đang lấy danh sách cổ phiếu HOSE và HNX..."
            ):

                symbols = fetch_exchange_symbols()

            st.write(
                f"**Danh sách cổ phiếu:** {len(symbols)} mã."
            )

            if not symbols:

                raise RuntimeError(
                    "Danh sách cổ phiếu đang rỗng."
                )

            live_quotes = None
            quote_errors = []

            # BƯỚC 3: Lấy bảng giá nếu đang trong phiên.

            if live_mode:

                st.info(
                    "Đang truy vấn bảng giá KBS cho toàn bộ danh sách."
                )

                with st.spinner(
                    "Đang lấy bảng giá..."
                ):

                    live_quotes = fetch_market_quotes(
                        symbols
                    )

                missing_symbols = sorted(
                    set(symbols) - set(live_quotes)
                )

                st.success(
                    f"Nhận được giá cho "
                    f"{len(live_quotes)}/{len(symbols)} mã."
                )

                for missing_symbol in missing_symbols:

                    quote_errors.append({
                        "symbol": missing_symbol,
                        "error": (
                            "Không nhận được giá bảng điện KBS. "
                            "Không dùng giá nến ngày thay thế."
                        )
                    })

                if missing_symbols:

                    st.warning(
                        f"Có {len(missing_symbols)} mã không nhận được giá. "
                        "Các mã này sẽ không được phân tích trong phiên."
                    )

            else:

                st.info(
                    "Ngoài khung giờ giao dịch: sử dụng nến ngày gần nhất."
                )

            # BƯỚC 4: Quét cổ phiếu.

            def update_progress(done, total):

                progress.progress(
                    done / total if total else 1.0
                )

                status_box.write(
                    f"Đang quét {done}/{total} mã..."
                )

            output = scan_market(
                capital=capital,
                risk_percent=risk_pct,
                max_symbols=len(symbols),
                progress_callback=update_progress,
                live_mode=live_mode,
                live_quotes=live_quotes
            )

            # Gộp lỗi giá và lỗi dữ liệu lịch sử.

            all_errors = (
                quote_errors
                + output.get("errors", [])
            )

            unique_errors = []
            seen_errors = set()

            for item in all_errors:

                key = (
                    item.get("symbol"),
                    item.get("error")
                )

                if key not in seen_errors:

                    seen_errors.add(key)
                    unique_errors.append(item)

            output["errors"] = unique_errors
            output["error_count"] = len(unique_errors)

            output["quote_received"] = (
                len(live_quotes)
                if live_mode
                else None
            )

            output["quote_total"] = (
                len(symbols)
                if live_mode
                else None
            )

            # LƯU KẾT QUẢ VÀO SESSION STATE.

            st.session_state["market_scan_output"] = output

            st.session_state["market_scan_meta"] = {
                "live_mode": live_mode,
                "market_status": market_status,
                "symbols_total": len(symbols),
                "completed_at": datetime.now(
                    VN_TZ
                ).strftime("%H:%M:%S %d/%m/%Y")
            }

        except Exception as exc:

            st.error(
                f"Không thể quét toàn thị trường: {exc}"
            )

    # HIỂN THỊ KẾT QUẢ ĐÃ LƯU.

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

        if meta["live_mode"]:

            st.caption(
                "Giá đánh giá: giá bảng điện KBS tại lúc truy vấn. "
                "Nguồn không cung cấp timestamp riêng từng mã; "
                "chỉ báo kỹ thuật vẫn dựa trên nến ngày."
            )

        else:

            st.caption(
                "Giá đánh giá: giá đóng cửa nến ngày gần nhất."
            )

        # Thống kê tổng quan.

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

        if output.get("filtered_count") is not None:

            st.caption(
                f"Số mã bị loại do thanh khoản hoặc điều kiện: "
                f"{output.get('filtered_count', 0)}"
            )

        # BẢNG BUY.

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
                "data_date",
                "price_mode"
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
                "price": "Giá đánh giá",
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
                "data_date": "Ngày nến",
                "price_mode": "Loại giá"
            })

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Không có mã nào đạt đủ điều kiện BUY. "
                "Không có tín hiệu cũng là kết quả hợp lệ."
            )

        # CHI TIẾT LỖI DỮ LIỆU.

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
                    "Hiển thị tối đa 20 lỗi đầu tiên để giảm tải trang."
                )

                st.dataframe(
                    error_df.head(20),
                    use_container_width=True,
                    hide_index=True
                )

                if len(error_df) > 20:

                    st.caption(
                        f"Còn {len(error_df) - 20} lỗi khác. "
                        "Tải CSV để xem toàn bộ."
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
    "Trong phiên: bảng giá VNStock KBS. "
    "Ngoài phiên: nến ngày gần nhất. "
    "Các chỉ báo kỹ thuật và khối lượng trung bình "
    "vẫn dựa trên dữ liệu nến ngày."
)
