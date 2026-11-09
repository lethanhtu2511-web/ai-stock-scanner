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
            st.error("Vui lòng nhập mã cổ phiếu.")

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
                            live_quote=live_quote
                        )

                        price_label = "Giá khớp gần nhất"
                        current_price = live_quote["last_price"]

                    else:

                        daily_quote = fetch_stock_quote(
                            symbol
                        )

                        result = analyze_stock(
                            history,
                            capital,
                            risk_pct
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
                        f"Thời điểm truy vấn: {market_status['checked_at']}"
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

                st.subheader("Yếu tố kỹ thuật")

                for reason in result.get("reasons", []):
                    st.write("• " + reason)

                st.caption(
                    "RSI, MACD, MA, ATR và khối lượng trung bình "
                    "được tính từ nến ngày. Trong khung giờ giao dịch, "
                    "giá khớp gần nhất được dùng để đánh giá mức giá hiện tại."
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
        "Bot tự kiểm tra khung giờ giao dịch. "
        "Trong khung giờ giao dịch sẽ truy vấn bảng giá KBS; "
        "ngoài khung giờ sẽ sử dụng nến ngày gần nhất."
    )

    st.warning(
        "Hệ thống nhận diện phiên theo thứ và giờ giao dịch, "
        "chưa có lịch nghỉ lễ tự động. Giá bảng điện không có "
        "dấu thời gian riêng cho từng mã."
    )

    if st.button(
        "QUÉT TOÀN BỘ HOSE + HNX",
        type="primary",
        use_container_width=True
    ):

        progress = st.progress(0)
        status_box = st.empty()

        try:

            # BƯỚC 1: Xác định khung giờ.
            with st.spinner(
                "Đang kiểm tra khung giờ giao dịch..."
            ):
                market_status = get_market_status()

            st.write(
                f"**Trạng thái:** {market_status['status']}"
            )

            st.caption(
                f"Thời điểm kiểm tra: {market_status['checked_at']}"
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

            # BƯỚC 3: Lấy bảng giá nếu trong khung giờ giao dịch.
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

                missing_count = (
                    len(symbols) - len(live_quotes)
                )

                st.success(
                    f"Nhận được giá cho {len(live_quotes)}/"
                    f"{len(symbols)} mã."
                )

                if missing_count > 0:
                    st.warning(
                        f"Có {missing_count} mã không nhận được giá. "
                        "Các mã này sẽ được ghi nhận lỗi, không dùng giá ngày thay thế."
                    )

            else:

                st.info(
                    "Ngoài khung giờ giao dịch: sử dụng dữ liệu nến ngày."
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

            st.success(
                f"Đã xử lý {output['scanned']} mã. "
                f"Tìm thấy {output['buy_count']} mã BUY."
            )

            if live_mode:
                st.caption(
                    "Giá đánh giá: giá khớp gần nhất từ bảng giá KBS. "
                    "Chỉ báo kỹ thuật: nến ngày."
                )

            else:
                st.caption(
                    "Giá đánh giá: giá đóng cửa nến ngày gần nhất."
                )

            # BƯỚC 5: Hiển thị danh sách BUY.
            if output["results"]:

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
                    "quantity",
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
                    "quantity": "Số lượng",
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

            # BƯỚC 6: Thống kê lỗi.
            st.caption(
                f"Số mã lỗi dữ liệu: {output['error_count']}"
            )

            if output["errors"]:

                with st.expander(
                    "Xem chi tiết lỗi dữ liệu"
                ):

                    st.dataframe(
                        pd.DataFrame(output["errors"]),
                        use_container_width=True,
                        hide_index=True
                    )

        except Exception as exc:

            st.error(
                f"Không thể quét toàn thị trường: {exc}"
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
