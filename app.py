import streamlit as st
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock, scan_market
from data_provider import (
    fetch_stock_history,
    fetch_stock_hourly,
    fetch_exchange_symbols,
)

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


st.title("📈 AI STOCK SCANNER")
st.caption("Sàng lọc cổ phiếu HOSE/HNX cho chiến lược ngắn hạn 2–5 phiên.")

st.info(
    "Bot chỉ phân tích và đưa ra tín hiệu tham khảo, không tự động đặt lệnh. "
    "Không có tín hiệu cũng là kết quả hợp lệ."
)

st.caption(
    "BUY xác nhận: điểm từ 78, RSI 45–70, H1 tăng và RR sau chi phí ≥ 2:1."
)
st.caption(
    "EARLY BUY: điểm từ 70, chỉ giải ngân thăm dò tối đa 25%; "
    "chỉ xem xét mua thêm khi breakout được xác nhận."
)
st.caption(
    "Thanh khoản tối thiểu: GTGD bình quân 20 phiên 5 tỷ/ngày "
    "và trung vị 3 tỷ/ngày."
)

tab1, tab2 = st.tabs([
    "Phân tích một mã",
    "Quét toàn thị trường",
])


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
        st.warning("🟡 EARLY BUY – MUA THĂM DÒ")
    elif signal == "FILTERED":
        st.warning("⚪ MÃ BỊ LOẠI KHỎI DANH SÁCH")
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
        "initial_quantity",
        "add_quantity",
        "breakout_trigger",
        "quantity",
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

    return df.rename(columns={
        "signal": "Tín hiệu",
        "symbol": "Mã",
        "score": "Điểm tín hiệu",
        "confirmed_score": "Điểm BUY xác nhận",
        "early_score": "Điểm EARLY BUY",
        "price": "Giá phân tích",
        "initial_quantity": "SL mua thăm dò",
        "add_quantity": "SL mua thêm tham khảo",
        "breakout_trigger": "Giá kích hoạt breakout",
        "quantity": "Tổng SL theo kế hoạch",
        "stop_loss": "Cắt lỗ",
        "target_price": "Mục tiêu",
        "reward_risk": "RR",
        "rsi": "RSI",
        "volume_ratio": "KL/TB20",
        "h1_ema9": "EMA9 H1",
        "h1_ema20": "EMA20 H1",
        "h1_status": "Trạng thái H1",
        "avg_turnover_20d": "GTGD TB20",
        "median_turnover_20d": "GTGD trung vị 20 phiên",
        "data_date": "Ngày dữ liệu",
        "reason": "Nhận định",
    })


def render_risk_plan(result):
    st.subheader("Kế hoạch quản trị rủi ro")

    c1, c2, c3 = st.columns(3)
    c1.metric("Giá phân tích", fmt_price(result.get("price")))
    c2.metric("Cắt lỗ", fmt_price(result.get("stop_loss")))
    c3.metric("Mục tiêu", fmt_price(result.get("target_price")))

    c1, c2, c3 = st.columns(3)
    c1.metric("RR sau chi phí", fmt_num(result.get("reward_risk")) + ":1")
    c2.metric("Số lượng theo kế hoạch", fmt_num(result.get("quantity"), 0))
    c3.metric("Vốn dự kiến", fmt_price(result.get("capital_used")))

    c1, c2 = st.columns(2)
    c1.metric("Rủi ro dự kiến", fmt_price(result.get("risk_money")))
    c2.metric("Ngân sách rủi ro", fmt_price(result.get("risk_budget")))


def render_early_plan(result):
    st.subheader("Kế hoạch giải ngân từng phần")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Mua thăm dò",
        fmt_num(result.get("initial_quantity"), 0) + " cp",
    )
    c2.metric(
        "Mua thêm tham khảo",
        fmt_num(result.get("add_quantity"), 0) + " cp",
    )
    c3.metric(
        "Giá kích hoạt breakout",
        fmt_price(result.get("breakout_trigger")),
    )

    c1, c2 = st.columns(2)
    c1.metric("Vốn mua thăm dò", fmt_price(result.get("initial_capital")))
    c2.metric("Rủi ro phần thăm dò", fmt_price(result.get("initial_risk")))

    st.warning(
        "EARLY BUY không phải khuyến nghị mua toàn bộ vị thế. "
        "Chỉ xem xét mua thăm dò tối đa 25% số lượng dự kiến."
    )

    st.write("**Điều kiện xem xét mua thêm:**")
    st.write(result.get("add_condition", ""))

    st.write(
        "Không mua thêm chỉ vì giá tăng. Cần có nến ngày hoàn tất "
        "đóng cửa trên giá kích hoạt, khối lượng xác nhận và H1 tích cực. "
        "Trước khi mua thêm phải tính lại RR, giá vốn và ngân sách rủi ro."
    )


# =====================================================
# TAB 1: PHÂN TÍCH MỘT MÃ
# =====================================================

with tab1:
    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10,
    ).strip().upper()

    capital = st.number_input(
        "Vốn dự kiến (đồng)",
        min_value=1_000_000,
        max_value=10_000_000_000,
        value=10_000_000,
        step=1_000_000,
    )

    risk_percent = st.number_input(
        "Ngân sách rủi ro (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
    ) / 100

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
                        capital=capital,
                        risk_percent=risk_percent,
                        symbol=symbol,
                        hourly_df=hourly,
                        hourly_error=hourly_error,
                    )

                st.subheader(f"Kết quả phân tích: {symbol}")
                st.caption(
                    "Ngày dữ liệu D1: "
                    + pd.Timestamp(history.iloc[-1]["Date"]).strftime(
                        "%d/%m/%Y"
                    )
                )

                render_signal(result.get("signal", "WAIT"))
                st.write(result.get("reason", ""))

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

                if result.get("signal") in ("BUY", "EARLY_BUY"):
                    render_risk_plan(result)

                if result.get("signal") == "EARLY_BUY":
                    render_early_plan(result)

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

                st.subheader("Xác nhận xu hướng H1")
                c1, c2, c3 = st.columns(3)
                c1.metric("EMA9 H1", fmt_price(result.get("h1_ema9")))
                c2.metric("EMA20 H1", fmt_price(result.get("h1_ema20")))
                c3.metric(
                    "Trạng thái",
                    "Đạt" if result.get("h1_confirmed") else "Chưa đạt",
                )

                st.write(result.get("h1_status", "Chưa có thông tin H1."))

                if hourly is not None:
                    st.success(
                        "Đã lấy được H1. Nguồn: "
                        + str(hourly.attrs.get("source", "Không xác định"))
                    )
                elif hourly_error:
                    st.warning(
                        "Không lấy được dữ liệu H1. Bot không xác nhận "
                        "BUY/EARLY BUY nếu thiếu H1. Chi tiết: "
                        + hourly_error
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


# =====================================================
# TAB 2: QUÉT TOÀN THỊ TRƯỜNG
# =====================================================

with tab2:
    st.subheader("Quét toàn bộ mã cổ phiếu HOSE và HNX")

    capital_scan = st.number_input(
        "Vốn quét thị trường (đồng)",
        min_value=1_000_000,
        max_value=10_000_000_000,
        value=10_000_000,
        step=1_000_000,
        key="capital_scan",
    )

    risk_scan = st.number_input(
        "Rủi ro tối đa mỗi giao dịch (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        key="risk_scan",
    ) / 100

    if st.button(
        "LẤY DANH SÁCH MÃ HOSE + HNX",
        use_container_width=True,
        key="get_symbols",
    ):
        try:
            symbols = fetch_exchange_symbols(force_refresh=True)
            stats = fetch_exchange_symbols.last_stats

            st.session_state["symbol_list"] = symbols
            st.session_state["symbol_stats"] = stats

        except Exception as exc:
            st.error(f"Không lấy được danh sách mã: {exc}")

    symbols = st.session_state.get("symbol_list", [])
    stats = st.session_state.get("symbol_stats", {})

    if symbols:
        st.write("**Thống kê danh sách mã:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HOSE", stats.get("hose_count", "—"))
        c2.metric("HNX", stats.get("hnx_count", "—"))
        c3.metric("Sau loại trùng", len(symbols))
        c4.metric("Mã trùng", stats.get("duplicate_count", "—"))

        st.caption(
            "Nguồn danh sách: "
            + str(stats.get("source", "KBS Listing"))
        )

        if len(symbols) < 600:
            st.warning(
                "Danh sách hiện có dưới 600 mã. Cần kiểm tra nguồn Listing "
                "KBS và phạm vi mã được trả về; không nên mặc định đây là "
                "toàn bộ thị trường."
            )

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
                status_box.write(
                    f"Đang phân tích {done}/{total} mã..."
                )

            try:
                with st.spinner(
                    "Đang lấy dữ liệu D1, H1 và phân tích từng mã..."
                ):
                    output = scan_market(
                        capital=capital_scan,
                        risk_percent=risk_scan,
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

    output = st.session_state.get("market_scan_output")
    meta = st.session_state.get("market_scan_meta")

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
        c2.metric("Có dữ liệu H1", output.get("h1_available_count", 0))
        c3.metric("Thiếu/lỗi H1", output.get("h1_unavailable_count", 0))

        st.metric("Lỗi D1/truy vấn", output.get("error_count", 0))

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

            st.download_button(
                "TẢI TOÀN BỘ TÍN HIỆU",
                data=display.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="tat_ca_tin_hieu.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "Không có mã nào đạt điều kiện BUY hoặc EARLY BUY. "
                "Không nên hạ tiêu chuẩn chỉ để tạo thêm tín hiệu."
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
    "Dữ liệu D1: vn-market-data. Dữ liệu H1: Vnstock KBS, "
    "dự phòng VCI. Không tự động đặt lệnh."
)
