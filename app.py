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


# =========================================================
# CẤU HÌNH TRANG
# =========================================================

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide",
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
    "Không tự động đặt lệnh và không bảo đảm lợi nhuận."
)

st.caption(
    "Thanh khoản tối thiểu: GTGD bình quân 20 phiên ≥ 5 tỷ/ngày "
    "và GTGD trung vị ≥ 3 tỷ/ngày."
)

st.caption(
    "BUY xác nhận: điểm kỹ thuật ≥ 78, RSI 45–70, "
    "xác nhận H1 và RR sau chi phí ≥ 2:1."
)

st.caption(
    "EARLY BUY: điểm mua sớm ≥ 70, giá ở vùng tích lũy gần đỉnh "
    "20 phiên, có xác nhận H1 và đạt điều kiện quản trị rủi ro."
)

tab1, tab2 = st.tabs([
    "Phân tích một mã",
    "Quét toàn thị trường",
])


# =========================================================
# HÀM HỖ TRỢ
# =========================================================

def is_valid_number(value):
    try:
        return (
            value is not None
            and pd.notna(value)
            and value != ""
        )
    except (TypeError, ValueError):
        return False


def format_price(value):
    if not is_valid_number(value):
        return "—"

    return f"{float(value):,.0f} đ"


def format_number(value, digits=2):
    if not is_valid_number(value):
        return "—"

    return f"{float(value):,.{digits}f}"


def format_turnover(value):
    if not is_valid_number(value):
        return "—"

    return f"{float(value) / 1e9:.2f} tỷ/ngày"


def safe_date(value):
    try:
        return pd.Timestamp(value).strftime("%d/%m/%Y")
    except Exception:
        return "Không xác định"


def render_signal(signal):
    if signal == "BUY":
        st.success("🟢 BUY XÁC NHẬN")

    elif signal == "EARLY_BUY":
        st.warning("🟡 EARLY BUY – TÍN HIỆU MUA SỚM")

    elif signal == "FILTERED":
        st.warning("⚪ MÃ BỊ LOẠI KHỎI DANH SÁCH")

    else:
        st.info("⚪ WAIT – TIẾP TỤC THEO DÕI")


def get_h1_status(result):
    available = result.get("h1_data_available", False)

    if not available:
        return (
            "Không có dữ liệu H1 hợp lệ",
            "warning",
        )

    if result.get("h1_confirmed"):
        return (
            "Xu hướng H1 tăng – đã xác nhận BUY",
            "success",
        )

    if result.get("h1_early"):
        return (
            "H1 đang cải thiện – có thể theo dõi mua sớm",
            "info",
        )

    return (
        "Có dữ liệu H1 nhưng xu hướng chưa xác nhận",
        "warning",
    )


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
        "quantity",
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

    df = df[
        [c for c in columns if c in df.columns]
    ]

    return df.rename(columns={
        "signal": "Tín hiệu",
        "symbol": "Mã",
        "score": "Điểm tín hiệu",
        "confirmed_score": "Điểm BUY xác nhận",
        "early_score": "Điểm EARLY BUY",
        "price": "Giá phân tích",
        "stop_loss": "Cắt lỗ",
        "target_price": "Mục tiêu",
        "reward_risk": "LN/RR",
        "quantity": "Số lượng",
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


def render_h1_section(result, hourly, hourly_error=None):
    st.subheader("Xác nhận xu hướng H1")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "EMA9 H1",
        format_price(result.get("h1_ema9")),
    )

    c2.metric(
        "EMA20 H1",
        format_price(result.get("h1_ema20")),
    )

    status, level = get_h1_status(result)

    c3.metric(
        "Trạng thái H1",
        status,
    )

    if level == "success":
        st.success(status)

    elif level == "warning":
        st.warning(status)

    else:
        st.info(status)

    detail = result.get("h1_status")

    if detail:
        st.caption(f"Chi tiết dữ liệu H1: {detail}")

    if hourly is not None:
        source = hourly.attrs.get(
            "source",
            "Không xác định",
        )

        st.success(
            f"Đã nhận DataFrame H1. Nguồn: {source}. "
            "Trạng thái tính EMA được xác định riêng."
        )

    elif hourly_error:
        st.warning(
            "Không lấy được dữ liệu H1. "
            f"Chi tiết: {hourly_error}"
        )


# =========================================================
# TAB 1: PHÂN TÍCH MỘT MÃ
# =========================================================

with tab1:

    symbol = st.text_input(
        "Nhập mã cổ phiếu",
        value="FPT",
        max_chars=10,
        key="single_symbol",
    ).strip().upper()

    if st.button(
        "PHÂN TÍCH CỔ PHIẾU",
        type="primary",
        use_container_width=True,
        key="analyze_one_stock",
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
                        length=260,
                    )

                    if history is None or history.empty:
                        raise RuntimeError(
                            "Không có dữ liệu D1 của cổ phiếu."
                        )

                    if len(history) < 200:
                        raise RuntimeError(
                            f"Chỉ có {len(history)} phiên D1; "
                            "yêu cầu tối thiểu 200 phiên."
                        )

                    hourly = None
                    hourly_error = None

                    try:
                        hourly = fetch_stock_hourly(symbol)

                        if (
                            hourly is None
                            or not isinstance(hourly, pd.DataFrame)
                            or hourly.empty
                        ):
                            hourly = None
                            hourly_error = (
                                "Nguồn dữ liệu H1 trả về rỗng."
                            )

                    except Exception as exc:
                        hourly_error = str(exc)

                    result = analyze_stock(
                        history,
                        symbol=symbol,
                        hourly_df=hourly,
                    )

                # -----------------------------------------
                # KẾT QUẢ CHUNG
                # -----------------------------------------

                st.subheader(
                    f"Kết quả phân tích: {symbol}"
                )

                st.info(
                    f"Dữ liệu D1 gần nhất: "
                    f"{safe_date(history.iloc[-1].get('Date'))}. "
                    "Bot sử dụng dữ liệu lịch sử, "
                    "không tự động đặt lệnh."
                )

                signal = result.get("signal", "WAIT")

                render_signal(signal)

                st.write(
                    result.get(
                        "reason",
                        result.get("message", ""),
                    )
                )

                # -----------------------------------------
                # ĐIỂM KỸ THUẬT
                # -----------------------------------------

                st.subheader("Điểm kỹ thuật")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá phân tích",
                    format_price(result.get("price")),
                )

                c2.metric(
                    "Điểm tín hiệu",
                    (
                        f"{result.get('score')}/100"
                        if is_valid_number(result.get("score"))
                        else "—"
                    ),
                )

                c3.metric(
                    "RSI",
                    format_number(result.get("rsi")),
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Điểm BUY xác nhận",
                    (
                        f"{result.get('confirmed_score')}/100"
                        if is_valid_number(
                            result.get("confirmed_score")
                        )
                        else "—"
                    ),
                )

                c2.metric(
                    "Điểm EARLY BUY",
                    (
                        f"{result.get('early_score')}/100"
                        if is_valid_number(
                            result.get("early_score")
                        )
                        else "—"
                    ),
                )

                c3.metric(
                    "Khối lượng/TB20",
                    format_number(
                        result.get("volume_ratio")
                    ),
                )

                # -----------------------------------------
                # QUẢN TRỊ RỦI RO
                # -----------------------------------------

                st.subheader(
                    "Kế hoạch quản trị rủi ro"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Giá cắt lỗ",
                    format_price(result.get("stop_loss")),
                )

                c2.metric(
                    "Giá mục tiêu",
                    format_price(result.get("target_price")),
                )

                c3.metric(
                    "Lợi nhuận/rủi ro",
                    (
                        f"{format_number(result.get('reward_risk'))}:1"
                        if is_valid_number(
                            result.get("reward_risk")
                        )
                        else "—"
                    ),
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "Số lượng cổ phiếu",
                    (
                        f"{result.get('quantity'):,}"
                        if is_valid_number(
                            result.get("quantity")
                        )
                        else "—"
                    ),
                )

                c2.metric(
                    "Vốn dự kiến sử dụng",
                    format_price(result.get("capital_used")),
                )

                # -----------------------------------------
                # THANH KHOẢN
                # -----------------------------------------

                st.subheader("Thanh khoản")

                c1, c2 = st.columns(2)

                c1.metric(
                    "GTGD bình quân 20 phiên",
                    format_turnover(
                        result.get("avg_turnover_20d")
                    ),
                )

                c2.metric(
                    "GTGD trung vị 20 phiên",
                    format_turnover(
                        result.get("median_turnover_20d")
                    ),
                )

                # -----------------------------------------
                # H1
                # -----------------------------------------

                render_h1_section(
                    result,
                    hourly,
                    hourly_error,
                )

                # -----------------------------------------
                # YẾU TỐ KỸ THUẬT
                # -----------------------------------------

                st.subheader("Các yếu tố kỹ thuật")

                reasons = result.get("reasons", [])

                if reasons:
                    for item in reasons:
                        st.write("• " + str(item))
                else:
                    st.write(
                        "Chưa có thông tin về yếu tố kỹ thuật."
                    )

                blockers = result.get("blockers", [])

                if blockers:
                    st.subheader("Điều kiện chưa đạt")

                    for item in blockers:
                        st.write("• " + str(item))

                if result.get("message"):
                    st.info(result["message"])

                # -----------------------------------------
                # DỮ LIỆU GỐC
                # -----------------------------------------

                st.caption(
                    "D1: MA20, MA50, MA200, RSI, MACD, ATR, "
                    "khối lượng, thanh khoản và vùng đỉnh "
                    "20/50 phiên. H1: EMA9 và EMA20."
                )

                with st.expander(
                    "Xem 10 phiên D1 gần nhất"
                ):
                    st.dataframe(
                        history.tail(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                if hourly is not None:
                    with st.expander(
                        "Xem 20 nến H1 gần nhất"
                    ):
                        st.dataframe(
                            hourly.tail(20),
                            use_container_width=True,
                            hide_index=True,
                        )

            except Exception as exc:
                st.error(
                    f"Không phân tích được {symbol}: {exc}"
                )


# =========================================================
# TAB 2: QUÉT TOÀN THỊ TRƯỜNG
# =========================================================

with tab2:

    st.subheader(
        "Quét toàn bộ mã cổ phiếu HOSE và HNX"
    )

    st.info(
        "Bot sử dụng dữ liệu D1 và H1 để xác nhận tín hiệu. "
        "Không tự động đặt lệnh."
    )

    st.warning(
        "BUY xác nhận và EARLY BUY được phân loại riêng. "
        "EARLY BUY có rủi ro cao hơn BUY xác nhận."
    )

    if st.button(
        "QUÉT TOÀN BỘ HOSE + HNX",
        type="primary",
        use_container_width=True,
        key="scan_all_market",
    ):

        progress = st.progress(0)
        status_box = st.empty()

        try:

            with st.spinner(
                "Đang lấy danh sách cổ phiếu..."
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
                "Đang phân tích D1, H1 và điều kiện kỹ thuật..."
            ):

                output = scan_market(
                    max_symbols=len(symbols),
                    progress_callback=update_progress,
                )

            st.session_state["market_scan_output"] = output

            st.session_state["market_scan_meta"] = {
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
                f"Không thể quét toàn thị trường: {exc}"
            )

    # ---------------------------------------------
    # HIỂN THỊ KẾT QUẢ ĐÃ QUÉT
    # ---------------------------------------------

    output = st.session_state.get(
        "market_scan_output"
    )

    meta = st.session_state.get(
        "market_scan_meta"
    )

    if output is not None and meta is not None:

        st.divider()

        st.caption(
            f"Kết quả lần quét lúc "
            f"{meta['completed_at']} "
            f"({meta['symbols_total']} mã trong danh sách)."
        )

        # Tổng quan
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Đã xử lý",
            f"{output.get('scanned', 0)} mã",
        )

        c2.metric(
            "BUY xác nhận",
            output.get("confirmed_buy_count", 0),
        )

        c3.metric(
            "EARLY BUY",
            output.get("early_buy_count", 0),
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Loại do điều kiện",
            output.get("filtered_count", 0),
        )

        c2.metric(
            "Lỗi dữ liệu/truy vấn",
            output.get("error_count", 0),
        )

        c3.metric(
            "Không có H1 hợp lệ",
            output.get("h1_unavailable_count", 0),
        )

        results = output.get("results", [])

        if results:

            display = make_results_dataframe(results)

            st.subheader("Danh sách tín hiệu")

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

            # BUY xác nhận
            if confirmed:

                st.subheader(
                    "🟢 Danh sách BUY xác nhận"
                )

                confirmed_df = make_results_dataframe(
                    confirmed
                )

                st.dataframe(
                    confirmed_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "TẢI DANH SÁCH BUY XÁC NHẬN",
                    data=confirmed_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="co_phieu_buy_xac_nhan.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # EARLY BUY
            if early:

                st.subheader(
                    "🟡 Danh sách EARLY BUY"
                )

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
                    file_name="co_phieu_early_buy.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Toàn bộ tín hiệu
            st.download_button(
                "TẢI TOÀN BỘ TÍN HIỆU",
                data=display.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="toan_bo_tin_hieu_co_phieu.csv",
                mime="text/csv",
                use_container_width=True,
            )

        else:

            st.info(
                "Không có mã nào đạt đủ điều kiện BUY "
                "hoặc EARLY BUY. Không có tín hiệu cũng "
                "là kết quả hợp lệ."
            )

        # -----------------------------------------
        # CHI TIẾT LỖI DỮ LIỆU
        # -----------------------------------------

        errors = output.get("errors", [])

        if errors:

            error_df = pd.DataFrame(errors)

            with st.expander(
                f"Xem chi tiết lỗi dữ liệu ({len(errors)} mã)"
            ):

                st.dataframe(
                    error_df.head(100),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "TẢI TOÀN BỘ LỖI DỮ LIỆU",
                    data=error_df.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="loi_du_lieu_co_phieu.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# =========================================================
# CHÂN TRANG
# =========================================================

st.divider()

st.caption(
    "Thời điểm hệ thống: "
    + datetime.now(VN_TZ).strftime(
        "%H:%M:%S %d/%m/%Y"
    )
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
