
import streamlit as st
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from scanner import analyze_stock, scan_market
from data_provider import fetch_exchange_symbols
from intraday import fetch_intraday_5m, analyze_intraday_5m


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide",
)

# =====================================================
# SESSION STATE
# =====================================================

DEFAULTS = {
    "market_scan_output": None,
    "market_scan_meta": None,
    "symbol_list": [],
    "symbol_stats": {},
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# CONFIG
# =====================================================

st.title("📈 AI STOCK SCANNER")

st.caption(
    "AI phân tích cổ phiếu HOSE và HNX – "
    "chiến lược lướt sóng 2–5 phiên."
)

st.info(
    "Phân tích mã đơn lẻ: nến 5 phút. "
    "Quét toàn thị trường: dữ liệu D1."
)

st.caption(
    "Hệ thống chỉ phân tích và đưa ra tín hiệu tham khảo, "
    "không tự động đặt lệnh."
)


# =====================================================
# HELPERS
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
    return f"{float(value) / 1e9:.2f} tỷ"


def render_signal(signal):
    if signal == "BUY":
        st.success("🟢 BUY XÁC NHẬN")
    elif signal == "EARLY_BUY":
        st.warning("🟡 EARLY BUY")
    elif signal == "NEAR_BUY":
        st.info("🔵 NEAR BUY – GẦN ĐẠT ĐIỀU KIỆN")
    elif signal == "FILTERED":
        st.warning("⚪ FILTERED")
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

    return df.rename(columns={
        "signal": "Tín hiệu",
        "symbol": "Mã",
        "score": "Điểm tín hiệu",
        "confirmed_score": "Điểm BUY",
        "early_score": "Điểm EARLY",
        "price": "Giá phân tích",
        "stop_loss": "Cắt lỗ",
        "target_price": "Mục tiêu",
        "reward_risk": "RR",
        "rsi": "RSI",
        "volume_ratio": "KL/TB20",
        "avg_turnover_20d": "GTGD TB20",
        "median_turnover_20d": "GTGD trung vị",
        "data_date": "Ngày dữ liệu",
        "reason": "Nhận định",
    })


# =====================================================
# POSITION SIZING
# =====================================================

def calculate_position(price, stop, capital, risk_pct):
    if not all(valid_number(x) for x in [price, stop, capital]):
        return None

    price = float(price)
    stop = float(stop)
    capital = float(capital)

    if price <= 0 or stop <= 0 or stop >= price:
        return None

    risk_budget = capital * risk_pct / 100

    buy_cost_per_share = price * (1 + 0.0015 + 0.002)

    stop_proceeds_per_share = stop * (
        1 - 0.0015 - 0.001 - 0.002
    )

    risk_per_share = (
        buy_cost_per_share - stop_proceeds_per_share
    )

    if risk_per_share <= 0:
        return None

    shares_by_risk = int(risk_budget / risk_per_share)

    # Cổ phiếu HOSE/HNX giao dịch theo lô 100
    shares_by_risk = (shares_by_risk // 100) * 100

    shares_by_capital = int(
        capital / buy_cost_per_share
    )

    shares_by_capital = (
        shares_by_capital // 100
    ) * 100

    shares = min(shares_by_risk, shares_by_capital)

    return {
        "shares": shares,
        "risk_budget": risk_budget,
        "risk_per_share": risk_per_share,
        "estimated_value": shares * buy_cost_per_share,
        "estimated_loss": shares * risk_per_share,
    }


# =====================================================
# INTRADAY ANALYSIS
# =====================================================

def render_intraday_analysis(symbol):

    st.subheader("Phân tích cổ phiếu theo thời gian thực")

    st.caption(
        "Khung 5 phút | Nguồn KBS | "
        "Tự cập nhật mỗi 45 giây."
    )

    st.warning(
        "Tín hiệu 5 phút không thay thế xác nhận xu hướng D1."
    )

    try:
        with st.spinner(f"Đang lấy dữ liệu {symbol}..."):

            candles = fetch_intraday_5m(
                symbol,
                force_refresh=True,
            )

            result = analyze_intraday_5m(
                candles,
                symbol=symbol,
            )

        st.divider()
        st.subheader(f"Kết quả: {symbol}")

        if result.get("timestamp") is not None:
            st.caption(
                "Nến gần nhất: "
                + pd.Timestamp(
                    result["timestamp"]
                ).strftime("%H:%M %d/%m/%Y")
            )

        st.caption(
            "Nguồn dữ liệu: "
            + str(candles.attrs.get("source", "KBS"))
        )

        render_intraday_signal = result["signal"]

        if render_intraday_signal == "TÍN HIỆU TĂNG":
            st.success("🟢 TÍN HIỆU TĂNG")
        elif render_intraday_signal == "THEO DÕI":
            st.warning("🟡 THEO DÕI")
        else:
            st.info("⚪ CHỜ")

        st.write(result["explanation"])

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Giá", fmt_price(result["price"]))
        c2.metric("Điểm", f'{result["score"]}/100')
        c3.metric("RSI", fmt_num(result["rsi"]))
        c4.metric("KL/TB20", fmt_num(result["volume_ratio"]))

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("EMA9", fmt_price(result["ema9"]))
        c2.metric("EMA20", fmt_price(result["ema20"]))
        c3.metric("VWAP", fmt_price(result["vwap"]))
        c4.metric("MACD", fmt_num(result["macd"], 4))

        st.subheader("Vùng giá tham khảo")

        c1, c2, c3 = st.columns(3)

        c1.metric("Hỗ trợ", fmt_price(result["support"]))
        c2.metric("Kháng cự", fmt_price(result["resistance"]))
        c3.metric("Cắt lỗ", fmt_price(result["stop_reference"]))

        st.subheader("Biểu đồ 5 phút")

        chart_data = result["candles"].copy()
        chart_data = chart_data.set_index("Date")

        chart_columns = [
            c for c in ["Close", "EMA9", "EMA20", "VWAP"]
            if c in chart_data.columns
        ]

        st.line_chart(
            chart_data[chart_columns],
            height=400,
        )

        if result.get("reasons"):
            st.subheader("Yếu tố tích cực")
            for item in result["reasons"]:
                st.write("✅ " + item)

        if result.get("blockers"):
            st.subheader("Cảnh báo")
            for item in result["blockers"]:
                st.write("⚠️ " + item)

        with st.expander("Dữ liệu 30 nến gần nhất"):
            display_columns = [
                "Open", "High", "Low", "Close",
                "Volume", "EMA9", "EMA20",
                "RSI14", "VWAP", "MACD",
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
        st.error(f"Không phân tích được {symbol}: {exc}")

        st.caption(
            "Ngoài phiên có thể không có nến mới. "
            "Trong phiên, nếu dữ liệu trễ hoặc không đầy đủ, "
            "bot sẽ không đưa ra kết luận đáng tin cậy."
        )


@st.fragment(run_every="45s")
def intraday_panel(symbol):
    if not symbol:
        st.warning("Vui lòng nhập mã cổ phiếu.")
        return

    render_intraday_analysis(symbol)


# =====================================================
# MARKET RESULT
# =====================================================

def render_market_results(output, meta):

    if output is None or meta is None:
        return

    st.divider()

    st.caption(
        f"Kết quả lúc {meta['completed_at']} – "
        f"{meta['symbols_total']} mã."
    )

    st.subheader("Tổng kết quét")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("BUY", output.get("confirmed_buy_count", 0))
    c2.metric("EARLY BUY", output.get("early_buy_count", 0))
    c3.metric("NEAR BUY", output.get("near_buy_count", 0))
    c4.metric("WAIT", output.get("wait_count", 0))
    c5.metric("FILTERED", output.get("filtered_count", 0))

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Lọc thanh khoản",
        output.get("liquidity_filtered_count", 0),
    )

    c2.metric(
        "Thiếu lịch sử",
        output.get("insufficient_history_count", 0),
    )

    c3.metric(
        "Lỗi dữ liệu",
        output.get("d1_error_count", 0),
    )

    results = output.get("results", [])

    if not results:
        st.info("Không có kết quả.")
        return

    st.subheader("Xếp hạng cơ hội")

    display = make_results_dataframe(results)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    groups = [
        ("BUY", "🟢 BUY xác nhận"),
        ("EARLY_BUY", "🟡 EARLY BUY"),
        ("NEAR_BUY", "🔵 NEAR BUY"),
        ("WAIT", "⚪ WAIT"),
    ]

    for signal, title in groups:

        items = [
            x for x in results
            if x.get("signal") == signal
        ]

        if not items:
            continue

        with st.expander(f"{title} ({len(items)} mã)"):

            for item in items:

                st.markdown(
                    f"### {item.get('symbol', '')} – "
                    f"{item.get('score', 0)} điểm"
                )

                st.write(item.get("reason", ""))

                if item.get("reasons"):
                    st.write("**Yếu tố đạt:**")
                    for reason in item["reasons"]:
                        st.write("✅ " + reason)

                if item.get("blockers"):
                    st.write("**Điều kiện chưa đạt:**")
                    for blocker in item["blockers"]:
                        st.write("⚠️ " + blocker)

                if item.get("signal") in ["BUY", "EARLY_BUY"]:

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Giá",
                        fmt_price(item.get("price")),
                    )

                    c2.metric(
                        "Cắt lỗ",
                        fmt_price(item.get("stop_loss")),
                    )

                    c3.metric(
                        "Mục tiêu",
                        fmt_price(item.get("target_price")),
                    )

                    st.metric(
                        "RR",
                        fmt_num(item.get("reward_risk")),
                    )

                st.divider()

    st.subheader("Quản trị vốn và khối lượng")

    capital = st.number_input(
        "Vốn giao dịch (đồng)",
        min_value=1_000_000,
        max_value=10_000_000_000,
        value=10_000_000,
        step=1_000_000,
        key="position_capital",
    )

    risk_pct = st.number_input(
        "Rủi ro tối đa mỗi lệnh (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        key="position_risk",
    )

    candidates = [
        x for x in results
        if x.get("signal") in ["BUY", "EARLY_BUY"]
        and valid_number(x.get("stop_loss"))
    ]

    if candidates:

        position_rows = []

        for item in candidates:

            plan = calculate_position(
                item.get("price"),
                item.get("stop_loss"),
                capital,
                risk_pct,
            )

            if plan is None:
                continue

            position_rows.append({
                "Mã": item["symbol"],
                "Giá tham chiếu": item["price"],
                "Cắt lỗ": item["stop_loss"],
                "Số lượng tham khảo": plan["shares"],
                "Giá trị mua ước tính": round(plan["estimated_value"]),
                "Rủi ro ước tính": round(plan["estimated_loss"]),
            })

        if position_rows:
            st.dataframe(
                pd.DataFrame(position_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Đây là số lượng tối đa theo rủi ro và vốn. "
                "Không có nghĩa là nên mua tất cả các mã cùng lúc."
            )
    else:
        st.info(
            "Chưa có mã BUY/EARLY BUY đủ dữ liệu để tính khối lượng."
        )

    st.subheader("Tải kết quả")

    st.download_button(
        "TẢI TOÀN BỘ KẾT QUẢ",
        data=display.to_csv(index=False).encode("utf-8-sig"),
        file_name="ket_qua_quet_thi_truong.csv",
        mime="text/csv",
        use_container_width=True,
    )

    errors = output.get("errors", [])

    if errors:
        with st.expander(f"Lỗi dữ liệu ({len(errors)} mã)"):

            error_df = pd.DataFrame(errors)

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
# TABS
# =====================================================

tab1, tab2 = st.tabs([
    "Phân tích một mã – 5 phút",
    "Quét toàn thị trường – D1",
])


# =====================================================
# TAB 1
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
# TAB 2
# =====================================================

with tab2:

    st.subheader("Quét HOSE và HNX")

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
                f"Đã lấy được {len(symbols)} mã."
            )

        except Exception as exc:
            st.error(f"Không lấy được danh sách mã: {exc}")

    symbols = st.session_state.symbol_list
    stats = st.session_state.symbol_stats

    if symbols:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("HOSE", stats.get("hose_count", "—"))
        c2.metric("HNX", stats.get("hnx_count", "—"))
        c3.metric("Tổng mã", len(symbols))
        c4.metric("Mã trùng", stats.get("duplicate_count", "—"))

        st.caption(
            "Nguồn: " + str(stats.get("source", "Không rõ"))
        )

        with st.expander("Danh sách mã"):
            st.write(", ".join(symbols))

        if st.button(
            "QUÉT TOÀN BỘ THỊ TRƯỜNG",
            type="primary",
            use_container_width=True,
            key="scan_all_market",
        ):

            progress = st.progress(0)
            status = st.empty()

            def update_progress(done, total):
                progress.progress(
                    done / total if total else 1.0
                )
                status.write(
                    f"Đang phân tích {done}/{total} mã..."
                )

            try:
                with st.spinner("Đang phân tích D1..."):

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
                status.success("Đã hoàn tất quét.")

            except Exception as exc:
                st.error(f"Lỗi quét thị trường: {exc}")

    render_market_results(
        st.session_state.market_scan_output,
        st.session_state.market_scan_meta,
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Thời gian hệ thống: "
    + datetime.now(VN_TZ).strftime("%H:%M:%S %d/%m/%Y")
)

st.caption(
    "D1: vn-market-data | Intraday: KBS | "
    "Không tự động đặt lệnh."
)
