
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from scanner import scan_market
from data_provider import fetch_exchange_symbols
from intraday import fetch_intraday_5m, analyze_intraday_5m

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

st.set_page_config(
    page_title="AI STOCK SCANNER",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI STOCK SCANNER")
st.caption("HOSE/HNX | Lướt sóng 2–5 phiên | Không tự đặt lệnh")
st.info("Quét thị trường dùng D1. Phân tích mã đơn lẻ dùng nến 5 phút.")


def fmt(x, digits=0):
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def joined(x):
    if isinstance(x, list):
        return "; ".join(str(v) for v in x)
    if isinstance(x, dict):
        return "; ".join(
            f"{k}: {'Đạt' if v else 'Không đạt'}"
            for k, v in x.items()
        )
    return "" if x is None else str(x)


def export_frame(results):
    rows = []
    for r in results:
        rows.append({
            "Tín hiệu": r.get("signal"),
            "Mã": r.get("symbol"),
            "Điểm tín hiệu": r.get("score"),
            "Điểm BUY": r.get("confirmed_score"),
            "Điểm EARLY": r.get("early_score"),
            "Giá phân tích": r.get("price"),
            "Cắt lỗ": r.get("stop_loss"),
            "Mục tiêu": r.get("target_price"),
            "RR": r.get("reward_risk"),
            "RSI": r.get("rsi"),
            "KL/TB20": r.get("volume_ratio"),
            "ATR %": r.get("atr_pct"),
            "GTGD TB20": r.get("avg_turnover_20d"),
            "GTGD trung vị": r.get("median_turnover_20d"),
            "Ngày dữ liệu": r.get("data_date"),
            "Nhận định": r.get("reason"),
            "Điều kiện thiếu BUY": joined(r.get("failed_buy")),
            "Điều kiện thiếu EARLY": joined(r.get("failed_early")),
            "Điều kiện thiếu NEAR": joined(r.get("failed_near")),
            "Lý do bị chặn": joined(r.get("blockers")),
            "Điểm phạt": joined(r.get("penalties")),
            "Điểm tích cực": joined(r.get("reasons")),
            "Kiểm tra BUY": joined(r.get("buy_conditions")),
            "Kiểm tra EARLY": joined(r.get("early_conditions")),
            "Kiểm tra NEAR": joined(r.get("near_conditions")),
            "Lý do RR": r.get("risk_note"),
            "Cách MA20 (%)": r.get("extension_pct"),
            "Giá/đỉnh 20 phiên (%)": r.get("proximity_pct")
        })
    return pd.DataFrame(rows)


def position_size(price, stop, capital, risk_pct):
    try:
        price = float(price)
        stop = float(stop)
        if price <= 0 or stop <= 0 or stop >= price:
            return None

        cost = price * (1 + 0.0015 + 0.002)
        stop_net = stop * (1 - 0.0015 - 0.001 - 0.002)
        risk_share = cost - stop_net

        if risk_share <= 0:
            return None

        by_risk = int(capital * risk_pct / 100 / risk_share)
        by_cash = int(capital / cost)
        shares = (min(by_risk, by_cash) // 100) * 100

        return {
            "shares": shares,
            "cost": round(shares * cost),
            "loss": round(shares * risk_share)
        }
    except (TypeError, ValueError):
        return None


def render_detail(r):
    signal = r.get("signal", "WAIT")

    if signal == "BUY":
        st.success("BUY xác nhận D1")
    elif signal == "EARLY_BUY":
        st.warning("EARLY BUY - Chưa xác nhận")
    elif signal == "NEAR_BUY":
        st.info("NEAR BUY - Chỉ theo dõi")
    else:
        st.info(signal)

    st.write(r.get("reason", ""))

    a, b, c, d = st.columns(4)
    a.metric("Giá", fmt(r.get("price")))
    b.metric("Điểm BUY", fmt(r.get("confirmed_score")))
    c.metric("Điểm EARLY", fmt(r.get("early_score")))
    d.metric("RSI", fmt(r.get("rsi"), 2))

    a, b, c, d = st.columns(4)
    a.metric("KL/TB20", fmt(r.get("volume_ratio"), 2))
    b.metric("Cắt lỗ", fmt(r.get("stop_loss")))
    c.metric("Mục tiêu", fmt(r.get("target_price")))
    d.metric("RR", fmt(r.get("reward_risk"), 2))

    st.caption("Ngày dữ liệu: " + str(r.get("data_date", "—")))

    for title, key in [
        ("Điều kiện BUY", "buy_conditions"),
        ("Điều kiện EARLY", "early_conditions"),
        ("Điều kiện NEAR", "near_conditions")
    ]:
        checks = r.get(key) or {}
        if checks:
            with st.expander(title):
                for name, ok in checks.items():
                    st.write(("✅ " if ok else "❌ ") + name)

    if r.get("risk_note"):
        st.warning("RR: " + r["risk_note"])

    if r.get("blockers"):
        st.markdown("**Điều kiện chưa đạt**")
        for item in r["blockers"]:
            st.write("❌ " + str(item))

    if r.get("penalties"):
        st.markdown("**Điểm phạt**")
        for item in r["penalties"]:
            st.write("⚠️ " + str(item))

    if r.get("reasons"):
        with st.expander("Yếu tố tích cực"):
            for item in r["reasons"]:
                st.write("✅ " + str(item))


# =====================================================
# PHÂN TÍCH 5 PHÚT
# =====================================================

@st.fragment(run_every="45s")
def intraday_panel(symbol):
    try:
        candles = fetch_intraday_5m(
            symbol, force_refresh=False
        )
        r = analyze_intraday_5m(candles, symbol=symbol)

        st.subheader(f"{symbol} - Nến 5 phút")
        st.write(r.get("explanation", ""))

        a, b, c, d = st.columns(4)
        a.metric("Tín hiệu", r.get("signal", "—"))
        b.metric("Giá", fmt(r.get("price")))
        c.metric("Điểm", fmt(r.get("score")))
        d.metric("RSI", fmt(r.get("rsi"), 2))

        a, b, c, d = st.columns(4)
        a.metric("EMA9", fmt(r.get("ema9")))
        b.metric("EMA20", fmt(r.get("ema20")))
        c.metric("VWAP", fmt(r.get("vwap")))
        d.metric("KL/TB20", fmt(r.get("volume_ratio"), 2))

        st.caption(
            "Nến cuối: " + str(r.get("timestamp", "—"))
            + " | Nguồn: KBS"
        )

        chart = r["candles"].copy().set_index("Date")
        cols = [
            x for x in ["Close", "EMA9", "EMA20", "VWAP"]
            if x in chart
        ]
        st.line_chart(chart[cols], height=350)

        a, b, c = st.columns(3)
        a.metric("Hỗ trợ", fmt(r.get("support")))
        b.metric("Kháng cự", fmt(r.get("resistance")))
        c.metric("Cắt lỗ tham khảo", fmt(r.get("stop_reference")))

        for item in r.get("reasons", []):
            st.write("✅ " + item)
        for item in r.get("blockers", []):
            st.write("⚠️ " + item)

    except Exception as exc:
        st.warning(f"Không có phân tích 5 phút hợp lệ: {exc}")


tab_market, tab_intraday = st.tabs([
    "🔎 Quét HOSE/HNX",
    "📊 Phân tích 5 phút"
])

with tab_intraday:
    symbol = st.text_input(
        "Mã cổ phiếu",
        value="SAB",
        max_chars=10
    ).strip().upper()

    if symbol and symbol.isalnum():
        intraday_panel(symbol)
    else:
        st.warning("Nhập mã cổ phiếu hợp lệ.")


# =====================================================
# QUÉT THỊ TRƯỜNG D1
# =====================================================

with tab_market:
    st.subheader("Quét toàn thị trường")

    if "scan_output" not in st.session_state:
        st.session_state.scan_output = None

    if "scan_time" not in st.session_state:
        st.session_state.scan_time = None

    if st.button(
        "🚀 BẮT ĐẦU QUÉT",
        type="primary",
        use_container_width=True
    ):
        progress = st.progress(0)
        status = st.empty()

        try:
            symbols = fetch_exchange_symbols()
            total = len(symbols)

            if not total:
                raise RuntimeError("Danh sách HOSE/HNX trống")

            def update(i, n):
                progress.progress(min(i / max(n, 1), 1.0))
                status.caption(f"Đang quét {i}/{n} mã")

            with st.spinner("Đang phân tích D1..."):
                output = scan_market(
                    symbols=symbols,
                    progress_callback=update
                )

            st.session_state.scan_output = output
            st.session_state.scan_time = datetime.now(
                VN_TZ
            ).strftime("%H:%M %d/%m/%Y")

            progress.progress(1.0)
            status.success("Hoàn tất quét")

        except Exception as exc:
            st.error(f"Quét thất bại: {exc}")

    output = st.session_state.scan_output

    if output:
        st.caption(
            "Lần quét: "
            + str(st.session_state.scan_time)
        )

        a, b, c, d, e = st.columns(5)
        a.metric("BUY", output.get("confirmed_buy_count", 0))
        b.metric("EARLY", output.get("early_buy_count", 0))
        c.metric("NEAR", output.get("near_buy_count", 0))
        d.metric("WAIT", output.get("wait_count", 0))
        e.metric("FILTERED", output.get("filtered_count", 0))

        a, b, c = st.columns(3)
        a.metric("Đã quét", output.get("scanned", 0))
        b.metric(
            "Thiếu lịch sử",
            output.get("insufficient_history_count", 0)
        )
        c.metric("Lỗi D1", output.get("d1_error_count", 0))

        results = output.get("results", [])
        full_df = export_frame(results)

        if not full_df.empty:
            st.download_button(
                "⬇️ TẢI CSV ĐẦY ĐỦ LÝ DO",
                data=full_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="ai_stock_scanner_full.csv",
                mime="text/csv"
            )

            st.subheader("Bảng xếp hạng")

            choices = [
                "Tất cả", "BUY", "EARLY_BUY",
                "NEAR_BUY", "WAIT", "FILTERED"
            ]
            selected_signal = st.selectbox(
                "Lọc tín hiệu", choices
            )

            if selected_signal == "Tất cả":
                filtered = results
            else:
                filtered = [
                    x for x in results
                    if x.get("signal") == selected_signal
                ]

            filtered_df = export_frame(filtered)

            visible_columns = [
                "Tín hiệu", "Mã", "Điểm tín hiệu",
                "Điểm BUY", "Điểm EARLY",
                "Giá phân tích", "RR", "RSI",
                "KL/TB20", "Ngày dữ liệu",
                "Nhận định", "Điều kiện thiếu BUY",
                "Điều kiện thiếu EARLY",
                "Điều kiện thiếu NEAR"
            ]

            if not filtered_df.empty:
                st.dataframe(
                    filtered_df[visible_columns],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Không có mã thuộc nhóm này")

            if filtered:
                st.subheader("Phân tích chi tiết")

                names = [
                    f"{x.get('symbol')} | {x.get('signal')}"
                    for x in filtered
                ]

                selected_index = st.selectbox(
                    "Chọn mã để xem",
                    range(len(names)),
                    format_func=lambda i: names[i]
                )

                chosen = filtered[selected_index]
                render_detail(chosen)

            st.subheader("Quản trị vốn")

            capital = st.number_input(
                "Vốn (đồng)",
                min_value=1_000_000,
                value=10_000_000,
                step=1_000_000
            )
            risk_pct = st.number_input(
                "Rủi ro mỗi lệnh (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5
            )

            candidates = [
                r for r in results
                if r.get("signal") in ("BUY", "EARLY_BUY")
                and r.get("stop_loss") is not None
            ]

            plans = []
            for r in candidates:
                p = position_size(
                    r.get("price"),
                    r.get("stop_loss"),
                    capital,
                    risk_pct
                )
                if p:
                    plans.append({
                        "Mã": r["symbol"],
                        "Tín hiệu": r["signal"],
                        "Số cổ phiếu": p["shares"],
                        "Vốn cần": p["cost"],
                        "Lỗ ước tính": p["loss"]
                    })

            if plans:
                st.dataframe(
                    pd.DataFrame(plans),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "Khối lượng làm tròn theo lô 100 cổ phiếu. "
                    "Chưa tính rủi ro trượt giá vượt dự kiến."
                )
            else:
                st.info("Chưa có mã đủ điều kiện lập kế hoạch vốn.")

        if output.get("errors"):
            with st.expander("Chi tiết lỗi dữ liệu"):
                st.dataframe(
                    pd.DataFrame(output["errors"]),
                    use_container_width=True
                )

st.caption(
    "Điểm số là thang đánh giá kỹ thuật, không phải xác suất thắng. "
    "Tín hiệu D1 chỉ được xác nhận sau khi nến ngày hoàn tất."
)
