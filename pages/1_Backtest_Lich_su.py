import io

import pandas as pd
import streamlit as st

from historical_backtest import run_historical_backtest

st.set_page_config(
    page_title="Backtest lịch sử | AI Stock Scanner",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 BACKTEST LỊCH SỬ")
st.caption(
    "Kiểm định walk-forward: mỗi tín hiệu chỉ dùng dữ liệu có tại ngày phát tín hiệu; "
    "mô phỏng mua ở phiên kế tiếp, có phí, thuế và trượt giá."
)

with st.sidebar:
    st.subheader("Thiết lập kiểm định")
    max_symbols = st.slider(
        "Số mã tối đa trong lượt thử",
        min_value=10,
        max_value=300,
        value=50,
        step=10,
        help="Lượt đầu nên thử 30–50 mã để kiểm tra pipeline. Tăng dần khi nguồn dữ liệu ổn định.",
    )
    lookback = st.slider(
        "Số phiên lịch sử để kiểm tra",
        min_value=30,
        max_value=240,
        value=120,
        step=10,
    )
    strategy = st.selectbox(
        "Chiến lược",
        ["ALL", "EARLY", "SWING", "REVERSAL", "PULLBACK"],
        format_func=lambda x: {
            "ALL": "Tất cả chiến lược + tín hiệu sớm",
            "EARLY": "Tín hiệu sớm: PREPARE / EARLY_BREAKOUT",
            "SWING": "Swing 2–5 phiên",
            "REVERSAL": "Reversal 5–10 phiên",
            "PULLBACK": "Pullback 3–7 phiên",
        }[x],
    )
    st.caption(
        "Chi phí mô phỏng: mua 0,15% · bán 0,15% · thuế bán 0,10% · "
        "trượt giá 0,20% mỗi chiều."
    )

run = st.button(
    "▶ CHẠY BACKTEST",
    type="primary",
    use_container_width=True,
)

if run:
    progress = st.progress(0, text="Chuẩn bị dữ liệu...")
    status = st.empty()

    def update_progress(done, total, message):
        total = max(1, int(total))
        progress.progress(min(1.0, done / total), text=message)
        status.caption(f"{done:,}/{total:,} · {message}")

    try:
        with st.spinner("Đang tải dữ liệu D1 và kiểm định theo thời gian..."):
            result = run_historical_backtest(
                lookback_sessions=lookback,
                max_symbols=max_symbols,
                strategy_filter=strategy,
                progress_callback=update_progress,
            )
        st.session_state["historical_backtest_result"] = result
        progress.progress(1.0, text="Hoàn tất")
        status.success("Backtest đã hoàn tất.")
    except Exception as exc:
        st.error(f"Không thể chạy backtest: {exc}")

result = st.session_state.get("historical_backtest_result")

if result:
    summary = result["summary"]
    st.subheader("Kết quả tổng quan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số giao dịch", f'{summary["trades"]:,}')
    c2.metric("Tỷ lệ thắng", f'{summary["win_rate_pct"]:.1f}%')
    c3.metric("Lợi nhuận TB / lệnh", f'{summary["avg_return_pct"]:.2f}%')
    c4.metric("Profit factor", "—" if summary["profit_factor"] is None else f'{summary["profit_factor"]:.2f}')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lũy kế chuỗi lệnh (tham khảo)", f'{summary["total_return_pct"]:.2f}%')
    c2.metric("Sụt giảm chuỗi lệnh", f'{summary["max_drawdown_pct"]:.2f}%')
    c3.metric("Lãi/lỗ trung vị", f'{summary["median_return_pct"]:.2f}%')
    c4.metric("Thời gian nắm giữ TB", f'{summary["avg_holding_sessions"]:.1f} phiên')

    st.subheader("Hiệu quả theo chiến lược và tín hiệu sớm")
    st.caption(
        "PREPARE và EARLY_BREAKOUT được đo riêng bằng giá đóng cửa sau 5 phiên; "
        "không giả định có quote realtime hay mức stop/target xác nhận."
    )
    if result["by_strategy"]:
        strategy_df = pd.DataFrame(result["by_strategy"]).rename(columns={
            "strategy": "Chiến lược",
            "trades": "Số lệnh",
            "win_rate_pct": "Tỷ lệ thắng (%)",
            "avg_return_pct": "Lợi nhuận TB/lệnh (%)",
            "median_return_pct": "Trung vị (%)",
            "profit_factor": "Profit factor",
            "max_drawdown_pct": "Drawdown tối đa (%)",
            "avg_holding_sessions": "Nắm giữ TB (phiên)",
            "total_return_pct": "Lợi nhuận tích lũy (%)",
        })
        st.dataframe(strategy_df, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có tín hiệu MUA xác nhận trong mẫu kiểm định.")

    trades = result["trades"]
    st.subheader("Chi tiết giao dịch giả lập")
    if not trades.empty:
        st.dataframe(trades, use_container_width=True, hide_index=True)
        csv = trades.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Tải kết quả giao dịch CSV",
            data=csv,
            file_name="ai_stock_scanner_backtest.csv",
            mime="text/csv",
        )
    else:
        st.info("Không phát sinh giao dịch đạt điều kiện MUA trong giai đoạn kiểm định.")

    portfolio = result.get("portfolio", {})
    st.subheader("Mô phỏng quản trị vốn 10 triệu đồng")
    if portfolio:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Lệnh được nhận", portfolio.get("accepted_trades", 0))
        p2.metric("Lệnh bị bỏ qua", portfolio.get("skipped_trades", 0))
        p3.metric("Vốn cuối kỳ", f'{portfolio.get("ending_equity", 0):,.0f} đ')
        p4.metric("Hiệu suất danh mục", f'{portfolio.get("return_pct", 0):.2f}%')
        skip_reasons = portfolio.get("skip_reasons", {})
        if skip_reasons:
            reason_labels = {
                "MAX_POSITIONS": "Đã đủ số vị thế tối đa",
                "DUPLICATE_SYMBOL_POSITION": "Trùng mã đang nắm giữ",
                "MISSING_OR_INVALID_FIELDS": "Thiếu dữ liệu đầu vào",
                "INVALID_STOP_OR_PRICE": "Stop-loss/giá không hợp lệ",
                "INSUFFICIENT_ALLOCATION_FOR_BOARD_LOT": "Không đủ hạn mức vốn cho lô 100",
                "RISK_BUDGET_TOO_SMALL_FOR_BOARD_LOT": "Ngân sách rủi ro không đủ cho lô 100",
                "INSUFFICIENT_SHARES_AFTER_ROUNDING": "Không đủ cổ phiếu sau khi làm tròn lô",
            }
            st.caption("Lý do bỏ qua lệnh:")
            st.dataframe(
                pd.DataFrame([
                    {"Lý do": reason_labels.get(key, key), "Số lệnh": value}
                    for key, value in skip_reasons.items()
                ]),
                use_container_width=True, hide_index=True,
            )
        sizing_details = portfolio.get("sizing_diagnostics", [])
        if sizing_details:
            st.caption("Chi tiết vì sao từng lệnh không đủ điều kiện vào danh mục:")
            sizing_df = pd.DataFrame(sizing_details).rename(columns={
                "symbol": "Mã", "strategy": "Chiến lược",
                "entry_price": "Giá dự kiến", "stop_loss": "Stop-loss",
                "risk_budget": "Ngân sách rủi ro",
                "allocation_limit": "Hạn mức vốn/vị thế",
                "risk_shares_before_lot": "Số cp theo rủi ro",
                "allocation_shares_before_lot": "Số cp theo hạn mức vốn",
                "minimum_lot_value": "Giá trị lô 100",
                "risk_for_one_lot": "Rủi ro nếu mua lô 100",
                "allocation_constraint_failed": "Vướng hạn mức vốn",
                "risk_constraint_failed": "Vướng ngân sách rủi ro",
                "reason": "Lý do chính",
            })
            st.dataframe(sizing_df, use_container_width=True, hide_index=True)
        st.caption(portfolio.get("note", ""))
    else:
        st.info("Chưa có dữ liệu mô phỏng danh mục.")

    st.subheader("Phạm vi và chất lượng dữ liệu")
    st.caption(
        f"Ngày dữ liệu cuối: {result.get('data_end_date') or 'không xác định'} · "
        f"VN-Index benchmark: {'Có' if result.get('benchmark_available') else 'Không có; bỏ qua Relative Strength'}"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Mã trong danh sách", result["symbols_requested"])
    c2.metric("Mã đã kiểm định", result["symbols_tested"])
    c3.metric("Lỗi tải dữ liệu", len(result["data_errors"]))
    st.metric(
        "Mã có lịch sử khối ngoại",
        f'{result.get("foreign_flow_symbols", 0)}/{result["symbols_tested"]}',
    )
    missing_foreign = result.get("foreign_flow_missing_symbols", [])
    if missing_foreign:
        st.caption("Thiếu dữ liệu khối ngoại: " + ", ".join(missing_foreign))
    st.caption("Mã được chọn: " + ", ".join(result["selected_symbols"]))
    if result["data_errors"]:
        with st.expander("Mã lỗi dữ liệu"):
            st.dataframe(pd.DataFrame(result["data_errors"]), use_container_width=True, hide_index=True)

    st.warning(result["method_note"])
    st.info(
        "Lưu ý: đây là mô phỏng nghiên cứu, không phải bằng chứng bot sẽ sinh lời trong tương lai. "
        "Cần chạy nhiều giai đoạn thị trường và kiểm tra ngoài mẫu trước khi dùng tín hiệu thật."
    )
else:
    st.info(
        "Chưa có kết quả trong phiên này. Hãy bắt đầu với 30–50 mã và 120 phiên; "
        "sau khi kiểm tra nguồn dữ liệu ổn định, tăng phạm vi để kiểm định rộng hơn."
    )
