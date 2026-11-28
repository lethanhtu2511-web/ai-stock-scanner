"""Streamlit page for the historical stock-behavior research module.

Research runs only after the user clicks the button. It is separate from live
scanner signals and never places orders or changes production strategy scores.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

REPORT_PATH = Path("pattern_research_output/pattern_research_report.json")
EVENTS_PATH = Path("pattern_research_output/pattern_events.csv")


def _load_report():
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Không đọc được báo cáo nghiên cứu: {exc}")
        return None


def _show_report(report):
    st.subheader("Kết quả nghiên cứu gần nhất")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mã đã nghiên cứu", report.get("symbols_tested", "—"))
    m2.metric("Số quan sát", f"{report.get('observations', 0):,}")
    m3.metric("Ngày bắt đầu", report.get("data_start", "—"))
    m4.metric("Ngày kết thúc", report.get("data_end", "—"))
    st.caption(
        f"Walk-forward: {report.get('status', '—')} · "
        f"mốc chia train/holdout: {report.get('discovery_cutoff', '—')} · "
        f"lỗi dữ liệu: {report.get('data_errors_count', 0)}"
    )

    holdout = report.get("holdout_by_strategy", {})
    if holdout:
        st.markdown("**Hiệu quả giai đoạn holdout (ngoài mẫu)**")
        rows = []
        names = {"SWING": "Swing 2–5 phiên", "REVERSAL": "Reversal 5–10 phiên",
                 "PULLBACK": "Pullback 3–7 phiên"}
        for key, values in holdout.items():
            rows.append({
                "Chiến lược": names.get(key, key),
                "Số mẫu": values.get("samples"),
                "Tỷ lệ thắng (%)": values.get("win_rate_pct"),
                "Lợi nhuận ròng TB (%)": values.get("avg_net_return_pct"),
                "Profit Factor": values.get("profit_factor"),
                "MFE TB (%)": values.get("avg_mfe_pct"),
                "MAE TB (%)": values.get("avg_mae_pct"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    patterns = report.get("pattern_holdout", [])
    if patterns:
        st.markdown("**Quy luật được phát hiện ở train và kiểm tra ở holdout**")
        rows = []
        for item in patterns:
            metric = item.get("holdout", {})
            rows.append({
                "Chiến lược": item.get("strategy"),
                "Loại": item.get("type"),
                "Quy luật / yếu tố": item.get("pattern", item.get("feature")),
                "Số mẫu train": item.get("training_samples"),
                "Số mẫu holdout": metric.get("samples"),
                "Tỷ lệ thắng holdout (%)": metric.get("win_rate_pct"),
                "Lợi nhuận TB holdout (%)": metric.get("avg_net_return_pct"),
                "Profit Factor holdout": metric.get("profit_factor"),
                "MAE TB (%)": metric.get("avg_mae_pct"),
            })
        pattern_df = pd.DataFrame(rows)
        st.dataframe(pattern_df, use_container_width=True, hide_index=True)
        if "Lợi nhuận TB holdout (%)" in pattern_df:
            chart = pattern_df.set_index("Quy luật / yếu tố")[
                "Lợi nhuận TB holdout (%)"
            ].dropna()
            if not chart.empty:
                st.bar_chart(chart)

    discovery = report.get("discovery", {})
    relationships = discovery.get("feature_relationships", [])
    foreign = [row for row in relationships
               if "foreign" in str(row.get("feature", "")).lower()]
    if foreign:
        st.markdown("**Mối liên hệ của dòng tiền khối ngoại (giai đoạn train)**")
        st.dataframe(pd.DataFrame(foreign), use_container_width=True, hide_index=True)
    elif report.get("symbols_tested"):
        st.info(
            "Chưa có bucket khối ngoại đủ số mẫu để xếp hạng. "
            "Dữ liệu thiếu được giữ là UNKNOWN/NaN, không coi là mua ròng."
        )

    with st.expander("Phương pháp và giới hạn"):
        st.write(report.get("method", "Point-in-time event study"))
        st.write(report.get("note", ""))
        st.warning(report.get("universe_warning", ""))
        st.write(
            "Báo cáo là nghiên cứu lịch sử, không phải khuyến nghị mua. "
            "Các ngày tín hiệu gần nhau có thể chồng lấn kỳ nắm giữ; cổ phiếu hủy niêm yết "
            "không nằm trong danh sách mã hiện tại có thể tạo survivorship bias."
        )

    if EVENTS_PATH.exists():
        st.download_button(
            "Tải dữ liệu sự kiện nghiên cứu (CSV)",
            data=EVENTS_PATH.read_bytes(),
            file_name="pattern_research_events.csv",
            mime="text/csv",
        )
    st.download_button(
        "Tải báo cáo đầy đủ (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="pattern_research_report.json",
        mime="application/json",
    )


def main():
    st.set_page_config(
        page_title="BOT 2 - Nghiên cứu quy luật",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🤖 BOT 2 — NGHIÊN CỨU QUY LUẬT CỔ PHIẾU")
    st.caption(
        "Tìm các hành vi giá lặp lại trên HOSE/HNX, kiểm tra tỷ lệ tăng riêng sau 2, 3, 4 và 5 phiên "
        "và xác minh trên giai đoạn holdout. Có dữ liệu khối ngoại khi nguồn lịch sử cung cấp."
    )
    st.info(
        "Trang này không tự quét khi mở. Chỉ bắt đầu tải dữ liệu khi bạn bấm "
        "**Chạy nghiên cứu**; không phát lệnh mua/bán và không thay đổi điểm Scanner."
    )

    with st.expander("Thiết lập lần nghiên cứu", expanded=True):
        max_symbols = st.select_slider(
            "Số mã thanh khoản cao tối đa để nghiên cứu",
            options=[10, 20, 30, 50, 75, 120],
            value=30,
            help="Chọn 120 để bao phủ rộng hơn; thời gian chạy và giới hạn nguồn dữ liệu tăng đáng kể.",
        )
        lookback = st.select_slider(
            "Số phiên lịch sử yêu cầu",
            options=[260, 420, 600, 800],
            value=420,
            help="Mã phải có tối thiểu 200 phiên. Khoảng thời gian được yêu cầu tăng theo số phiên.",
        )
        st.caption("Bộ lọc: tối thiểu 200 phiên; GTGD bình quân 20 phiên ≥ 5 tỷ đồng; GTGD trung vị 20 phiên ≥ 3 tỷ đồng. Lookback là số phiên tín hiệu; bot tự lấy thêm dữ liệu làm nóng chỉ báo và các phiên để xác định kết quả.")
        st.caption("Chi phí mô phỏng: phí mua 0,15%, phí bán 0,15%, thuế bán 0,10%, trượt giá 0,20% mỗi chiều. Dữ liệu khối ngoại có thể ngắn hơn lịch sử giá.")

    if st.button("▶ Chạy nghiên cứu", type="primary", use_container_width=True):
        old_symbols = os.environ.get("PATTERN_RESEARCH_MAX_SYMBOLS")
        old_lookback = os.environ.get("PATTERN_RESEARCH_LOOKBACK")
        os.environ["PATTERN_RESEARCH_MAX_SYMBOLS"] = str(max_symbols)
        os.environ["PATTERN_RESEARCH_LOOKBACK"] = str(lookback)
        try:
            with st.spinner(
                f"Đang thu thập dữ liệu và kiểm tra {max_symbols} mã. "
                "Quá trình có thể mất nhiều phút tùy nguồn dữ liệu..."
            ):
                from run_pattern_research import main as run_research
                run_research()
            st.success("Đã hoàn tất nghiên cứu. Báo cáo được cập nhật bên dưới.")
        except Exception as exc:
            st.error(f"Nghiên cứu chưa hoàn tất: {exc}")
            st.caption(
                "Có thể do giới hạn truy cập nguồn dữ liệu hoặc thời gian chạy. "
                "Báo cáo cũ (nếu có) vẫn được giữ."
            )
        finally:
            if old_symbols is None:
                os.environ.pop("PATTERN_RESEARCH_MAX_SYMBOLS", None)
            else:
                os.environ["PATTERN_RESEARCH_MAX_SYMBOLS"] = old_symbols
            if old_lookback is None:
                os.environ.pop("PATTERN_RESEARCH_LOOKBACK", None)
            else:
                os.environ["PATTERN_RESEARCH_LOOKBACK"] = old_lookback

    report = _load_report()
    if report:
        _show_report(report)
    else:
        st.warning(
            "Chưa có báo cáo nghiên cứu. Hãy bấm **Chạy nghiên cứu** để tạo báo cáo đầu tiên."
        )


if __name__ == "__main__":
    main()
