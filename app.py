
import os
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from scan_store import (
    init_scan_db,
    get_latest_scan,
    get_scan_history,
    get_scan_summary,
)
from signal_evaluator import evaluate_signal_history
from worker import run_scan, ScanAlreadyRunningError
from portfolio_advisor import advise_position
from portfolio import (
    add_transaction,
    get_positions,
    get_transactions,
    get_portfolio_summary,
    get_cash_summary,
    get_settings,
    save_settings,
    get_storage_backend,
)


# =========================================================
# CONFIGURATION
# =========================================================

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DB_PATH = os.getenv("DATABASE_PATH", "data/stock_scanner.sqlite3")
APP_VERSION = "3.2"
AUTO_REFRESH_SECONDS = 300


st.set_page_config(
    page_title=f"AI Stock Scanner {APP_VERSION}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_scan_db(DB_PATH)

SIGNAL_NAMES = {
    "BUY": "MUA XÁC NHẬN",
    "WAIT": "CHỜ XÁC NHẬN",
    "WATCH": "THEO DÕI",
    "SKIP": "BỎ QUA",
    "FILTERED": "BỊ LỌC",
}

STRATEGY_NAMES = {
    "SWING": "Swing 2–5 phiên",
    "REVERSAL": "Reversal 5–10 phiên",
    "PULLBACK": "Pullback 3–7 phiên",
}

STRATEGY_ORDER = [
    "SWING",
    "REVERSAL",
    "PULLBACK",
]


# =========================================================
# UTILITIES
# =========================================================

def money(value):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.0f} đ"
    except (TypeError, ValueError):
        return "—"


def number(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def is_market_session(now=None):
    now = now or datetime.now(VN_TZ)

    if now.weekday() >= 5:
        return False

    current = now.time()

    return (
        dt_time(9, 0) <= current <= dt_time(11, 30)
        or dt_time(13, 0) <= current <= dt_time(15, 0)
    )


def parse_scan_time(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=VN_TZ)

        return parsed.astimezone(VN_TZ)

    except (TypeError, ValueError):
        return None


def auto_scan_due(now, latest_scan_at):
    if now.weekday() >= 5:
        return False

    latest = parse_scan_time(latest_scan_at)

    if is_market_session(now):
        if latest is None:
            return True

        age = (now - latest).total_seconds()

        return age >= AUTO_REFRESH_SECONDS - 10

    # Quét D1 sau khi thị trường đóng cửa.
    if now.time() >= dt_time(15, 35):
        if latest is None:
            return True

        return not (
            latest.date() == now.date()
            and latest.time() >= dt_time(15, 35)
        )

    return False


def current_strategy_rows(rows):
    return [
        row for row in rows
        if row.get("strategy") in STRATEGY_ORDER
    ]


# =========================================================
# TABLE
# =========================================================

def display_scan_table(rows):
    if not rows:
        st.info("Chưa có kết quả phù hợp.")
        return

    df = pd.DataFrame(rows)

    if "signal" in df.columns:
        df["Tín hiệu"] = (
            df["signal"]
            .map(SIGNAL_NAMES)
            .fillna(df["signal"])
        )

    if "strategy" in df.columns:
        df["Chiến lược"] = (
            df["strategy"]
            .map(STRATEGY_NAMES)
            .fillna(df["strategy"])
        )

    rename = {
        "symbol": "Mã",
        "score": "Điểm",
        "early_signal": "Cảnh báo sớm",
        "early_score": "Điểm cảnh báo",
        "early_volume_ratio": "KL/TB20",
        "price": "Giá phân tích",
        "realtime_price": "Giá realtime",
        "d1_close": "Đóng cửa D1",
        "live_change_pct": "Biến động phiên (%)",
        "stop_loss": "Cắt lỗ",
        "target_price": "Mục tiêu",
        "reward_risk": "RR sau phí",
        "relative_strength": "Sức mạnh tương đối (%)",
        "market_context": "Thị trường",
        "price_data_status": "Trạng thái giá",
        "reason": "Lý do",
        "data_date": "Ngày D1",
        "foreign_net_value": "Khối ngoại ròng (đ)",
        "foreign_net_5d": "Khối ngoại ròng 5 phiên (đ)",
        "foreign_signal": "Xu hướng khối ngoại",
        "foreign_flow_status": "Dữ liệu khối ngoại",
    }

    if "adaptive_research" in df.columns:
        adaptive = df["adaptive_research"].apply(
            lambda value: value if isinstance(value, dict) else {}
        )
        df["adaptive_status"] = adaptive.apply(lambda value: value.get("status", "—"))
        df["adaptive_score"] = adaptive.apply(lambda value: value.get("score"))

    df = df.rename(columns=rename)
    df = df.rename(columns={
        "adaptive_status": "Adaptive (thử nghiệm)",
        "adaptive_score": "Điểm Adaptive",
    })

    preferred = [
        "Mã",
        "Cảnh báo sớm",
        "Điểm cảnh báo",
        "Chiến lược",
        "Tín hiệu",
        "Điểm",
        "Điểm Adaptive",
        "Adaptive (thử nghiệm)",
        "Khối ngoại ròng (đ)",
        "Khối ngoại ròng 5 phiên (đ)",
        "Xu hướng khối ngoại",
        "Giá realtime",
        "Đóng cửa D1",
        "Biến động phiên (%)",
        "Cắt lỗ",
        "Mục tiêu",
        "RR sau phí",
        "Sức mạnh tương đối (%)",
        "Thị trường",
        "Ngày D1",
        "Lý do",
    ]

    columns = [
        col for col in preferred
        if col in df.columns
    ]

    if columns:
        df = df[columns]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# STRATEGY METRICS
# =========================================================

def show_strategy_metrics(df):
    if df.empty:
        return

    cols = st.columns(3)

    for index, strategy in enumerate(STRATEGY_ORDER):
        subset = df[df["strategy"] == strategy]

        buys = int((subset["signal"] == "BUY").sum())
        waits = int((subset["signal"] == "WAIT").sum())
        watches = int((subset["signal"] == "WATCH").sum())

        cols[index].metric(
            STRATEGY_NAMES[strategy],
            f"{buys} MUA",
        )

        cols[index].caption(
            f"Chờ: {waits} | Theo dõi: {watches}"
        )


# =========================================================
# SIGNAL DETAIL
# =========================================================

def show_signal_detail(row):
    st.subheader(f"Phân tích {row.get('symbol', '')}")

    adaptive = row.get("adaptive_research")
    if isinstance(adaptive, dict):
        st.markdown("**Đánh giá Adaptive (thử nghiệm, không phải tín hiệu mua)**")
        st.metric("Điểm Adaptive", adaptive.get("score", "—"))
        st.write(f"Trạng thái: {adaptive.get('status', '—')}")
        if adaptive.get("hard_risks"):
            st.warning("Rủi ro chặn: " + "; ".join(adaptive["hard_risks"]))
        if adaptive.get("soft_confirmations_missing"):
            st.info("Yếu tố cần xác nhận thêm: " + "; ".join(adaptive["soft_confirmations_missing"]))
        st.caption(adaptive.get("note", "Chỉ dùng để tham khảo nghiên cứu."))

    st.write(
        f"**Chiến lược:** "
        f"{STRATEGY_NAMES.get(row.get('strategy'), row.get('strategy'))}"
    )

    candle = row.get("candlestick_analysis")
    if isinstance(candle, dict):
        st.markdown("**Phân tích mô hình nến D1**")
        patterns = candle.get("patterns") or []
        if patterns:
            st.write("Mô hình nhận diện: " + ", ".join(str(x) for x in patterns))
        else:
            st.write("Chưa phát hiện mô hình nến nổi bật.")
        bullish = candle.get("confirmed_bullish") or []
        bearish = candle.get("confirmed_bearish") or []
        neutral = candle.get("neutral") or []
        c1, c2, c3 = st.columns(3)
        c1.metric("Nến tăng xác nhận", len(bullish))
        c2.metric("Nến giảm xác nhận", len(bearish))
        c3.metric("Nến trung tính", len(neutral))
        context = candle.get("context") or {}
        st.caption(
            f"Hỗ trợ gần: {'Có' if context.get('near_support') else 'Không'} · "
            f"Kháng cự gần: {'Có' if context.get('near_resistance') else 'Không'} · "
            f"KL/TB20: {number(context.get('volume_ratio'))} · "
            f"Biến động trước đó 10 phiên: {number(context.get('prior_10d_return_pct'))}%"
        )
        if bullish:
            st.success("Tín hiệu nến tăng: " + ", ".join(bullish))
        if bearish:
            st.warning("Cảnh báo nến giảm: " + ", ".join(bearish))
        for note in candle.get("notes") or []:
            st.write("- " + str(note))

    if row.get("foreign_flow_status") == "AVAILABLE":
        st.write("**Khối ngoại:**")
        st.write(
            f"Mua {money(row.get('foreign_buy_value'))} · "
            f"Bán {money(row.get('foreign_sell_value'))} · "
            f"Ròng {money(row.get('foreign_net_value'))}"
        )
        st.write(
            f"Mua ròng/bán ròng 3 phiên: {money(row.get('foreign_net_3d'))} · "
            f"5 phiên: {money(row.get('foreign_net_5d'))} · "
            f"10 phiên: {money(row.get('foreign_net_10d'))}"
        )
        st.caption(
            f"Ngày dữ liệu: {row.get('foreign_data_date')} · "
            f"Trạng thái: {row.get('foreign_signal')}"
        )
    else:
        st.info("Chưa có dữ liệu giao dịch khối ngoại đáng tin cậy cho mã này.")

    st.write(
        f"**Tín hiệu:** "
        f"{SIGNAL_NAMES.get(row.get('signal'), row.get('signal'))}"
    )

    st.write(
        f"**Điểm kỹ thuật:** "
        f"{number(row.get('score'), 0)}/100"
    )

    status = row.get("price_data_status")
    status_text = (
        row.get("price_data_message")
        or status
        or "Không rõ"
    )

    if status == "REALTIME":
        st.success(status_text)

    elif status in (
        "LAST_SESSION",
        "D1_ONLY",
        "UNVERIFIED",
    ):
        st.warning(status_text)

    else:
        st.info(status_text)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Giá realtime",
        money(row.get("realtime_price")),
    )

    c2.metric(
        "Đóng cửa D1",
        money(row.get("d1_close")),
    )

    c3.metric(
        "Biến động phiên",
        f"{number(row.get('live_change_pct'))}%",
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Cắt lỗ",
        money(row.get("stop_loss")),
    )

    c2.metric(
        "Mục tiêu",
        money(row.get("target_price")),
    )

    c3.metric(
        "RR sau phí",
        number(row.get("reward_risk")),
    )

    for key, heading in [
        ("reasons", "Điều kiện đạt"),
        ("blockers", "Điều kiện chưa đạt"),
        ("penalties", "Điểm cần thận trọng"),
    ]:
        items = row.get(key) or []

        if items:
            st.markdown(f"**{heading}**")

            for item in items:
                st.write(f"- {item}")

    st.caption(
        "Điểm kỹ thuật không phải xác suất thắng. "
        "Giá realtime chỉ được xác nhận khi timestamp từ nguồn còn mới. "
        "Tín hiệu không bảo đảm lợi nhuận."
    )


# =========================================================
# PERIODIC SCAN AND DISPLAY
# =========================================================

@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def auto_refresh_panel(show_details=True):
    summary = get_scan_summary(DB_PATH)
    latest_time = summary.get("latest_scan_at")
    now = datetime.now(VN_TZ)

    if auto_scan_due(now, latest_time):
        try:
            with st.spinner("Đang cập nhật dữ liệu và quét HOSE/HNX..."):
                saved = run_scan()
            st.success(f"Đã hoàn tất lượt quét tự động, lưu {saved:,} dòng.")
            summary = get_scan_summary(DB_PATH)
            latest_time = summary.get("latest_scan_at")
        except ScanAlreadyRunningError:
            st.info("Đang có lượt quét khác xử lý. Trang sẽ tự kiểm tra lại sau, không tạo lượt quét trùng.")
        except Exception as exc:
            st.error(f"Lượt quét tự động chưa thành công: {exc}")

    latest_dt = parse_scan_time(latest_time)
    if latest_dt:
        age_seconds = max(0, (datetime.now(VN_TZ) - latest_dt).total_seconds())
        st.caption(
            f"Lượt quét gần nhất: {latest_dt.strftime('%d/%m/%Y %H:%M:%S')} "
            f"· Cách đây {int(age_seconds)} giây"
        )
    else:
        st.caption("Chưa có lượt quét nào được lưu.")

    if is_market_session():
        st.info("Tự động cập nhật mỗi 300 giây khi trang đang mở. Nếu đóng trang, lượt quét tự động sẽ tạm dừng.")
    elif datetime.now(VN_TZ).time() >= dt_time(15, 35):
        st.info("Ngoài giờ giao dịch: bot thực hiện lượt quét D1 cuối ngày một lần.")
    else:
        st.caption("Ngoài giờ giao dịch: không xác nhận giá realtime.")

    latest = current_strategy_rows(get_latest_scan(DB_PATH))
    if not latest:
        st.warning("Chưa có kết quả quét. Nhấn QUÉT NGAY để chạy lượt đầu tiên.")
        return

    df = pd.DataFrame(latest)
    breadth = next((row.get("market_breadth") for row in latest if isinstance(row.get("market_breadth"), dict)), None)
    if breadth and breadth.get("status") == "AVAILABLE":
        st.subheader("Độ rộng thị trường trong lượt quét")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Mã trong mẫu", breadth.get("sample_size", 0))
        b2.metric("Trên EMA20", f"{number(breadth.get('above_ema20_pct'))}%")
        b3.metric("Trên EMA50", f"{number(breadth.get('above_ema50_pct'))}%")
        b4.metric("Tăng / giảm", f"{breadth.get('advancing', 0)} / {breadth.get('declining', 0)}")
        st.caption(
            f"Bối cảnh độ rộng: {breadth.get('regime', 'UNKNOWN')} · "
            f"Lợi suất bình quân 5 phiên: {number(breadth.get('average_5d_return_pct'))}% · "
            f"{breadth.get('note', '')}"
        )
    elif breadth:
        st.info("Chưa đủ dữ liệu D1 để tính độ rộng thị trường trong lượt quét này.")

    st.subheader("Trạng thái dữ liệu")

    status_source = (
        df.drop_duplicates("symbol")
        if "symbol" in df.columns
        else df
    )

    status_counts = status_source.get(
        "price_data_status",
        pd.Series(dtype=str),
    ).value_counts()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Realtime xác minh",
        int(status_counts.get("REALTIME", 0)),
    )

    c2.metric(
        "Giá phiên gần nhất",
        int(status_counts.get("LAST_SESSION", 0)),
    )

    c3.metric(
        "Chưa xác minh",
        int(status_counts.get("UNVERIFIED", 0)),
    )

    c4.metric(
        "D1 cũ",
        int(status_counts.get("D1_STALE", 0)),
    )

    verified_rows = (
        status_source[
            status_source.get("price_data_status") == "REALTIME"
        ]
        if "price_data_status" in status_source.columns
        else pd.DataFrame()
    )

    if (
        not verified_rows.empty
        and "quote_source_time" in verified_rows.columns
    ):

        source_times = pd.to_datetime(
            verified_rows["quote_source_time"],
            errors="coerce",
        ).dropna()

        if not source_times.empty:
            st.caption(
                "Timestamp mới nhất do nguồn báo giá cung cấp: "
                f"{source_times.max().strftime('%d/%m/%Y %H:%M:%S')}"
            )

    if "quote_age_seconds" in df.columns:

        verified = df[
            df.get("price_data_status") == "REALTIME"
        ]

        ages = pd.to_numeric(
            verified.get("quote_age_seconds"),
            errors="coerce",
        ).dropna()

        if not ages.empty:
            st.caption(
                "Độ trễ lớn nhất của báo giá đã xác minh: "
                f"{int(ages.max())} giây."
            )

    show_strategy_metrics(df)

    signal_counts = df["signal"].value_counts()

    cols = st.columns(5)

    for idx, signal in enumerate(
        ["BUY", "WAIT", "WATCH", "SKIP", "FILTERED"]
    ):

        cols[idx].metric(
            SIGNAL_NAMES[signal],
            int(signal_counts.get(signal, 0)),
        )

    st.divider()

    st.subheader("Cơ hội ưu tiên")

    candidates = df[
        df["signal"].isin(["BUY", "WAIT"])
    ].copy()

    if not candidates.empty:

        candidates["_rank"] = candidates["signal"].map(
            {"BUY": 0, "WAIT": 1}
        )

        candidates["_rr"] = pd.to_numeric(
            candidates.get("reward_risk"),
            errors="coerce",
        )

        candidates = candidates.sort_values(
            ["_rank", "score", "_rr"],
            ascending=[True, False, False],
            na_position="last",
        )

        display_scan_table(
            candidates.head(20).to_dict("records")
        )

        if show_details:

            top_candidates = candidates.head(30)

            options = [
                (
                    f"{row['symbol']} · "
                    f"{STRATEGY_NAMES.get(row['strategy'], row['strategy'])} · "
                    f"{SIGNAL_NAMES.get(row['signal'], row['signal'])} · "
                    f"{row.get('score', 0)} điểm"
                )
                for _, row in top_candidates.iterrows()
            ]

            selected = st.selectbox(
                "Chọn mã xem phân tích",
                options,
                key="auto_scan_candidate_detail",
            )

            selected_index = options.index(selected)

            show_signal_detail(
                top_candidates.iloc[selected_index].to_dict()
            )

    else:
        st.info(
            "Hiện chưa có mã đủ điều kiện MUA hoặc CHỜ. "
            "Bot không cố tạo tín hiệu khi thị trường chưa phù hợp."
        )

    if show_details:
        with st.expander("Toàn bộ kết quả quét"):
            display_scan_table(latest)


# =========================================================
# PAGES
# =========================================================

def show_dashboard():

    st.title("📈 AI STOCK SCANNER")

    st.caption(
        "HOSE/HNX · Swing 2–5 phiên · "
        "Reversal 5–10 phiên · Pullback 3–7 phiên"
    )

    auto_refresh_panel(show_details=False)


def show_scanner():

    st.title("🔎 QUÉT CỔ PHIẾU")

    st.info(
        "Bot tự cập nhật mỗi 300 giây trong giờ giao dịch khi trang đang mở; "
        "anh cũng có thể chủ động quét ngay. Bot không tự đặt lệnh."
    )

    if st.button("🔄 LẤY DỮ LIỆU MỚI & QUÉT LẠI", type="primary", use_container_width=True):
        try:
            with st.spinner("Đang cập nhật dữ liệu và phân tích HOSE/HNX..."):
                saved = run_scan()
            st.success(f"Đã hoàn tất lượt quét, lưu {saved:,} dòng kết quả.")
            st.rerun()
        except ScanAlreadyRunningError:
            st.info("Một lượt quét khác đang chạy. Bot sẽ cập nhật kết quả khi lượt đó hoàn tất.")
        except Exception as exc:
            st.error(f"Quét thất bại: {exc}")

    auto_refresh_panel(show_details=True)


def show_self_evaluation():
    st.title("🧪 TỰ ĐÁNH GIÁ TÍN HIỆU")
    st.caption(
        "Đối chiếu tín hiệu BUY và cảnh báo sớm với diễn biến giá D1 "
        "sau 1, 3, 5 và 10 phiên."
    )

    with st.spinner("Đang tổng hợp lịch sử tín hiệu..."):
        history = get_scan_history(limit=5000, db_path=DB_PATH)
        evaluation = evaluate_signal_history(history)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tín hiệu đã ghi nhận", evaluation["predictions_recorded"])
    c2.metric("Mẫu đã đủ 5 phiên", evaluation["five_session_matured"])
    c3.metric("Mẫu chưa đủ 5 phiên", evaluation["open_predictions"])

    metrics = evaluation["metrics"]
    if not metrics:
        st.info(
            "Chưa có đủ dữ liệu lịch sử để đánh giá. Bot sẽ tự tích lũy "
            "các tín hiệu và giá đóng cửa D1 qua những phiên tiếp theo."
        )
    else:
        df = pd.DataFrame(metrics)
        df = df.rename(columns={
            "strategy": "Chiến lược",
            "signal_type": "Loại tín hiệu",
            "horizon": "Số phiên",
            "sample_size": "Số mẫu",
            "win_rate_pct": "Tỷ lệ tăng (%)",
            "average_return_pct": "Lợi nhuận TB (%)",
            "median_return_pct": "Trung vị (%)",
            "best_return_pct": "Tốt nhất (%)",
            "worst_return_pct": "Thấp nhất (%)",
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption(
            "Tỷ lệ tăng được tính từ giá đóng cửa D1 ngày phát tín hiệu "
            "đến giá đóng cửa ở mốc tương ứng; không phải tỷ lệ thắng theo "
            "lệnh thực tế. Cần tối thiểu 30 mẫu cho mỗi nhóm trước khi "
            "dùng thống kê để cân nhắc điều chỉnh chiến lược."
        )

    st.warning(evaluation["method_note"])


def show_history():

    st.title("🕘 LỊCH SỬ QUÉT")

    history = current_strategy_rows(
        get_scan_history(
            limit=500,
            db_path=DB_PATH,
        )
    )

    if not history:
        st.info("Chưa có lịch sử quét.")
        return

    df = pd.DataFrame(history)

    if "scanned_at" in df.columns:
        st.caption(
            "Lịch sử quét được lưu theo cấu hình cơ sở dữ liệu hiện tại; "
            "nếu đã bật đồng bộ Supabase, kết quả được chia sẻ giữa các phiên."
        )

    signal_filter = st.selectbox(
        "Lọc tín hiệu",
        [
            "Tất cả",
            "MUA",
            "CHỜ",
            "THEO DÕI",
            "BỎ QUA",
            "BỊ LỌC",
        ],
    )

    mapping = {
        "MUA": "BUY",
        "CHỜ": "WAIT",
        "THEO DÕI": "WATCH",
        "BỎ QUA": "SKIP",
        "BỊ LỌC": "FILTERED",
    }

    if signal_filter != "Tất cả":
        df = df[
            df["signal"] == mapping[signal_filter]
        ]

    display_scan_table(
        df.to_dict("records")
    )



# =========================================================
# PORTFOLIO
# =========================================================

def show_portfolio():
    st.title("💼 QUẢN LÝ DANH MỤC")

    st.caption(
        "Theo dõi vị thế, giá vốn bình quân đã gồm phí mua, "
        "lãi/lỗ tạm tính sau phí và thuế bán. Bot chỉ tư vấn, "
        "không tự đặt lệnh."
    )

    settings = get_settings(DB_PATH)
    storage_backend = get_storage_backend()
    if storage_backend == "SQLITE":
        st.warning(
            "Danh mục đang lưu bằng SQLite tại môi trường chạy hiện tại. "
            "Nếu ứng dụng chạy trên nền tảng cloud có ổ đĩa tạm thời, dữ liệu có thể mất khi khởi động lại/deploy. "
            "Chỉ dùng giao dịch thật sau khi đã xác nhận cơ chế lưu trữ bền vững hoặc sao lưu dữ liệu."
        )
    else:
        st.success("Danh mục đang đồng bộ qua Supabase.")
    positions = get_positions(DB_PATH)
    latest_rows = current_strategy_rows(get_latest_scan(DB_PATH))

    # Lấy một dòng có dữ liệu giá tốt nhất cho mỗi mã.
    market_by_symbol = {}
    strategy_rows_by_symbol = {}
    signal_rank = {"BUY": 0, "WAIT": 1, "WATCH": 2, "SKIP": 3, "FILTERED": 4}
    price_rank = {
        "REALTIME": 0,
        "LAST_SESSION": 1,
        "D1_ONLY": 2,
        "UNVERIFIED": 3,
        "D1_STALE": 9,
    }
    for row in latest_rows:
        row_symbol = row.get("symbol")
        if row_symbol:
            strategy_rows_by_symbol.setdefault(row_symbol, []).append(row)

    for row in sorted(
        latest_rows,
        key=lambda x: (
            price_rank.get(x.get("price_data_status"), 4),
            signal_rank.get(x.get("signal"), 9),
            -(float(x.get("score") or 0)),
        ),
    ):
        symbol = row.get("symbol")
        if not symbol or symbol in market_by_symbol:
            continue
        price_status = row.get("price_data_status")
        quote_is_verified = price_status == "REALTIME"

        # Không định giá vị thế bằng nến D1 đã bị đánh dấu quá cũ.
        if price_status == "D1_STALE":
            market_price = None
        elif quote_is_verified:
            market_price = row.get("realtime_price")
        else:
            market_price = row.get("d1_close") or row.get("price")

        market_by_symbol[symbol] = {
            "price": float(market_price) * 1000 if market_price is not None else None,
            "source": (
                "Realtime đã xác minh"
                if quote_is_verified
                else (
                    f"D1/phiên gần nhất · {row.get('data_date') or 'ngày chưa rõ'}"
                    if market_price is not None
                    else "Không dùng được: D1 đã cũ"
                )
            ),
            "data_date": row.get("data_date"),
            "signal": row.get("signal"),
            "strategy": STRATEGY_NAMES.get(row.get("strategy"), row.get("strategy")),
            "score": row.get("score"),
            "reason": row.get("reason"),
            "price_status": price_status,
        }

    st.subheader("Tổng quan")
    summary = get_portfolio_summary(DB_PATH)
    cash = get_cash_summary(DB_PATH)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mã đang nắm giữ", len(positions))
    c2.metric("Giá trị vốn còn lại", money(summary.get("total_cost", 0)))
    c3.metric("Lãi/lỗ đã thực hiện", money(summary.get("realized_pnl", 0)))
    c4.metric("Vốn ròng đã giải ngân", money(cash.get("net_buys", 0) - cash.get("net_sells", 0)))

    if positions:
        display_rows = []
        total_market_value = 0.0
        total_unrealized = 0.0
        priced_positions = 0
        for position in positions:
            symbol = position["symbol"]
            market = market_by_symbol.get(symbol, {})
            current = market.get("price")
            avg_cost = float(position["avg_cost"])
            quantity = int(position["quantity"])
            if current is not None:
                current = float(current)
                net_value = quantity * current * (
                    1 - float(settings["fee_sell"]) - float(settings["sell_tax"])
                )
                cost_value = quantity * avg_cost
                unrealized = net_value - cost_value
                total_market_value += net_value
                total_unrealized += unrealized
                priced_positions += 1
                pnl_pct = unrealized / cost_value * 100 if cost_value else 0
            else:
                unrealized = None
                pnl_pct = None
            advice = market.get("signal", "CHƯA CÓ TÍN HIỆU")
            recommendation = advise_position(
                position, market, strategy_rows_by_symbol.get(symbol, []), settings
            )
            estimated_sell_value = (
                recommendation["quantity"] * current
                * (1 - float(settings["fee_sell"]) - float(settings["sell_tax"]))
                if current is not None and recommendation["quantity"] > 0 else None
            )
            display_rows.append({
                "Mã": symbol,
                "Số lượng": quantity,
                "Giá vốn TB (nghìn đồng/cp, gồm phí)": round(avg_cost / 1000, 3),
                "Giá tham chiếu (nghìn đồng/cp)": round(current / 1000, 3) if current is not None else None,
                "Ngày dữ liệu giá": market.get("data_date") or "—",
                "Nguồn giá": market.get("source", "Chưa có dữ liệu"),
                "Lãi/lỗ tạm tính": round(unrealized, 0) if unrealized is not None else None,
                "Lãi/lỗ (%)": round(pnl_pct, 2) if pnl_pct is not None else None,
                "Tín hiệu mới nhất": SIGNAL_NAMES.get(advice, advice),
                "Chiến lược": market.get("strategy", "—"),
                "ĐỀ XUẤT": recommendation["label"],
                "SL đề xuất bán": recommendation["quantity"] or "—",
                "Giá trị bán ròng ước tính": round(estimated_sell_value, 0) if estimated_sell_value is not None else None,
                "Lý do đề xuất": recommendation["reason"],
            })
        st.subheader("Khuyến nghị hành động cho từng mã")
        advice_df = pd.DataFrame(display_rows)
        advice_columns = [
            "Mã", "ĐỀ XUẤT", "SL đề xuất bán", "Giá tham chiếu (nghìn đồng/cp)",
            "Giá trị bán ròng ước tính", "Lãi/lỗ (%)", "Lý do đề xuất", "Nguồn giá",
        ]
        st.dataframe(advice_df[advice_columns], use_container_width=True, hide_index=True)
        st.caption(
            "BÁN HẾT ưu tiên khi giá chạm ngưỡng cắt lỗ; BÁN MỘT PHẦN khi chạm mục tiêu hoặc có đồng thuận suy yếu. "
            "Khối lượng bán được làm tròn theo lô 100 cổ phiếu. Đây là khuyến nghị, không phải lệnh giao dịch."
        )
        st.subheader("Chi tiết vị thế đang nắm giữ")
        detail_columns = [
            "Mã", "Số lượng", "Giá vốn TB (nghìn đồng/cp, gồm phí)",
            "Giá tham chiếu (nghìn đồng/cp)", "Ngày dữ liệu giá", "Lãi/lỗ tạm tính",
            "Lãi/lỗ (%)", "Tín hiệu mới nhất", "Chiến lược",
        ]
        st.dataframe(advice_df[detail_columns], use_container_width=True, hide_index=True)
        st.metric("Lãi/lỗ tạm tính sau phí bán và thuế (các mã có giá)", money(total_unrealized))
        st.caption(f"Đã định giá {priced_positions}/{len(positions)} mã đang nắm giữ. Các mã chưa có giá hợp lệ không được cộng vào tổng này.")
        st.caption(
            "Giá tham chiếu ưu tiên realtime chỉ khi timestamp nguồn được xác minh; "
            "nếu không dùng D1/phiên gần nhất và hiển thị ngày dữ liệu. "
            "Nếu D1 đã quá cũ, bot không tính lãi/lỗ cho mã đó. "
            "Đây là số liệu ước tính, không phải giá khớp lệnh."
        )
        st.metric("Giá trị thị trường ròng ước tính (các mã có giá)", money(total_market_value))
    else:
        st.info("Chưa có vị thế. Ghi nhận giao dịch mua đầu tiên ở phần bên dưới.")

    st.divider()
    st.subheader("Ghi nhận giao dịch")
    with st.form("portfolio_trade_form", clear_on_submit=True):
        left, right = st.columns(2)
        side = left.selectbox("Loại giao dịch", ["MUA", "BÁN"])
        symbol = right.text_input("Mã cổ phiếu", max_chars=10).strip().upper()
        qty_col, price_col = st.columns(2)
        quantity = qty_col.number_input("Số lượng", min_value=100, step=100, value=100)
        price = price_col.number_input("Giá khớp (nghìn đồng/cổ phiếu, ví dụ 24.2)", min_value=0.01, step=0.1, value=10.0)
        note = st.text_input("Ghi chú (không bắt buộc)")
        submitted = st.form_submit_button("LƯU GIAO DỊCH", type="primary", use_container_width=True)
        if submitted:
            try:
                result = add_transaction(
                    side, symbol, int(quantity), float(price) * 1000, note, db_path=DB_PATH
                )
                st.success(
                    f"Đã ghi nhận {side} {symbol}: {int(quantity):,} cp. "
                    f"Phí: {money(result['fee'])}; thuế: {money(result['tax'])}."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Không thể lưu giao dịch: {exc}")

    with st.expander("Cài đặt vốn và phí giao dịch"):
        with st.form("portfolio_settings_form"):
            capital = st.number_input("Vốn dự kiến (đồng)", min_value=0.0, value=float(settings["capital"]), step=1000000.0)
            risk_pct = st.number_input("Rủi ro tối đa mỗi lệnh (%)", min_value=0.1, max_value=10.0, value=float(settings["risk_pct"]), step=0.1)
            fee_buy = st.number_input("Phí mua (%)", min_value=0.0, max_value=10.0, value=float(settings["fee_buy"]) * 100, step=0.01) / 100
            fee_sell = st.number_input("Phí bán (%)", min_value=0.0, max_value=10.0, value=float(settings["fee_sell"]) * 100, step=0.01) / 100
            sell_tax = st.number_input("Thuế bán (%)", min_value=0.0, max_value=10.0, value=float(settings["sell_tax"]) * 100, step=0.01) / 100
            save = st.form_submit_button("LƯU CÀI ĐẶT")
            if save:
                try:
                    save_settings({
                        "capital": capital,
                        "risk_pct": risk_pct,
                        "fee_buy": fee_buy,
                        "fee_sell": fee_sell,
                        "sell_tax": sell_tax,
                    }, DB_PATH)
                    st.success("Đã lưu cài đặt.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể lưu cài đặt: {exc}")

    with st.expander("Lịch sử giao dịch"):
        transactions = get_transactions(DB_PATH)
        if transactions:
            st.dataframe(pd.DataFrame(transactions), use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có giao dịch được ghi nhận.")

def show_system_info():

    st.title("⚙️ THÔNG TIN HỆ THỐNG")

    st.write(f"Phiên bản: AI Stock Scanner {APP_VERSION}")
    st.write("Thị trường: HOSE / HNX")
    st.write("Cập nhật dữ liệu: mỗi 300 giây khi trang đang mở hoặc bấm Quét lại")
    st.write("Phân tích kỹ thuật: D1")
    st.write("Chiến lược: Swing / Reversal / Pullback")
    st.write("Tự động đặt lệnh: Không")
    st.write("Quản lý danh mục: Đã bật")

    st.info("Bot tự quét mỗi 300 giây trong giờ giao dịch khi phiên Streamlit còn mở; nếu đóng trang, việc quét tự động tạm dừng. Giá chỉ được xác nhận realtime khi timestamp nguồn hợp lệ.")


# =========================================================
# MAIN
# =========================================================

def main():

    with st.sidebar:

        st.title("AI STOCK SCANNER")

        page = st.radio(
            "Điều hướng",
            [
                "Tổng quan",
                "Quét cổ phiếu",
                "Lịch sử quét",
                "Tự đánh giá",
                "Backtest lịch sử",
                "Quản lý danh mục",
                "Thông tin hệ thống",
            ],
        )

        st.divider()

        st.caption(
            f"AI Stock Scanner {APP_VERSION}"
        )

        st.caption("Tự quét 300 giây khi phiên còn mở")

    if page == "Tổng quan":
        show_dashboard()

    elif page == "Quét cổ phiếu":
        show_scanner()

    elif page == "Lịch sử quét":
        show_history()

    elif page == "Tự đánh giá":
        show_self_evaluation()

    elif page == "Backtest lịch sử":
        st.switch_page("pages/1_Backtest_Lich_su.py")

    elif page == "Quản lý danh mục":
        show_portfolio()

    else:
        show_system_info()


if __name__ == "__main__":
    main()
