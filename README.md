# AI Stock Scanner

Bot phân tích cổ phiếu HOSE/HNX theo dữ liệu D1, kết hợp bảng giá trong phiên nếu nguồn cung cấp timestamp hợp lệ.

## Chiến lược
- Swing: 2–5 phiên
- Reversal: 5–10 phiên
- Pullback: 3–7 phiên

Bot chỉ đưa ra phân tích và khuyến nghị. Bot không tự đặt lệnh.

## Chạy bằng Docker Compose

Yêu cầu: Docker Engine và Docker Compose plugin.

```bash
git clone https://github.com/lethanhtu2511-web/ai-stock-scanner.git
cd ai-stock-scanner
docker compose up -d --build
```

Mở giao diện tại `http://localhost:8501`.

Compose khởi chạy hai dịch vụ:
- `app`: giao diện Streamlit.
- `worker`: quét nền theo chu kỳ 300 giây trong giờ giao dịch, quét D1 cuối ngày sau 15:35 và tự sao lưu SQLite sau 15:45 (giờ Việt Nam).

Hai dịch vụ dùng chung volume Docker `stock_data` tại `/app/data`, vì vậy lịch sử quét và danh mục được lưu trong cùng cơ sở dữ liệu SQLite.

## Kiểm tra dịch vụ

```bash
docker compose ps
docker compose logs -f worker
docker compose logs -f app
```

Dừng dịch vụ:

```bash
docker compose down
```

Lệnh trên không xóa volume dữ liệu. Không chạy `docker compose down -v` nếu chưa sao lưu vì lệnh đó xóa volume.

## Sao lưu và phục hồi dữ liệu

- Worker tự tạo bản sao lưu SQLite mỗi ngày giao dịch sau 15:45.
- Bản sao lưu nằm trong thư mục `./backups` trên máy chạy Docker (mount vào `/app/backups`); mặc định giữ 14 bản gần nhất.
- Thư mục backup tách khỏi Docker volume `stock_data`, nên xóa riêng volume dữ liệu không xóa luôn các bản sao lưu trên máy chủ.
- Sao lưu dùng SQLite Online Backup API và kiểm tra `PRAGMA integrity_check` trước khi công bố file.
- Thư mục `./backups` tách khỏi volume dữ liệu, giúp khôi phục khi volume SQLite bị xóa/hỏng. Tuy nhiên thư mục vẫn nằm trên cùng máy chủ; cần sao chép định kỳ ra một thiết bị hoặc dịch vụ lưu trữ độc lập để phòng mất máy chủ.

Liệt kê bản sao lưu trên máy chủ:

```bash
ls -lh ./backups
```

Trước khi phục hồi, dừng cả app và worker, sau đó chép file backup cần dùng thành database chính. Không phục hồi khi app/worker còn đang chạy.

```bash
docker compose stop app worker
# Thực hiện restore bằng backup_manager.restore_database(..., confirm=True)
docker compose up -d
```

Hàm restore từ chối chạy nếu không truyền xác nhận tường minh và từ chối file backup không vượt qua integrity check.

## Dữ liệu và giới hạn

- Dữ liệu khối ngoại: lấy lịch sử mua/bán đã chốt theo mã qua get_foreign_history của vn-market-data, nguồn CafeF; cache cục bộ theo cơ chế của provider. Dữ liệu lịch sử hiện được công bố khoảng 63 phiên, không đủ bao phủ toàn bộ backtest 120 phiên.
- Khối ngoại hiển thị giá trị/khối lượng mua, bán, ròng và tổng ròng 3/5/10 phiên; điểm cộng/trừ được chuẩn hóa theo tổng GTGD 5 phiên.
- Bán ròng 5 phiên có tỷ lệ từ 2% GTGD 5 phiên trở lên sẽ hạ tín hiệu BUY xuống WAIT; mua ròng chỉ hỗ trợ điểm, không tự tạo lệnh mua.
- Khi thiếu dữ liệu khối ngoại, bot ghi UNAVAILABLE và không giả định là mua ròng hay bán ròng.
- Backtest cắt dữ liệu khối ngoại theo đúng ngày tín hiệu để tránh nhìn trước tương lai.
- Giá realtime chỉ được xác nhận khi timestamp đầy đủ do nguồn cung cấp, nằm trong phiên giao dịch và không quá 300 giây.
- Nếu không xác minh được timestamp, dữ liệu được đánh dấu chưa xác minh hoặc dùng giá D1/phiên gần nhất; không được xem là realtime.
- Dữ liệu D1 cần tối thiểu 200 phiên và được lọc theo thanh khoản.
- Chu kỳ 300 giây là chu kỳ worker yêu cầu quét, không bảo đảm nhà cung cấp cập nhật giá liên tục.
- Lãi/lỗ danh mục là ước tính theo giá tham chiếu, không phải giá khớp lệnh thực tế.
- Dữ liệu SQLite trong volume Docker vẫn có thể mất nếu máy chủ hoặc volume bị xóa/hỏng. Cần sao lưu định kỳ ra nơi lưu trữ độc lập.

## Phát triển và kiểm thử

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
docker build --tag ai-stock-scanner .
```

GitHub Actions tự chạy các bước trên khi có thay đổi trên nhánh làm việc.


## Đọc đúng báo cáo backtest

- `EARLY_BREAKOUT` và `PREPARE` là cảnh báo sớm/nghiên cứu setup, không phải lệnh mua đã xác nhận.
- Chỉ `SWING`, `REVERSAL`, `PULLBACK` có tín hiệu BUY xác nhận mới được tính vào phần giao dịch thực thi giả lập.
- Báo cáo mới xuất riêng `executable_trades.csv` và `early_warning_research.csv`; không gộp hai nhóm để đánh giá hiệu quả mua bán.
- Mô phỏng danh mục dùng vốn tiền mặt 10 triệu đồng, rủi ro mục tiêu 2% vốn/vị thế, tối đa 5 vị thế (mỗi vị thế tối đa 20% vốn), khối lượng theo lô 100 và chỉ nhận giao dịch có stop-loss hợp lệ.
- Nếu không có BUY xác nhận hoặc không đủ điều kiện khối lượng theo lô, báo cáo phải ghi rõ không có giao dịch; không được suy diễn tỷ lệ thắng/lợi nhuận từ cảnh báo sớm.
- Đây vẫn là mô phỏng nghiên cứu trên mẫu đại diện, không phải cam kết lợi nhuận hay xác nhận chiến lược đã sẵn sàng giao dịch thật.

## Adaptive research profile (experimental)

- The scanner now attaches `adaptive_research` assessments to historical strategy evaluations and reports status/score distributions in walk-forward diagnostics.
- The experimental profile separates hard risk gates (liquidity, ATR, reward/risk and strong foreign net selling) from softer technical confirmations. Missing a soft confirmation reduces conviction rather than automatically vetoing a setup.
- Market regime (bullish/neutral/bearish) is computed only from benchmark bars available as of the signal date.
- **Important:** Adaptive v1 is research-only. It never emits an executable BUY and does not change production BUY/WAIT decisions, position sizing, or order permissions. Promote it only after a broader out-of-sample comparison across market regimes, with costs and realistic execution assumptions.
- The baseline strategy remains the production reference so adaptive results can be compared without contaminating the existing strategy.


## Cash-constrained top-five portfolio research

- `top5_walkforward_backtest.py` selects at most five highest-ranked eligible setups at each of 30 point-in-time checkpoints, spaced five trading sessions apart.
- `portfolio_execution.py` then simulates cash reservation/release, duplicate-ticker prevention, at most five concurrent positions, 2% target risk per trade, 20% maximum capital allocation per position, and 100-share board lots.
- Position sizing includes the strategy's estimated slippage plus buy fee, sell fee and sell tax. Net trade outcomes already include the same round-trip costs; the portfolio layer does not subtract them twice.
- Outputs include `portfolio_accepted_trades.csv`, `portfolio_skipped_trades.csv` and `cash_portfolio_simulation` in `report.json`.
- The portfolio drawdown is realized-only because this trade-level report does not contain daily mark-to-market equity. It must not be interpreted as intraday or floating drawdown.
- This is research-only. It does not alter production BUY rules, place orders, or establish that any strategy has positive expected returns.


## Conservative top-3 adaptive research (2026-11-21)

- Market-wide discovery remains enabled, but each checkpoint can select at most three highest-quality candidates; no candidate is forced.
- Candidates below a normalized setup score of 0.65 are excluded before ranking.
- Portfolio sizing applies each candidate's regime/setup risk_factor to the base per-trade risk budget, reducing exposure for lower-conviction setups.
- This adaptive selection is research-only and does not change production scanner BUY/WAIT signals or permit automatic orders.
- A successful code/test run is not evidence of a profitable strategy; require a new out-of-sample report before considering promotion.
