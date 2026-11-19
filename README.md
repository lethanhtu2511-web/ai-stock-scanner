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
- Bản sao lưu nằm trong volume dùng chung tại `/app/data/backups`; mặc định giữ 14 bản gần nhất.
- Sao lưu dùng SQLite Online Backup API và kiểm tra `PRAGMA integrity_check` trước khi công bố file.
- Bản sao lưu cùng volume giúp khôi phục khi lỗi logic/xóa nhầm dữ liệu, nhưng **không bảo vệ trước mất/hỏng máy chủ hoặc volume**. Cần định kỳ chép bản sao lưu ra máy/nơi lưu trữ độc lập.

Liệt kê bản sao lưu:

```bash
docker compose exec app ls -lh /app/data/backups
```

Trước khi phục hồi, dừng cả app và worker, sau đó chép file backup cần dùng thành database chính. Không phục hồi khi app/worker còn đang chạy.

```bash
docker compose stop app worker
# Thực hiện restore bằng backup_manager.restore_database(..., confirm=True)
docker compose up -d
```

Hàm restore từ chối chạy nếu không truyền xác nhận tường minh và từ chối file backup không vượt qua integrity check.

## Dữ liệu và giới hạn

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
