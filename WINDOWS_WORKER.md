# Chạy AI Stock Scanner Worker trên Windows

## Điều kiện
- Windows bật, không Sleep/Hibernate, có Internet trong giờ giao dịch.
- Python 3.11+.
- Không chạy đồng thời nhiều worker trên cùng một máy.
- Worker chỉ phân tích và cảnh báo, không đặt lệnh.

## Cài đặt lần đầu
Mở PowerShell tại thư mục dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tạo file `.env` cục bộ nếu triển khai có cấu hình môi trường. Không đưa API key hoặc Supabase secret vào GitHub.

## Chạy
Nhấp đúp `run_worker_windows.bat`, hoặc chạy:

```powershell
.\.venv\Scripts\python.exe worker.py
```

Giữ cửa sổ worker mở trong giờ giao dịch. Nếu máy ngủ, tắt hoặc mất mạng, worker sẽ ngừng cập nhật.

## Giới hạn hiện tại cần hiểu đúng
Worker hiện tại chạy chu kỳ quét theo `SCAN_INTERVAL` (mặc định 300 giây), không phải luồng tick 1–5 giây. KBS public board không có cam kết SLA/độ trễ trong mã hiện tại. Không được đổi nhãn thành realtime nếu timestamp nguồn không xác minh được.

Để bật chế độ điều khiển liên tục/ngắt quãng từ điện thoại, cần hoàn tất bảng điều khiển runtime dùng chung trên Supabase và heartbeat của worker. Không được dùng bảng `settings` hiện tại cho chuỗi trạng thái vì cột `value` đang là số. Phần này cần migration riêng và kiểm thử trước khi bật mặc định.
