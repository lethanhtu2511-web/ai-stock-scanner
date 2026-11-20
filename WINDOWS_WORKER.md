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

Tạo file `.env` cục bộ ngay cạnh `worker.py` với các biến sau (dùng đúng URL và secret trong Supabase Project Settings):

```dotenv
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SECRET_KEY=YOUR_SECRET_KEY
SCAN_INTERVAL=300
```

`python-dotenv` sẽ tự nạp `.env`. Không đưa file này lên GitHub; `.env` đã được loại khỏi Git. Không dán secret vào cuộc trò chuyện.

## Chạy
Nhấp đúp `run_worker_windows.bat`, hoặc chạy:

```powershell
.\.venv\Scripts\python.exe worker.py
```

Giữ cửa sổ worker mở trong giờ giao dịch. Nếu máy ngủ, tắt hoặc mất mạng, worker sẽ ngừng cập nhật.

## Giới hạn hiện tại cần hiểu đúng
Worker hiện tại chạy chu kỳ quét theo `SCAN_INTERVAL` (mặc định 300 giây), không phải luồng tick 1–5 giây. KBS public board không có cam kết SLA/độ trễ trong mã hiện tại. Không được đổi nhãn thành realtime nếu timestamp nguồn không xác minh được.

Chế độ `continuous` / `on_demand` được lưu ở `runtime_control` trên Supabase. Worker kiểm tra điều khiển khoảng 10 giây/lần; chế độ `on_demand` chỉ quét khi giao diện gửi yêu cầu. Trang Streamlit và worker phải cùng dùng đúng Supabase project và secret server-side. Không đưa secret vào mã nguồn.
