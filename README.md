# Vietnamese Handwriting OCR

Hệ thống nhận dạng chữ viết tay tiếng Việt ở mức dòng ảnh, sử dụng mô hình CRNN kết hợp CTC Loss. Project bao gồm pipeline chuẩn bị dữ liệu, huấn luyện, đánh giá, dự đoán ảnh đơn, API FastAPI và giao diện web React/Vite.

## Tính năng chính

- Chuẩn bị dữ liệu từ bộ `UIT_HWDB_line`, tách train/validation theo folder để giảm rò rỉ dữ liệu.
- Lọc mẫu không hợp lệ với CTC dựa trên số timestep sau CNN.
- Xây dựng bộ ký tự tiếng Việt và ánh xạ `char2idx`/`idx2char`.
- Huấn luyện mô hình CRNN-CTC bằng PyTorch.
- Đánh giá bằng CER, WER và Exact Match.
- Dự đoán một ảnh dòng chữ với tùy chọn auto-crop.
- Cung cấp REST API bằng FastAPI.
- Cung cấp giao diện web bằng React/Vite để upload ảnh và xem kết quả OCR.

## Cấu trúc thư mục

```text
vietnamese-handwriting-ocr/
├── api/
│   └── main.py                  # FastAPI OCR service
├── scripts/
│   ├── prepare_data.py           # Quét dữ liệu, tách train/val/test, tạo vocab
│   ├── train_crnn.py             # Huấn luyện CRNN-CTC
│   ├── evaluate.py               # Đánh giá checkpoint trên test set
│   ├── predict_line.py           # Dự đoán một ảnh dòng chữ
│   ├── crop_line_image.py        # Auto-crop ảnh dòng chữ
│   ├── check_split.py            # Kiểm tra split dữ liệu
│   └── find_ctc_invalid.py       # Tìm mẫu không hợp lệ với CTC
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── collate.py
│   │   ├── transforms.py
│   │   └── UIT_HWDB_line/        # Dataset local
│   ├── evaluate/
│   │   └── metrics.py
│   └── models/
│       ├── crnn_ctc.py
│       └── decoder.py
├── web/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── outputs/                      # Kết quả sinh ra khi chạy, không commit
├── requirements.txt
└── README.md
```

## Yêu cầu

- Python 3.8 trở lên.
- `pip` và virtual environment.
- Node.js/npm nếu chạy giao diện web.
- GPU CUDA là tùy chọn, nhưng nên dùng khi huấn luyện.

## Cài đặt

Clone repository và tạo môi trường Python:

```powershell
git clone https://github.com/dangnguyenhoai/vietnamese-handwriting-ocr.git
cd vietnamese-handwriting-ocr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu PowerShell chặn script activate, chạy trong phiên hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Dữ liệu

Pipeline hiện tại đọc dữ liệu tại:

```text
src/data/UIT_HWDB_line/
├── train_data/
│   └── <folder_id>/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── label.json
└── test_data/
    └── <folder_id>/
        ├── 1.jpg
        ├── 2.jpg
        └── label.json
```

Mỗi file `label.json` là một object ánh xạ tên ảnh sang nội dung chữ viết tay:

```json
{
  "1.jpg": "nội dung dòng chữ",
  "2.jpg": "nội dung dòng chữ khác"
}
```

## Quy trình chạy

### 1. Chuẩn bị dữ liệu

```powershell
python scripts/prepare_data.py
```

Script này tạo các file trong `outputs/ocr_outputs/`, gồm:

- `train_samples.json`
- `val_samples.json`
- `test_samples.json`
- `char2idx.json`
- `idx2char.json`
- `data_stats.json`

### 2. Huấn luyện mô hình

```powershell
python scripts/train_crnn.py
```

Checkpoint được lưu tại:

```text
outputs/checkpoints/latest_crnn_ctc.pth
outputs/checkpoints/best_crnn_ctc.pth
```

Log huấn luyện được lưu tại:

```text
outputs/logs/history.json
```

### 3. Đánh giá trên test set

```powershell
python scripts/evaluate.py
```

Kết quả được lưu tại:

```text
outputs/eval/test_results.json
outputs/eval/test_predictions.json
```

Kết quả chạy gần nhất trong project local:

```text
Test CER         : 0.0781
Test WER         : 0.2183
Test Exact Match : 0.1294
Num samples      : 201
```

### 4. Dự đoán một ảnh dòng chữ

```powershell
python scripts/predict_line.py --image src/data/Real_line/1.jpg
```

Tắt auto-crop nếu muốn dùng trực tiếp ảnh gốc:

```powershell
python scripts/predict_line.py --image src/data/Real_line/1.jpg --no-auto-crop
```

Chọn ảnh crop dùng cho OCR:

```powershell
python scripts/predict_line.py --image src/data/Real_line/1.jpg --crop-use gray
```

Các lựa chọn `--crop-use` hiện có: `gray`, `bw`, `color`.

## Chạy API

API cần các file sau tồn tại trước khi start:

```text
outputs/ocr_outputs/char2idx.json
outputs/ocr_outputs/idx2char.json
outputs/checkpoints/best_crnn_ctc.pth
```

Start FastAPI:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Các endpoint chính:

- `GET /health`: kiểm tra trạng thái model.
- `GET /labels`: xem bộ ký tự.
- `POST /predict/line`: upload ảnh và OCR.
- `POST /predict/line-url`: OCR ảnh từ URL.
- `GET /docs`: Swagger UI.

## Chạy giao diện web

Mở terminal mới:

```powershell
cd web
npm install
npm run dev
```

Frontend chạy mặc định tại:

```text
http://localhost:5173
```

Script `web/scripts/write-api-env.mjs` tự ghi `web/.env.local` với địa chỉ API dạng `http://<ipv4>:8000`.

## Ghi chú về file sinh ra

Các thư mục/file như `outputs/`, checkpoint `.pth`, cache Python và virtual environment được ignore trong Git. Khi clone project mới, cần chạy lại bước chuẩn bị dữ liệu và huấn luyện hoặc tự đặt checkpoint/vocab đúng vị trí trước khi chạy API, web hoặc predict.

## Tác giả

Dang Nguyen Hoai
