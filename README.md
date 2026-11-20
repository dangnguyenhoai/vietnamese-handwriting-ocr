# Vietnamese Handwriting OCR

> **Dataset access and usage:** The UIT-HWDB line images derived from HANDS-VNOnDB are **not distributed in this repository**. To request access to HANDS-VNOnDB, consult the official [Zenodo record](https://zenodo.org/records/23864366) and its application form. Under the stated Conditions of Use (Version 2.0), the data and derived material are restricted to non-commercial research and must not be redistributed. The local `src/data/UIT_HWDB_line/` path described below is only an expected input directory for users who have independently obtained the data with authorization. Do not commit or redistribute these data files.

Hệ thống nhận dạng chữ viết tay tiếng Việt ở mức dòng ảnh, sử dụng mô hình **CRNN kết hợp CTC Loss**. Dự án bao gồm pipeline chuẩn bị dữ liệu, huấn luyện, đánh giá, dự đoán ảnh đơn, API FastAPI và giao diện web React/Vite.

## Tính năng chính

- Chuẩn bị dữ liệu từ bộ `UIT_HWDB_line`, tách train/validation/test theo folder để hạn chế rò rỉ dữ liệu.
- Lọc mẫu không hợp lệ với CTC dựa trên số timestep sau CNN.
- Xây dựng bộ ký tự tiếng Việt và ánh xạ `char2idx`/`idx2char`.
- Huấn luyện mô hình CRNN-CTC bằng PyTorch.
- Đánh giá bằng các chỉ số CER, WER và Exact Match.
- Dự đoán một ảnh dòng chữ với tùy chọn auto-crop.
- Cung cấp REST API bằng FastAPI.
- Cung cấp giao diện web React/Vite để upload ảnh và xem kết quả OCR.
- Sinh biểu đồ trực quan hóa quá trình huấn luyện và kết quả đánh giá.

## Cấu trúc thư mục

```text
vietnamese-handwriting-ocr/
|-- api/
|   |-- main.py                     # FastAPI OCR service
|   `-- __init__.py
|-- scripts/
|   |-- prepare_data.py             # Quét dữ liệu, tách train/val/test, tạo vocab
|   |-- train_crnn.py               # Huấn luyện CRNN-CTC
|   |-- evaluate.py                 # Đánh giá checkpoint trên test set
|   |-- predict_line.py             # Dự đoán một ảnh dòng chữ
|   |-- crop_line_image.py          # Auto-crop ảnh dòng chữ
|   |-- check_split.py              # Kiểm tra split dữ liệu
|   |-- find_ctc_invalid.py         # Tìm mẫu không hợp lệ với CTC
|   `-- test_*.py                   # Một số script kiểm thử nhanh
|-- src/
|   |-- data/                       # Dataset, collate, transform
|   |-- evaluate/                   # Metrics và hàm đánh giá
|   |-- inference/                  # Logic dự đoán OCR
|   |-- models/                     # CRNN-CTC và CTC decoder
|   |-- train/                      # Trainer
|   `-- utils/                      # Tiện ích xử lý ảnh/text
|-- visualization/
|   |-- visualization.py            # Sinh biểu đồ từ log/kết quả
|   `-- figures/                    # Biểu đồ kết quả
|-- web/
|   |-- src/
|   |-- package.json
|   `-- vite.config.js
|-- outputs/                        # Kết quả sinh ra khi chạy local
|-- requirements.txt
`-- README.md
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
|-- train_data/
|   `-- <folder_id>/
|       |-- 1.jpg
|       |-- 2.jpg
|       `-- label.json
`-- test_data/
    `-- <folder_id>/
        |-- 1.jpg
        |-- 2.jpg
        `-- label.json
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

Script này tạo các file trong `outputs/ocr_outputs/`:

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

Kết quả test gần nhất trong project local:

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

## Kết quả và trực quan hóa

Một số chỉ số từ quá trình huấn luyện gần nhất:

| Chỉ số | Giá trị |
| --- | ---: |
| Số epoch đã chạy | 73 |
| Train loss cuối | 0.0164 |
| Validation loss cuối | 0.2854 |
| Best validation loss | 0.2456 tại epoch 35 |
| Best validation CER | 0.0416 tại epoch 53 |
| Best validation WER | 0.1297 tại epoch 73 |
| Best validation Exact Match | 0.2843 tại epoch 68 |
| Thời gian huấn luyện | khoảng 108.27 phút |
| Thời gian trung bình mỗi epoch | khoảng 88.99 giây |

Các biểu đồ dưới đây được lấy từ `visualization/figures/`.

### Tổng quan metrics

![Tổng quan metrics trong quá trình huấn luyện](visualization/figures/metrics_overview.png)

### Loss trong quá trình huấn luyện

![Diễn biến loss trong quá trình huấn luyện](visualization/figures/loss_curve.png)

### CER/WER trên validation

![Diễn biến CER và WER trên tập validation](visualization/figures/cer_wer_curve.png)

### So sánh validation và test

![So sánh validation và test](visualization/figures/validation_test_comparison.png)

Có thể sinh lại biểu đồ bằng lệnh:

```powershell
python visualization/visualization.py
```

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

Các thư mục/file như `outputs/`, checkpoint `.pth`, cache Python và virtual environment được ignore trong Git. Khi clone project mới, cần chạy lại bước chuẩn bị dữ liệu và huấn luyện, hoặc tự đặt checkpoint/vocab đúng vị trí trước khi chạy API, web hoặc predict.

## Tác giả

Dang Nguyen Hoai
