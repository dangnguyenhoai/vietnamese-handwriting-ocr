# Vietnamese Handwriting OCR

Dự án nhận dạng chữ viết tay tiếng Việt sử dụng Deep Learning.

## Mô tả

Dự án này xây dựng một hệ thống OCR (Optical Character Recognition) chuyên biệt để nhận dạng chữ viết tay tiếng Việt. Sử dụng các mô hình Deep Learning hiện đại để đạt độ chính xác cao.

## Cấu trúc Dự án

```
vietnamese-handwriting-ocr/
│
├── src/                    # Thư mục chứa code chính
│   ├── data/              # Xử lý dữ liệu
│   ├── models/            # Định nghĩa mô hình
│   ├── train/             # Huấn luyện mô hình
│   ├── evaluate/          # Đánh giá mô hình
│   └── inference/         # Dự đoán trên dữ liệu mới
│
├── app.py                 # Ứng dụng chính (Streamlit/Flask)
├── train.py               # Script huấn luyện
├── evaluate.py            # Script đánh giá
├── config.py              # Cấu hình dự án
├── requirements.txt       # Các thư viện phụ thuộc
├── .gitignore            # Git ignore file
└── README.md             # Tài liệu này
```

## Yêu cầu

- Python 3.8+
- pip hoặc conda

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd vietnamese-handwriting-ocr
```

2. Tạo virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

## Sử dụng

### Huấn luyện mô hình
```bash
python train.py
```

### Đánh giá mô hình
```bash
python evaluate.py
```

### Chạy ứng dụng
```bash
python app.py
```

## Dữ liệu

- Đặt dữ liệu huấn luyện vào `data/raw/`
- Dữ liệu được xử lý sẽ được lưu vào `data/processed/`

## Mô hình

- Mô hình được lưu trong thư mục `models/`
- Chi tiết mô hình được định nghĩa trong `src/models/`

## Kết quả

- Kết quả huấn luyện được lưu trong `logs/`
- Kết quả đánh giá được lưu trong `results/`

## Tác giả

[Tên của bạn]

## License

[Chọn license phù hợp, ví dụ: MIT]
