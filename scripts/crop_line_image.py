import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_text_bbox(gray, pad_x=25, pad_y=35, dilate_iter=1):
    """
    Tìm bounding box vùng chữ trong ảnh grayscale.
    Dùng threshold chỉ để phát hiện vùng chữ, không dùng để OCR trực tiếp.
    """
    h, w = gray.shape

    # Làm mượt nhẹ để threshold ổn hơn
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Chữ đen -> trắng, nền sáng -> đen
    _, binary_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Nối các nét chữ nhẹ để bbox ổn định
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.dilate(binary_inv, kernel, iterations=dilate_iter)

    coords = cv2.findNonZero(merged)

    if coords is None:
        return 0, 0, w, h, binary_inv, merged

    x, y, bw, bh = cv2.boundingRect(coords)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    return x1, y1, x2, y2, binary_inv, merged


def make_binary_white_background(gray_crop):
    """
    Chuyển crop grayscale sang ảnh trắng đen:
    chữ đen, nền trắng.
    """
    blur = cv2.GaussianBlur(gray_crop, (3, 3), 0)

    _, bw = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return bw


def crop_line_image(
    image_path,
    out_dir="outputs/cropped_lines",
    pad_x=25,
    pad_y=35,
    dilate_iter=1,
):
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x1, y1, x2, y2, binary_inv, merged = detect_text_bbox(
        gray,
        pad_x=pad_x,
        pad_y=pad_y,
        dilate_iter=dilate_iter,
    )

    # Crop từ ảnh gốc và ảnh xám
    cropped_color = img[y1:y2, x1:x2]
    cropped_gray = gray[y1:y2, x1:x2]

    # Tạo bản trắng đen
    cropped_bw = make_binary_white_background(cropped_gray)

    # Vẽ bbox để kiểm tra
    bbox_vis = img.copy()
    cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    stem = image_path.stem

    paths = {
        "gray": out_dir / f"{stem}_crop_gray.jpg",
        "bw": out_dir / f"{stem}_crop_bw.jpg",
        "color": out_dir / f"{stem}_crop_color.jpg",
        "bbox": out_dir / f"{stem}_bbox.jpg",
        "binary_inv": out_dir / f"{stem}_binary_inv.jpg",
        "merged": out_dir / f"{stem}_merged.jpg",
    }

    cv2.imwrite(str(paths["gray"]), cropped_gray)
    cv2.imwrite(str(paths["bw"]), cropped_bw)
    cv2.imwrite(str(paths["color"]), cropped_color)
    cv2.imwrite(str(paths["bbox"]), bbox_vis)
    cv2.imwrite(str(paths["binary_inv"]), binary_inv)
    cv2.imwrite(str(paths["merged"]), merged)

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Auto crop one handwritten line image and convert it to black-white."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn ảnh đầu vào."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/cropped_lines",
        help="Thư mục lưu ảnh crop."
    )

    parser.add_argument(
        "--pad-x",
        type=int,
        default=15,
        help="Padding trái/phải quanh vùng chữ."
    )

    parser.add_argument(
        "--pad-y",
        type=int,
        default=25,
        help="Padding trên/dưới quanh vùng chữ. Tăng nếu bị cắt mất dấu."
    )

    parser.add_argument(
        "--dilate-iter",
        type=int,
        default=1,
        help="Số lần dilate để nối nét khi phát hiện bbox."
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CROP LINE IMAGE")
    print("=" * 80)
    print("Image:", args.image)
    print("Output dir:", args.out_dir)
    print("pad_x:", args.pad_x)
    print("pad_y:", args.pad_y)
    print("dilate_iter:", args.dilate_iter)

    paths = crop_line_image(
        image_path=args.image,
        out_dir=args.out_dir,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        dilate_iter=args.dilate_iter,
    )

    print("\nSaved files:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()