import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.crnn_ctc import CRNNCTC
from src.models.decoder import CTCGreedyDecoder


OUTPUT_DIR = Path("outputs/ocr_outputs")
CHECKPOINT_DIR = Path("outputs/checkpoints")

DEFAULT_CHAR2IDX = OUTPUT_DIR / "char2idx.json"
DEFAULT_IDX2CHAR = OUTPUT_DIR / "idx2char.json"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_crnn_ctc.pth"

IMAGE_HEIGHT = 64


def detect_text_bbox(gray, pad=20):
    """
    Tìm bounding box chứa chữ bằng Otsu threshold + tìm non-zero.
    Chỉ dùng để crop, không dùng trực tiếp ảnh binary cho model.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Chữ đen -> trắng, nền sáng -> đen
    binary_inv = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Nối nét chữ nhẹ để bbox ổn hơn
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.dilate(binary_inv, kernel, iterations=1)

    coords = cv2.findNonZero(merged)

    h, w = gray.shape

    if coords is None:
        return 0, 0, w, h, binary_inv, merged

    x, y, bw, bh = cv2.boundingRect(coords)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    return x1, y1, x2, y2, binary_inv, merged


def make_white_background(gray_crop, white_thresh=210):
    """
    Làm nền trắng hơn nhưng vẫn giữ nét chữ từ ảnh grayscale gốc.
    Không biến toàn bộ ảnh thành binary cứng.
    """
    result = gray_crop.copy()

    # Pixel nào sáng thì đẩy thành trắng
    result[result >= white_thresh] = 255

    return result


def preprocess_autocrop(
    image_path,
    image_height=64,
    pad=20,
    white_bg=True,
    white_thresh=210,
    debug_dir="outputs/debug_autocrop",
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Tìm bbox có chữ
    x1, y1, x2, y2, binary_inv, merged = detect_text_bbox(gray, pad=pad)

    # 2) Crop từ grayscale gốc
    cropped = gray[y1:y2, x1:x2]

    # 3) Làm nền trắng hơn
    if white_bg:
        processed = make_white_background(cropped, white_thresh=white_thresh)
    else:
        processed = cropped.copy()

    # 4) Resize giữ tỉ lệ
    original_h, original_w = processed.shape
    new_h = image_height
    new_w = int(original_w * new_h / original_h)
    new_w = max(1, new_w)

    resized = cv2.resize(
        processed,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Lưu debug
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem

        # Vẽ bbox lên ảnh gốc để kiểm tra
        bbox_vis = img.copy()
        cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(debug_dir / f"{stem}_01_gray.jpg"), gray)
        cv2.imwrite(str(debug_dir / f"{stem}_02_binary_inv.jpg"), binary_inv)
        cv2.imwrite(str(debug_dir / f"{stem}_03_merged.jpg"), merged)
        cv2.imwrite(str(debug_dir / f"{stem}_04_bbox.jpg"), bbox_vis)
        cv2.imwrite(str(debug_dir / f"{stem}_05_cropped.jpg"), cropped)
        cv2.imwrite(str(debug_dir / f"{stem}_06_processed.jpg"), processed)
        cv2.imwrite(str(debug_dir / f"{stem}_07_resized.jpg"), resized)

    # 5) Chuẩn hóa đúng như model cũ
    image_np = resized.astype(np.float32) / 255.0

    # [H, W] -> [1, H, W]
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)

    # [0,1] -> [-1,1]
    image_tensor = (image_tensor - 0.5) / 0.5

    # [1,H,W] -> [1,1,H,W]
    image_tensor = image_tensor.unsqueeze(0)

    image_width = torch.tensor([new_w], dtype=torch.long)

    return image_tensor, image_width


def load_model(checkpoint_path, num_classes, device):
    model = CRNNCTC(
        num_classes=num_classes,
        input_channels=1,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.3
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint val_metrics:", checkpoint.get("val_metrics"))

    return model


def predict_line_autocrop(
    image_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    char2idx_path=DEFAULT_CHAR2IDX,
    idx2char_path=DEFAULT_IDX2CHAR,
    image_height=IMAGE_HEIGHT,
    pad=20,
    white_bg=True,
    white_thresh=210,
    debug_dir="outputs/debug_autocrop",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    char2idx_path = Path(char2idx_path)
    idx2char_path = Path(idx2char_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    if not char2idx_path.exists():
        raise FileNotFoundError(f"Không tìm thấy char2idx: {char2idx_path}")
    if not idx2char_path.exists():
        raise FileNotFoundError(f"Không tìm thấy idx2char: {idx2char_path}")

    with open(char2idx_path, "r", encoding="utf-8") as f:
        char2idx = json.load(f)

    with open(idx2char_path, "r", encoding="utf-8") as f:
        idx2char = json.load(f)

    model = load_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(char2idx),
        device=device
    )

    decoder = CTCGreedyDecoder(
        idx2char=idx2char,
        blank_idx=0
    )

    image_tensor, image_width = preprocess_autocrop(
        image_path=image_path,
        image_height=image_height,
        pad=pad,
        white_bg=white_bg,
        white_thresh=white_thresh,
        debug_dir=debug_dir,
    )

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prediction = decoder.decode_logits(logits)[0]

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description="Predict handwritten Vietnamese line with auto-crop + white background."
    )

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--char2idx", type=str, default=str(DEFAULT_CHAR2IDX))
    parser.add_argument("--idx2char", type=str, default=str(DEFAULT_IDX2CHAR))
    parser.add_argument("--image-height", type=int, default=IMAGE_HEIGHT)

    parser.add_argument(
        "--pad",
        type=int,
        default=20,
        help="Padding thêm quanh vùng chữ khi crop."
    )

    parser.add_argument(
        "--no-white-bg",
        action="store_true",
        help="Tắt làm nền trắng."
    )

    parser.add_argument(
        "--white-thresh",
        type=int,
        default=210,
        help="Ngưỡng pixel để đẩy nền về trắng. 200-230 thường ổn."
    )

    parser.add_argument(
        "--debug-dir",
        type=str,
        default="outputs/debug_autocrop",
        help="Folder lưu ảnh debug."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PREDICT LINE AUTOCROP")
    print("=" * 80)
    print("Using device:", device)
    print("Image:", args.image)
    print("pad:", args.pad)
    print("white_bg:", not args.no_white_bg)
    print("white_thresh:", args.white_thresh)

    pred = predict_line_autocrop(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        char2idx_path=args.char2idx,
        idx2char_path=args.idx2char,
        image_height=args.image_height,
        pad=args.pad,
        white_bg=not args.no_white_bg,
        white_thresh=args.white_thresh,
        debug_dir=args.debug_dir,
        device=device,
    )

    print("\nPrediction:")
    print(pred)
    print("\nDebug images saved to:", args.debug_dir)


if __name__ == "__main__":
    main()