import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.crnn_ctc import CRNNCTC
from src.models.decoder import CTCGreedyDecoder
from scripts.crop_line_image import crop_line_image


OUTPUT_DIR = Path("outputs/ocr_outputs")
CHECKPOINT_DIR = Path("outputs/checkpoints")

DEFAULT_CHAR2IDX = OUTPUT_DIR / "char2idx.json"
DEFAULT_IDX2CHAR = OUTPUT_DIR / "idx2char.json"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_crnn_ctc.pth"

IMAGE_HEIGHT = 64


def load_image_for_prediction(image_path, image_height=64):
    """
    Preprocess ảnh giống lúc train:
    1. Grayscale
    2. Resize height = 64, giữ tỉ lệ width
    3. Convert numpy -> tensor
    4. Normalize [-1, 1]
    5. Thêm batch dimension: [1, 1, H, W]
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image = Image.open(image_path).convert("L")

    original_w, original_h = image.size

    new_h = image_height
    new_w = int(original_w * new_h / original_h)

    if new_w < 1:
        new_w = 1

    image = image.resize((new_w, new_h), Image.BILINEAR)

    image_np = np.array(image, dtype=np.float32)
    image_np = image_np / 255.0

    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
    image_tensor = (image_tensor - 0.5) / 0.5

    # [1, H, W] -> [1, 1, H, W]
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


def maybe_auto_crop_image(
    image_path,
    auto_crop=True,
    crop_use="gray",
    crop_out_dir="outputs/cropped_lines",
    pad_x=15,
    pad_y=25,
    dilate_iter=1,
):
    """
    Nếu auto_crop=True:
        ảnh gốc -> crop_line_image() -> lấy crop_use làm input OCR.
    Nếu auto_crop=False:
        dùng ảnh gốc.
    """
    image_path = Path(image_path)

    if not auto_crop:
        return image_path

    print("Auto crop: ON")
    print("Crop use:", crop_use)
    print("Crop pad_x:", pad_x)
    print("Crop pad_y:", pad_y)

    crop_paths = crop_line_image(
        image_path=image_path,
        out_dir=crop_out_dir,
        pad_x=pad_x,
        pad_y=pad_y,
        dilate_iter=dilate_iter,
    )

    if crop_use not in crop_paths:
        raise ValueError(
            f"crop_use không hợp lệ: {crop_use}. "
            f"Chỉ nhận một trong: {list(crop_paths.keys())}"
        )

    cropped_image_path = crop_paths[crop_use]

    print("Cropped image used:", cropped_image_path)
    print("Crop debug bbox:", crop_paths.get("bbox"))

    return cropped_image_path


def predict_line(
    image_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    char2idx_path=DEFAULT_CHAR2IDX,
    idx2char_path=DEFAULT_IDX2CHAR,
    image_height=IMAGE_HEIGHT,
    device=None,
    auto_crop=True,
    crop_use="gray",
    crop_out_dir="outputs/cropped_lines",
    pad_x=15,
    pad_y=25,
    dilate_iter=1,
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

    num_classes = len(char2idx)

    model = load_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device
    )

    decoder = CTCGreedyDecoder(
        idx2char=idx2char,
        blank_idx=0
    )

    input_image_path = maybe_auto_crop_image(
        image_path=image_path,
        auto_crop=auto_crop,
        crop_use=crop_use,
        crop_out_dir=crop_out_dir,
        pad_x=pad_x,
        pad_y=pad_y,
        dilate_iter=dilate_iter,
    )

    image_tensor, image_width = load_image_for_prediction(
        image_path=input_image_path,
        image_height=image_height
    )

    image_tensor = image_tensor.to(device)
    image_width = image_width.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prediction = decoder.decode_logits(logits)[0]

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description="Predict text from one handwritten Vietnamese line image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Đường dẫn ảnh dòng cần OCR."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Đường dẫn checkpoint best_crnn_ctc.pth."
    )

    parser.add_argument(
        "--char2idx",
        type=str,
        default=str(DEFAULT_CHAR2IDX),
        help="Đường dẫn char2idx.json."
    )

    parser.add_argument(
        "--idx2char",
        type=str,
        default=str(DEFAULT_IDX2CHAR),
        help="Đường dẫn idx2char.json."
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=IMAGE_HEIGHT,
        help="Chiều cao ảnh sau resize."
    )

    parser.add_argument(
        "--no-auto-crop",
        action="store_true",
        help="Tắt auto crop, dùng ảnh gốc như bản predict_line cũ."
    )

    parser.add_argument(
        "--crop-use",
        type=str,
        default="gray",
        choices=["gray", "bw", "color"],
        help="Chọn ảnh sau crop để OCR: gray, bw hoặc color."
    )

    parser.add_argument(
        "--crop-out-dir",
        type=str,
        default="outputs/cropped_lines",
        help="Folder lưu ảnh crop/debug."
    )

    parser.add_argument(
        "--pad-x",
        type=int,
        default=15,
        help="Padding trái/phải khi auto crop."
    )

    parser.add_argument(
        "--pad-y",
        type=int,
        default=25,
        help="Padding trên/dưới khi auto crop."
    )

    parser.add_argument(
        "--dilate-iter",
        type=int,
        default=1,
        help="Số lần dilate khi phát hiện bbox chữ."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("PREDICT LINE")
    print("=" * 80)
    print("Using device:", device)
    print("Image:", args.image)
    print("Auto crop:", not args.no_auto_crop)

    pred = predict_line(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        char2idx_path=args.char2idx,
        idx2char_path=args.idx2char,
        image_height=args.image_height,
        device=device,
        auto_crop=not args.no_auto_crop,
        crop_use=args.crop_use,
        crop_out_dir=args.crop_out_dir,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        dilate_iter=args.dilate_iter,
    )

    print("\nPrediction:")
    print(pred)


if __name__ == "__main__":
    main()