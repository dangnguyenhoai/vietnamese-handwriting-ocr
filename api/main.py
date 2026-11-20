import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.models.crnn_ctc import CRNNCTC
from src.models.decoder import CTCGreedyDecoder
from scripts.crop_line_image import crop_line_image


OUTPUT_DIR = Path("outputs/ocr_outputs")
CHECKPOINT_DIR = Path("outputs/checkpoints")

DEFAULT_CHAR2IDX = OUTPUT_DIR / "char2idx.json"
DEFAULT_IDX2CHAR = OUTPUT_DIR / "idx2char.json"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_crnn_ctc.pth"

API_UPLOAD_DIR = Path("outputs/api_uploads")
API_CROP_DIR = Path("outputs/api_crops")

IMAGE_HEIGHT = 64


class PredictUrlRequest(BaseModel):
    image_url: str
    auto_crop: bool = True
    crop_use: str = "gray"
    pad_x: int = 15
    pad_y: int = 25
    dilate_iter: int = 1


class OCRService:
    def __init__(
        self,
        checkpoint_path=DEFAULT_CHECKPOINT,
        char2idx_path=DEFAULT_CHAR2IDX,
        idx2char_path=DEFAULT_IDX2CHAR,
        image_height=IMAGE_HEIGHT,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.char2idx_path = Path(char2idx_path)
        self.idx2char_path = Path(idx2char_path)
        self.image_height = image_height

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._check_required_files()
        self._load_vocab()
        self._load_model()

    def _check_required_files(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if not self.char2idx_path.exists():
            raise FileNotFoundError(f"char2idx not found: {self.char2idx_path}")

        if not self.idx2char_path.exists():
            raise FileNotFoundError(f"idx2char not found: {self.idx2char_path}")

    def _load_vocab(self):
        with open(self.char2idx_path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)

        with open(self.idx2char_path, "r", encoding="utf-8") as f:
            self.idx2char = json.load(f)

        self.num_classes = len(self.char2idx)

    def _load_model(self):
        self.model = CRNNCTC(
            num_classes=self.num_classes,
            input_channels=1,
            hidden_size=256,
            num_lstm_layers=2,
            dropout=0.3,
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_info = {
            "checkpoint_path": str(self.checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
            "val_metrics": checkpoint.get("val_metrics"),
        }

        self.decoder = CTCGreedyDecoder(
            idx2char=self.idx2char,
            blank_idx=0,
        )

        print("=" * 80)
        print("OCR MODEL LOADED")
        print("=" * 80)
        print("Device:", self.device)
        print("Num classes:", self.num_classes)
        print("Checkpoint:", self.checkpoint_info)

    def preprocess_image(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")

        original_w, original_h = image.size

        new_h = self.image_height
        new_w = int(original_w * new_h / original_h)
        new_w = max(1, new_w)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        image_np = np.array(image, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        image_tensor = (image_tensor - 0.5) / 0.5
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def maybe_crop(
        self,
        image_path,
        auto_crop=True,
        crop_use="gray",
        pad_x=15,
        pad_y=25,
        dilate_iter=1,
    ):
        image_path = Path(image_path)

        if not auto_crop:
            return image_path, {}

        crop_paths = crop_line_image(
            image_path=image_path,
            out_dir=API_CROP_DIR,
            pad_x=pad_x,
            pad_y=pad_y,
            dilate_iter=dilate_iter,
        )

        if crop_use not in crop_paths:
            raise ValueError(
                f"Invalid crop_use={crop_use}. Must be one of {list(crop_paths.keys())}"
            )

        used_path = crop_paths[crop_use]

        crop_paths_str = {
            key: str(value).replace("\\", "/")
            for key, value in crop_paths.items()
        }

        return used_path, crop_paths_str

    def predict(
        self,
        image_path,
        auto_crop=True,
        crop_use="gray",
        pad_x=15,
        pad_y=25,
        dilate_iter=1,
    ):
        used_image_path, crop_paths = self.maybe_crop(
            image_path=image_path,
            auto_crop=auto_crop,
            crop_use=crop_use,
            pad_x=pad_x,
            pad_y=pad_y,
            dilate_iter=dilate_iter,
        )

        image_tensor = self.preprocess_image(used_image_path).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            prediction = self.decoder.decode_logits(logits)[0]

        return {
            "prediction": prediction,
            "input_image": str(image_path).replace("\\", "/"),
            "used_image": str(used_image_path).replace("\\", "/"),
            "auto_crop": auto_crop,
            "crop_use": crop_use if auto_crop else None,
            "crop_paths": crop_paths,
        }


app = FastAPI(
    title="Vietnamese Handwriting OCR API",
    description="CRNN-CTC API for Vietnamese handwritten line OCR",
    version="1.0.0",
)

# Cho frontend local gọi API.
# Khi deploy thật thì đừng để '*' bừa bãi.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
API_CROP_DIR.mkdir(parents=True, exist_ok=True)

# Cho phép xem ảnh debug/crop qua URL.
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

ocr_service = OCRService()


@app.get("/")
def root():
    return {
        "message": "Vietnamese Handwriting OCR API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(ocr_service.device),
        "num_classes": ocr_service.num_classes,
        "checkpoint": ocr_service.checkpoint_info,
    }


@app.get("/labels")
def labels():
    return {
        "num_classes": ocr_service.num_classes,
        "char2idx": ocr_service.char2idx,
    }


def save_upload_file(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()

    if suffix == "":
        suffix = ".jpg"

    if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension: {suffix}",
        )

    filename = f"{uuid.uuid4().hex}{suffix}"
    save_path = API_UPLOAD_DIR / filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path


@app.post("/predict/line")
def predict_line_upload(
    file: UploadFile = File(...),
    auto_crop: bool = Form(True),
    crop_use: str = Form("gray"),
    pad_x: int = Form(15),
    pad_y: int = Form(25),
    dilate_iter: int = Form(1),
):
    """
    Nhận ảnh upload từ web/app rồi OCR một dòng.
    """
    try:
        image_path = save_upload_file(file)

        result = ocr_service.predict(
            image_path=image_path,
            auto_crop=auto_crop,
            crop_use=crop_use,
            pad_x=pad_x,
            pad_y=pad_y,
            dilate_iter=dilate_iter,
        )

        return {
            "success": True,
            **result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def download_image_from_url(image_url: str) -> Path:
    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot download image: {e}",
        )

    content_type = response.headers.get("content-type", "")

    if "image" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"URL does not return an image. content-type={content_type}",
        )

    suffix = ".jpg"

    if "png" in content_type:
        suffix = ".png"
    elif "webp" in content_type:
        suffix = ".webp"
    elif "bmp" in content_type:
        suffix = ".bmp"

    filename = f"{uuid.uuid4().hex}{suffix}"
    save_path = API_UPLOAD_DIR / filename

    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path


@app.post("/predict/line-url")
def predict_line_url(payload: PredictUrlRequest):
    """
    Nhận URL ảnh rồi OCR.
    Chỉ dùng demo. Không nên mở public endpoint này bừa bãi.
    """
    try:
        image_path = download_image_from_url(payload.image_url)

        result = ocr_service.predict(
            image_path=image_path,
            auto_crop=payload.auto_crop,
            crop_use=payload.crop_use,
            pad_x=payload.pad_x,
            pad_y=payload.pad_y,
            dilate_iter=payload.dilate_iter,
        )

        return {
            "success": True,
            "image_url": payload.image_url,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))