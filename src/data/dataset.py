from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class OCRLineDataset(Dataset):
    """
    Dataset cho bài toán nhận diện chữ viết tay tiếng Việt mức dòng.
    """

    def __init__(self, samples_path, char2idx_path, image_height=64):
        self.samples_path = Path(samples_path)
        self.char2idx_path = Path(char2idx_path)
        self.image_height = image_height

        if not self.samples_path.exists():
            raise FileNotFoundError(f"Không tìm thấy samples file: {self.samples_path}")

        if not self.char2idx_path.exists():
            raise FileNotFoundError(f"Không tìm thấy char2idx file: {self.char2idx_path}")

        with open(self.samples_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        with open(self.char2idx_path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)

    def __len__(self):
        return len(self.samples)

    def encode_text(self, text):
        encoded = []

        for ch in text:
            if ch not in self.char2idx:
                raise KeyError(f"Ký tự không có trong char2idx: {repr(ch)}")

            encoded.append(self.char2idx[ch])

        return encoded

    def load_image(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        image = Image.open(image_path).convert("L")

        original_w, original_h = image.size

        new_h = self.image_height
        new_w = int(original_w * new_h / original_h)

        if new_w < 1:
            new_w = 1

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # PIL image: [H, W], 0..255
        image_np = np.array(image, dtype=np.float32)

        # 0..255 -> 0..1
        image_np = image_np / 255.0

        # [H, W] -> [1, H, W]
        image = torch.from_numpy(image_np).unsqueeze(0)

        # 0..1 -> -1..1
        image = (image - 0.5) / 0.5

        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = sample["image_path"]
        text = sample["label"]

        image = self.load_image(image_path)
        encoded = self.encode_text(text)

        label = torch.tensor(encoded, dtype=torch.long)
        label_length = torch.tensor(len(encoded), dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "label_length": label_length,
            "text": text,
            "image_path": image_path,
        }