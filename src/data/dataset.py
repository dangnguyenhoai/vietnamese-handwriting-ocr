from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class OCRLineDataset(Dataset):
    """
    Dataset cho OCR chữ viết tay tiếng Việt mức dòng.

    Output:
    - image: Tensor [1, H, W], normalized về [-1, 1]
    - label: Tensor label đã encode
    - label_length: độ dài label
    - text: nhãn gốc
    - image_path: đường dẫn ảnh
    """

    def __init__(
        self,
        samples_path,
        char2idx_path,
        image_height=64,
        transform=None,
    ):
        self.samples_path = Path(samples_path)
        self.char2idx_path = Path(char2idx_path)
        self.image_height = image_height
        self.transform = transform

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

    def resize_keep_ratio(self, image: Image.Image) -> Image.Image:
        original_w, original_h = image.size

        new_h = self.image_height
        new_w = int(round(original_w * new_h / original_h))
        new_w = max(1, new_w)

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return image

    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image_np = np.array(image, dtype=np.float32)
        image_np = image_np / 255.0

        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        # 0..1 -> -1..1
        image_tensor = (image_tensor - 0.5) / 0.5

        return image_tensor

    def load_image(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

        with Image.open(image_path) as img:
            image = img.convert("L")

        if self.transform is not None:
            image = self.transform(image)

        image = self.resize_keep_ratio(image)
        image = self.pil_to_tensor(image)

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