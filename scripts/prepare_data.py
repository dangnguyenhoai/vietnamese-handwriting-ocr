from pathlib import Path
import json
import random
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm


DATA_ROOT = Path("src/data/UIT_HWDB_line")
TRAIN_DIR = DATA_ROOT / "train_data"
TEST_DIR = DATA_ROOT / "test_data"

OUTPUT_DIR = Path("outputs/ocr_outputs")

TRAIN_RATIO = 0.9
SEED = 42

IMG_HEIGHT = 64
CNN_WIDTH_DOWNSAMPLE = 4

# Không lấy charset từ val/test. Đây là cấu hình sạch hơn cho báo cáo.
# base_plus_train = bộ ký tự tiếng Việt định nghĩa trước + ký tự xuất hiện trong train.
CHARSET_MODE = "base_plus_train"


BASE_VIETNAMESE_CHARS = (
    " "
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
    "đ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
    "Đ"
    ".,;:!?()[]{}'\"-_/\\@#$%&*+=<>|`~…"
)


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def is_valid_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # width, height


def compute_resized_width(width: int, height: int, target_height: int = IMG_HEIGHT) -> int:
    if height <= 0:
        return 0

    scale = target_height / float(height)
    resized_width = int(round(width * scale))
    return max(1, resized_width)


def count_adjacent_repeats(text: str) -> int:
    """
    CTC cần thêm blank giữa hai ký tự giống nhau đứng cạnh nhau.
    Ví dụ: 'aa' cần tối thiểu 3 timestep: a blank a.
    """
    repeats = 0
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            repeats += 1
    return repeats


def required_ctc_timesteps(text: str) -> int:
    return len(text) + count_adjacent_repeats(text)


def build_sample(image_path: Path, label: str, split_name: str, source_folder: str, image_name: str):
    width, height = get_image_size(image_path)
    resized_width = compute_resized_width(width, height)
    input_timesteps = max(1, resized_width // CNN_WIDTH_DOWNSAMPLE)
    ctc_required = required_ctc_timesteps(label)

    return {
        "image_path": str(image_path),
        "label": label,
        "split": split_name,
        "source_folder": source_folder,
        "image_name": image_name,
        "orig_width": width,
        "orig_height": height,
        "resized_width": resized_width,
        "input_timesteps": input_timesteps,
        "label_length": len(label),
        "ctc_required_timesteps": ctc_required,
        "ctc_valid": ctc_required <= input_timesteps,
    }


def build_samples_from_split(split_dir: Path, split_name: str):
    samples = []

    stats = {
        "folders": 0,
        "raw_items": 0,
        "valid_samples_before_ctc_filter": 0,
        "valid_samples_after_ctc_filter": 0,
        "missing_image": 0,
        "empty_label": 0,
        "invalid_image": 0,
        "missing_label_file": 0,
        "ctc_invalid": 0,
    }

    subfolders = sorted(
        [p for p in split_dir.iterdir() if p.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else x.name,
    )

    for folder in tqdm(subfolders, desc=f"Scanning {split_name}"):
        stats["folders"] += 1

        label_path = folder / "label.json"
        if not label_path.exists():
            stats["missing_label_file"] += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        if not isinstance(labels, dict):
            continue

        for image_name, label in labels.items():
            stats["raw_items"] += 1

            label = normalize_text(label)
            if not label:
                stats["empty_label"] += 1
                continue

            image_path = folder / image_name
            if not image_path.exists():
                stats["missing_image"] += 1
                continue

            if not is_valid_image(image_path):
                stats["invalid_image"] += 1
                continue

            sample = build_sample(
                image_path=image_path,
                label=label,
                split_name=split_name,
                source_folder=folder.name,
                image_name=image_name,
            )

            stats["valid_samples_before_ctc_filter"] += 1

            if not sample["ctc_valid"]:
                stats["ctc_invalid"] += 1
                continue

            samples.append(sample)

    stats["valid_samples_after_ctc_filter"] = len(samples)
    return samples, stats


def split_train_val_by_folder(train_samples_raw: List[Dict], train_ratio: float, seed: int):
    """
    Chia theo source_folder, không chia random từng ảnh.
    Nếu chia từng ảnh thì cùng nét chữ/source có thể lọt cả train và val.
    Metric khi đó dễ bị ảo.
    """
    folder_to_samples = defaultdict(list)

    for sample in train_samples_raw:
        folder_to_samples[sample["source_folder"]].append(sample)

    folders = sorted(folder_to_samples.keys())
    random.Random(seed).shuffle(folders)

    n_train_folders = int(len(folders) * train_ratio)
    train_folders = set(folders[:n_train_folders])
    val_folders = set(folders[n_train_folders:])

    train_samples = []
    val_samples = []

    for folder, folder_samples in folder_to_samples.items():
        if folder in train_folders:
            for sample in folder_samples:
                sample = sample.copy()
                sample["split"] = "train"
                train_samples.append(sample)
        else:
            for sample in folder_samples:
                sample = sample.copy()
                sample["split"] = "val"
                val_samples.append(sample)

    return train_samples, val_samples, sorted(train_folders), sorted(val_folders)


def build_charset(train_samples: List[Dict]):
    chars = set()
    counter = Counter()

    if CHARSET_MODE == "base_plus_train":
        for ch in BASE_VIETNAMESE_CHARS:
            chars.add(ch)

    for sample in train_samples:
        for ch in sample["label"]:
            chars.add(ch)
            counter[ch] += 1

    chars = sorted(chars)

    # 0 là blank token cho CTC.
    char2idx = {"": 0}
    idx2char = {"0": ""}

    for idx, ch in enumerate(chars, start=1):
        char2idx[ch] = idx
        idx2char[str(idx)] = ch

    return char2idx, idx2char, counter


def filter_unknown_chars(samples: List[Dict], char2idx: Dict[str, int], split_name: str):
    kept = []
    rejected = []

    for sample in samples:
        unknown_chars = sorted(set(ch for ch in sample["label"] if ch not in char2idx))

        if unknown_chars:
            bad = sample.copy()
            bad["unknown_chars"] = unknown_chars
            bad["split"] = split_name
            rejected.append(bad)
        else:
            kept.append(sample)

    return kept, rejected


def compute_label_stats(samples: List[Dict]):
    lengths = [len(sample["label"]) for sample in samples]
    widths = [sample["resized_width"] for sample in samples]
    timesteps = [sample["input_timesteps"] for sample in samples]

    if not samples:
        return {
            "num_samples": 0,
            "min_label_length": 0,
            "max_label_length": 0,
            "avg_label_length": 0,
            "min_resized_width": 0,
            "max_resized_width": 0,
            "avg_resized_width": 0,
            "min_input_timesteps": 0,
            "max_input_timesteps": 0,
            "avg_input_timesteps": 0,
        }

    return {
        "num_samples": len(samples),
        "min_label_length": min(lengths),
        "max_label_length": max(lengths),
        "avg_label_length": sum(lengths) / len(lengths),
        "min_resized_width": min(widths),
        "max_resized_width": max(widths),
        "avg_resized_width": sum(widths) / len(widths),
        "min_input_timesteps": min(timesteps),
        "max_input_timesteps": max(timesteps),
        "avg_input_timesteps": sum(timesteps) / len(timesteps),
    }


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 80)
    print("PREPARE DATA - CLEAN VERSION")
    print("=" * 80)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy train dir: {TRAIN_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy test dir: {TEST_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Train dir:", TRAIN_DIR)
    print("Test dir:", TEST_DIR)
    print("Output dir:", OUTPUT_DIR)
    print("Image height:", IMG_HEIGHT)
    print("CNN width downsample:", CNN_WIDTH_DOWNSAMPLE)
    print("Charset mode:", CHARSET_MODE)

    print("\nBuilding raw train samples...")
    train_samples_raw, train_scan_stats = build_samples_from_split(TRAIN_DIR, "train_raw")

    print("\nBuilding raw test samples...")
    test_samples_raw, test_scan_stats = build_samples_from_split(TEST_DIR, "test")

    print("\nSplitting train/val by source_folder...")
    train_samples, val_samples, train_folders, val_folders = split_train_val_by_folder(
        train_samples_raw,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )

    overlap = set(train_folders) & set(val_folders)
    if overlap:
        raise RuntimeError(f"Folder leakage giữa train và val: {sorted(overlap)[:10]}")

    print("Train folders:", len(train_folders))
    print("Val folders:", len(val_folders))
    print("Train samples before unknown-char filter:", len(train_samples))
    print("Val samples before unknown-char filter:", len(val_samples))
    print("Test samples before unknown-char filter:", len(test_samples_raw))

    print("\nBuilding charset from train/base only...")
    char2idx, idx2char, train_char_counter = build_charset(train_samples)

    train_samples, train_unknown = filter_unknown_chars(train_samples, char2idx, "train")
    val_samples, val_unknown = filter_unknown_chars(val_samples, char2idx, "val")
    test_samples, test_unknown = filter_unknown_chars(test_samples_raw, char2idx, "test")

    print("Charset size including blank:", len(char2idx))
    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))
    print("Test samples:", len(test_samples))

    print("Rejected unknown train samples:", len(train_unknown))
    print("Rejected unknown val samples:", len(val_unknown))
    print("Rejected unknown test samples:", len(test_unknown))

    data_stats = {
        "train_scan": train_scan_stats,
        "test_scan": test_scan_stats,
        "train": compute_label_stats(train_samples),
        "val": compute_label_stats(val_samples),
        "test": compute_label_stats(test_samples),
        "charset_size_including_blank": len(char2idx),
        "charset_mode": CHARSET_MODE,
        "top_30_train_chars": train_char_counter.most_common(30),
        "train_ratio_by_folder": TRAIN_RATIO,
        "seed": SEED,
        "img_height": IMG_HEIGHT,
        "cnn_width_downsample": CNN_WIDTH_DOWNSAMPLE,
        "num_train_folders": len(train_folders),
        "num_val_folders": len(val_folders),
        "folder_overlap_train_val": len(overlap),
        "unknown_char_rejected": {
            "train": len(train_unknown),
            "val": len(val_unknown),
            "test": len(test_unknown),
        },
    }

    print("\nData stats:")
    print(json.dumps(data_stats, ensure_ascii=False, indent=2))

    print("\nSaving outputs...")
    save_json(train_samples_raw, OUTPUT_DIR / "train_samples_raw.json")
    save_json(train_samples, OUTPUT_DIR / "train_samples.json")
    save_json(val_samples, OUTPUT_DIR / "val_samples.json")
    save_json(test_samples, OUTPUT_DIR / "test_samples.json")

    save_json(train_folders, OUTPUT_DIR / "train_folders.json")
    save_json(val_folders, OUTPUT_DIR / "val_folders.json")

    save_json(train_unknown, OUTPUT_DIR / "rejected_unknown_train.json")
    save_json(val_unknown, OUTPUT_DIR / "rejected_unknown_val.json")
    save_json(test_unknown, OUTPUT_DIR / "rejected_unknown_test.json")

    save_json(char2idx, OUTPUT_DIR / "char2idx.json")
    save_json(idx2char, OUTPUT_DIR / "idx2char.json")
    save_json(data_stats, OUTPUT_DIR / "data_stats.json")

    print("\nDone.")
    print("Saved to:", OUTPUT_DIR)

    print("\nFiles:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print("-", path)


if __name__ == "__main__":
    main()