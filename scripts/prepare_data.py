from pathlib import Path
import json
import random
import unicodedata
from PIL import Image
from tqdm import tqdm
from collections import Counter


DATA_ROOT = Path("src/data/UIT_HWDB_line")
TRAIN_DIR = DATA_ROOT / "train_data"
TEST_DIR = DATA_ROOT / "test_data"

OUTPUT_DIR = Path("outputs/ocr_outputs")

TRAIN_RATIO = 0.9
SEED = 42


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.strip()
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


def build_samples_from_split(split_dir: Path, split_name: str):
    samples = []

    stats = {
        "folders": 0,
        "raw_items": 0,
        "valid_samples": 0,
        "missing_image": 0,
        "empty_label": 0,
        "invalid_image": 0,
        "missing_label_file": 0,
    }

    subfolders = sorted(
        [p for p in split_dir.iterdir() if p.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else x.name
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

            samples.append({
                "image_path": str(image_path),
                "label": label,
                "split": split_name,
                "source_folder": folder.name,
                "image_name": image_name
            })

    stats["valid_samples"] = len(samples)
    return samples, stats


def split_train_val(train_samples_raw, train_ratio=0.9, seed=42):
    random.seed(seed)

    samples = train_samples_raw.copy()
    random.shuffle(samples)

    n_train = int(len(samples) * train_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    for sample in train_samples:
        sample["split"] = "train"

    for sample in val_samples:
        sample["split"] = "val"

    return train_samples, val_samples


def build_charset(*sample_lists):
    chars = set()
    counter = Counter()

    for samples in sample_lists:
        for sample in samples:
            text = sample["label"]

            for ch in text:
                chars.add(ch)
                counter[ch] += 1

    chars = sorted(list(chars))

    char2idx = {"<blank>": 0}
    idx2char = {"0": "<blank>"}

    for idx, ch in enumerate(chars, start=1):
        char2idx[ch] = idx
        idx2char[str(idx)] = ch

    return char2idx, idx2char, counter


def compute_label_stats(samples):
    lengths = [len(sample["label"]) for sample in samples]

    if not lengths:
        return {
            "num_samples": 0,
            "min_label_length": 0,
            "max_label_length": 0,
            "avg_label_length": 0,
        }

    return {
        "num_samples": len(samples),
        "min_label_length": min(lengths),
        "max_label_length": max(lengths),
        "avg_label_length": sum(lengths) / len(lengths),
    }


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 80)
    print("PREPARE DATA - STEP 5")
    print("=" * 80)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy train dir: {TRAIN_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy test dir: {TEST_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Train dir:", TRAIN_DIR)
    print("Test dir:", TEST_DIR)
    print("Output dir:", OUTPUT_DIR)

    print("\nBuilding train raw samples...")
    train_samples_raw, train_scan_stats = build_samples_from_split(TRAIN_DIR, "train")

    print("\nBuilding test samples...")
    test_samples, test_scan_stats = build_samples_from_split(TEST_DIR, "test")

    print("\nSplitting train/val...")
    train_samples, val_samples = split_train_val(
        train_samples_raw,
        train_ratio=TRAIN_RATIO,
        seed=SEED
    )

    print("Train samples:", len(train_samples))
    print("Val samples:", len(val_samples))
    print("Test samples:", len(test_samples))

    print("\nBuilding charset...")
    # Charset nên build từ train + val + test để tránh test có ký tự lạ không decode được.
    # Khi báo cáo, bạn có thể nói charset được xây từ toàn bộ corpus label.
    char2idx, idx2char, char_counter = build_charset(
        train_samples,
        val_samples,
        test_samples
    )

    print("Charset size including <blank>:", len(char2idx))
    print("First 100 chars:")
    print(list(char2idx.keys())[:100])

    top_chars = char_counter.most_common(20)

    data_stats = {
        "train_scan": train_scan_stats,
        "test_scan": test_scan_stats,
        "train": compute_label_stats(train_samples),
        "val": compute_label_stats(val_samples),
        "test": compute_label_stats(test_samples),
        "charset_size": len(char2idx),
        "top_20_chars": top_chars,
        "train_ratio": TRAIN_RATIO,
        "seed": SEED,
    }

    print("\nData stats:")
    print(json.dumps(data_stats, ensure_ascii=False, indent=2))

    print("\nSaving outputs...")
    save_json(train_samples_raw, OUTPUT_DIR / "train_samples_raw.json")
    save_json(train_samples, OUTPUT_DIR / "train_samples.json")
    save_json(val_samples, OUTPUT_DIR / "val_samples.json")
    save_json(test_samples, OUTPUT_DIR / "test_samples.json")
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