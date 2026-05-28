from pathlib import Path
import json
import unicodedata
from PIL import Image


DATA_ROOT = Path("src/data/UIT_HWDB_line")
TRAIN_DIR = DATA_ROOT / "train_data"

IMG_HEIGHT = 64
CNN_WIDTH_DOWNSAMPLE = 4


def normalize_text(text):
    text = str(text).strip()
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def compute_resized_width(width, height, target_height=IMG_HEIGHT):
    scale = target_height / float(height)
    return max(1, int(round(width * scale)))


def count_adjacent_repeats(text):
    return sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])


def required_ctc_timesteps(text):
    return len(text) + count_adjacent_repeats(text)


def main():
    bad_samples = []

    folders = sorted(
        [p for p in TRAIN_DIR.iterdir() if p.is_dir()],
        key=lambda x: int(x.name) if x.name.isdigit() else x.name,
    )

    for folder in folders:
        label_path = folder / "label.json"
        if not label_path.exists():
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        for image_name, label in labels.items():
            image_path = folder / image_name
            if not image_path.exists():
                continue

            label = normalize_text(label)

            with Image.open(image_path) as img:
                width, height = img.size

            resized_width = compute_resized_width(width, height)
            input_timesteps = resized_width // CNN_WIDTH_DOWNSAMPLE
            ctc_required = required_ctc_timesteps(label)

            if ctc_required > input_timesteps:
                bad_samples.append({
                    "image_path": str(image_path),
                    "source_folder": folder.name,
                    "image_name": image_name,
                    "label": label,
                    "label_length": len(label),
                    "adjacent_repeats": count_adjacent_repeats(label),
                    "ctc_required_timesteps": ctc_required,
                    "orig_width": width,
                    "orig_height": height,
                    "resized_width": resized_width,
                    "input_timesteps": input_timesteps,
                })

    out_path = Path("outputs/ocr_outputs/ctc_invalid_samples.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bad_samples, f, ensure_ascii=False, indent=2)

    print("CTC invalid samples:", len(bad_samples))
    print("Saved to:", out_path)

    for i, sample in enumerate(bad_samples, start=1):
        print("=" * 80)
        print("No:", i)
        print("image_path:", sample["image_path"])
        print("label_length:", sample["label_length"])
        print("adjacent_repeats:", sample["adjacent_repeats"])
        print("ctc_required_timesteps:", sample["ctc_required_timesteps"])
        print("input_timesteps:", sample["input_timesteps"])
        print("resized_width:", sample["resized_width"])
        print("label:", sample["label"])


if __name__ == "__main__":
    main()