from torch.utils.data import DataLoader

from src.data.dataset import OCRLineDataset
from src.data.collate import ocr_collate_fn


def main():
    dataset = OCRLineDataset(
        samples_path="outputs/ocr_outputs/train_samples.json",
        char2idx_path="outputs/ocr_outputs/char2idx.json",
        image_height=64
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=ocr_collate_fn
    )

    batch = next(iter(loader))

    print("=" * 80)
    print("TEST DATALOADER")
    print("=" * 80)

    print("Images shape:", batch["images"].shape)
    print("Labels concat shape:", batch["labels"].shape)
    print("Label lengths:", batch["label_lengths"])
    print("Image widths:", batch["image_widths"])

    print("\nTexts:")
    for text in batch["texts"]:
        print("-", text)

    print("\nImage paths:")
    for path in batch["image_paths"]:
        print("-", path)


if __name__ == "__main__":
    main()