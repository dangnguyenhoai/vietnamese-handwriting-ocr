from src.data.dataset import OCRLineDataset



def main():
    dataset = OCRLineDataset(
        samples_path="outputs/ocr_outputs/train_samples.json",
        char2idx_path="outputs/ocr_outputs/char2idx.json",
        image_height=64
    )

    print("=" * 80)
    print("TEST DATASET")
    print("=" * 80)

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Image shape:", sample["image"].shape)
    print("Label shape:", sample["label"].shape)
    print("Label length:", sample["label_length"])
    print("Text:", sample["text"])
    print("Image path:", sample["image_path"])
    print("First 30 label ids:", sample["label"][:30])

    print("\nMin pixel:", sample["image"].min().item())
    print("Max pixel:", sample["image"].max().item())


if __name__ == "__main__":
    main()