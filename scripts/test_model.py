import json
import torch
from torch.utils.data import DataLoader

from src.data.dataset import OCRLineDataset
from src.data.collate import ocr_collate_fn
from src.models.crnn_ctc import CRNNCTC


def main():
    with open("outputs/ocr_outputs/char2idx.json", "r", encoding="utf-8") as f:
        char2idx = json.load(f)

    num_classes = len(char2idx)

    print("=" * 80)
    print("TEST CRNN-CTC MODEL")
    print("=" * 80)

    print("Num classes:", num_classes)

    dataset = OCRLineDataset(
        samples_path="outputs/ocr_outputs/train_samples.json",
        char2idx_path="outputs/ocr_outputs/char2idx.json",
        image_height=64
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=ocr_collate_fn
    )

    batch = next(iter(loader))

    images = batch["images"]
    image_widths = batch["image_widths"]
    labels = batch["labels"]
    label_lengths = batch["label_lengths"]

    model = CRNNCTC(
        num_classes=num_classes,
        input_channels=1,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.3
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)
        input_lengths = model.get_output_lengths(image_widths)

    print("Images shape:", images.shape)
    print("Image widths:", image_widths)

    print("Logits shape:", logits.shape)
    print("Input lengths:", input_lengths)

    print("Labels concat shape:", labels.shape)
    print("Label lengths:", label_lengths)

    print("\nTexts:")
    for text in batch["texts"]:
        print("-", text)

    # Kiểm tra điều kiện cơ bản của CTC:
    # input_length phải đủ dài so với label_length.
    print("\nCTC length check:")
    for i in range(len(label_lengths)):
        print(
            f"sample {i}: input_length={input_lengths[i].item()}, "
            f"label_length={label_lengths[i].item()}"
        )


if __name__ == "__main__":
    main()