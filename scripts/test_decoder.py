import json
import torch
from torch.utils.data import DataLoader

from src.data.dataset import OCRLineDataset
from src.data.collate import ocr_collate_fn
from src.models.crnn_ctc import CRNNCTC
from src.models.decoder import CTCGreedyDecoder


def main():
    with open("outputs/ocr_outputs/char2idx.json", "r", encoding="utf-8") as f:
        char2idx = json.load(f)

    with open("outputs/ocr_outputs/idx2char.json", "r", encoding="utf-8") as f:
        idx2char = json.load(f)

    num_classes = len(char2idx)

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

    model = CRNNCTC(
        num_classes=num_classes,
        input_channels=1,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.3
    )

    decoder = CTCGreedyDecoder(
        idx2char=idx2char,
        blank_idx=0
    )

    model.eval()

    with torch.no_grad():
        logits = model(batch["images"])
        predictions = decoder.decode_logits(logits)

    print("=" * 80)
    print("TEST CTC DECODER")
    print("=" * 80)

    print("Logits shape:", logits.shape)

    print("\nPredictions vs Ground Truth:")
    for i, (pred, gt) in enumerate(zip(predictions, batch["texts"])):
        print("-" * 80)
        print(f"Sample {i}")
        print("PRED:", repr(pred))
        print("GT  :", repr(gt))


if __name__ == "__main__":
    main()