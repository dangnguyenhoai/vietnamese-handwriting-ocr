import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import OCRLineDataset
from src.data.collate import ocr_collate_fn
from src.models.crnn_ctc import CRNNCTC
from src.models.decoder import CTCGreedyDecoder
from src.evaluate.metrics import cer, wer, exact_match, compute_batch_metrics


OUTPUT_DIR = Path("outputs/ocr_outputs")
CHECKPOINT_DIR = Path("outputs/checkpoints")
EVAL_DIR = Path("outputs/eval")

TEST_SAMPLES = OUTPUT_DIR / "test_samples.json"
CHAR2IDX = OUTPUT_DIR / "char2idx.json"
IDX2CHAR = OUTPUT_DIR / "idx2char.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_crnn_ctc.pth"

IMAGE_HEIGHT = 64
BATCH_SIZE = 16
NUM_WORKERS = 2


def load_model(checkpoint_path, num_classes, device):
    model = CRNNCTC(
        num_classes=num_classes,
        input_channels=1,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.3
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint val_loss:", checkpoint.get("val_loss"))
    print("Checkpoint val_metrics:", checkpoint.get("val_metrics"))

    return model


def evaluate_test(model, loader, criterion, decoder, device):
    total_loss = 0.0
    total_batches = 0

    all_predictions = []
    all_targets = []
    prediction_records = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating test set"):
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            image_widths = batch["image_widths"].to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2)

            input_lengths = model.get_output_lengths(image_widths)

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            preds = decoder.decode_logits(logits)
            targets = batch["texts"]
            image_paths = batch["image_paths"]

            total_loss += loss.item()
            total_batches += 1

            all_predictions.extend(preds)
            all_targets.extend(targets)

            for pred, target, image_path in zip(preds, targets, image_paths):
                record = {
                    "image_path": image_path,
                    "ground_truth": target,
                    "prediction": pred,
                    "cer": cer(pred, target),
                    "wer": wer(pred, target),
                    "exact_match": exact_match(pred, target),
                }
                prediction_records.append(record)

    avg_loss = total_loss / max(total_batches, 1)
    metrics = compute_batch_metrics(all_predictions, all_targets)

    results = {
        "test_loss": avg_loss,
        "test_cer": metrics["cer"],
        "test_wer": metrics["wer"],
        "test_exact_match": metrics["exact_match"],
        "num_samples": len(all_targets),
    }

    return results, prediction_records


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_examples(prediction_records, n=10):
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    # In vài dòng đầu
    for i, record in enumerate(prediction_records[:n]):
        print("-" * 80)
        print(f"Sample {i + 1}")
        print("Image:", record["image_path"])
        print("GT   :", record["ground_truth"])
        print("PRED :", record["prediction"])
        print(f"CER  : {record['cer']:.4f}")
        print(f"WER  : {record['wer']:.4f}")
        print(f"EM   : {record['exact_match']:.0f}")


def print_best_worst_examples(prediction_records, n=5):
    sorted_by_cer = sorted(prediction_records, key=lambda x: x["cer"])

    best = sorted_by_cer[:n]
    worst = sorted_by_cer[-n:]

    print("\n" + "=" * 80)
    print("BEST EXAMPLES")
    print("=" * 80)

    for i, record in enumerate(best):
        print("-" * 80)
        print(f"Best {i + 1}")
        print("GT   :", record["ground_truth"])
        print("PRED :", record["prediction"])
        print(f"CER  : {record['cer']:.4f}")
        print(f"WER  : {record['wer']:.4f}")

    print("\n" + "=" * 80)
    print("WORST EXAMPLES")
    print("=" * 80)

    for i, record in enumerate(worst):
        print("-" * 80)
        print(f"Worst {i + 1}")
        print("GT   :", record["ground_truth"])
        print("PRED :", record["prediction"])
        print(f"CER  : {record['cer']:.4f}")
        print(f"WER  : {record['wer']:.4f}")


def main():
    print("=" * 80)
    print("EVALUATE CRNN-CTC ON TEST SET")
    print("=" * 80)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if not TEST_SAMPLES.exists():
        raise FileNotFoundError(f"Test samples not found: {TEST_SAMPLES}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open(CHAR2IDX, "r", encoding="utf-8") as f:
        char2idx = json.load(f)

    with open(IDX2CHAR, "r", encoding="utf-8") as f:
        idx2char = json.load(f)

    num_classes = len(char2idx)
    print("Num classes:", num_classes)

    test_dataset = OCRLineDataset(
        samples_path=TEST_SAMPLES,
        char2idx_path=CHAR2IDX,
        image_height=IMAGE_HEIGHT
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ocr_collate_fn
    )

    print("Test samples:", len(test_dataset))
    print("Test batches:", len(test_loader))

    model = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=num_classes,
        device=device
    )

    criterion = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True
    )

    decoder = CTCGreedyDecoder(
        idx2char=idx2char,
        blank_idx=0
    )

    results, prediction_records = evaluate_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        decoder=decoder,
        device=device
    )

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Test Loss        : {results['test_loss']:.4f}")
    print(f"Test CER         : {results['test_cer']:.4f}")
    print(f"Test WER         : {results['test_wer']:.4f}")
    print(f"Test Exact Match : {results['test_exact_match']:.4f}")
    print(f"Num samples      : {results['num_samples']}")

    save_json(results, EVAL_DIR / "test_results.json")
    save_json(prediction_records, EVAL_DIR / "test_predictions.json")

    print("\nSaved:")
    print("-", EVAL_DIR / "test_results.json")
    print("-", EVAL_DIR / "test_predictions.json")

    print_examples(prediction_records, n=10)
    print_best_worst_examples(prediction_records, n=5)


if __name__ == "__main__":
    main()