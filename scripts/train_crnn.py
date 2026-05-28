import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import OCRLineDataset
from src.data.collate import ocr_collate_fn
from src.data.transforms import build_train_transform, build_val_transform
from src.models.crnn_ctc import CRNNCTC
from src.models.decoder import CTCGreedyDecoder
from src.evaluate.metrics import compute_batch_metrics


OUTPUT_DIR = Path("outputs/ocr_outputs")
CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_DIR = Path("outputs/logs")

TRAIN_SAMPLES = OUTPUT_DIR / "train_samples.json"
VAL_SAMPLES = OUTPUT_DIR / "val_samples.json"
CHAR2IDX = OUTPUT_DIR / "char2idx.json"
IDX2CHAR = OUTPUT_DIR / "idx2char.json"
RESUME = False
RESUME_PATH = CHECKPOINT_DIR / "latest_crnn_ctc.pth"

IMAGE_HEIGHT = 64
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 5e-4
NUM_WORKERS = 2
GRAD_CLIP = 5.0
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 0.0

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
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

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"Batch {batch_idx + 1}/{len(loader)} - loss: {loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def evaluate(model, loader, criterion, decoder, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
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

            all_preds.extend(preds)
            all_targets.extend(batch["texts"])

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    metrics = compute_batch_metrics(all_preds, all_targets)

    return avg_loss, metrics


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_metrics,
    best_val_cer,
    history,
    path,
    no_improve_epochs=0,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "best_val_cer": best_val_cer,
        "history": history,
        "no_improve_epochs": no_improve_epochs,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
    }

    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_cer = checkpoint.get("best_val_cer", float("inf"))
    history = checkpoint.get("history", [])
    no_improve_epochs = checkpoint.get("no_improve_epochs", 0)

    print(f"Loaded checkpoint: {path}")
    print(f"Resume from epoch: {start_epoch}")
    print(f"Best Val CER so far: {best_val_cer}")
    print(f"No-improve epochs so far: {no_improve_epochs}")

    return start_epoch, best_val_cer, history, no_improve_epochs

def main():
    print("=" * 80)
    print("TRAIN CRNN-CTC")
    print("=" * 80)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with open(CHAR2IDX, "r", encoding="utf-8") as f:
        char2idx = json.load(f)

    with open(IDX2CHAR, "r", encoding="utf-8") as f:
        idx2char = json.load(f)

    num_classes = len(char2idx)
    print("Num classes:", num_classes)

    train_dataset = OCRLineDataset(
        samples_path=TRAIN_SAMPLES,
        char2idx_path=CHAR2IDX,
        image_height=IMAGE_HEIGHT,
        transform=build_train_transform(),
    )

    val_dataset = OCRLineDataset(
        samples_path=VAL_SAMPLES,
        char2idx_path=CHAR2IDX,
        image_height=IMAGE_HEIGHT,
        transform=build_val_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=ocr_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ocr_collate_fn
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    model = CRNNCTC(
        num_classes=num_classes,
        input_channels=1,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.3
    ).to(device)

    criterion = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    decoder = CTCGreedyDecoder(
        idx2char=idx2char,
        blank_idx=0
    )

    history = []
    best_val_cer = float("inf")
    no_improve_epochs = 0
    start_epoch = 1

    if RESUME and RESUME_PATH.exists():
        start_epoch, best_val_cer, history, no_improve_epochs = load_checkpoint(
            path=RESUME_PATH,
            model=model,
            optimizer=optimizer,
            device=device
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        print("\n" + "=" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 80)

        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            decoder=decoder,
            device=device
        )

        elapsed = time.time() - start_time

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_cer": val_metrics["cer"],
            "val_wer": val_metrics["wer"],
            "val_exact_match": val_metrics["exact_match"],
            "time_sec": elapsed,
        }

        history.append(epoch_log)

        print(f"Train loss      : {train_loss:.4f}")
        print(f"Val loss        : {val_loss:.4f}")
        print(f"Val CER         : {val_metrics['cer']:.4f}")
        print(f"Val WER         : {val_metrics['wer']:.4f}")
        print(f"Val Exact Match : {val_metrics['exact_match']:.4f}")
        print(f"Time            : {elapsed:.2f}s")

        with open(LOG_DIR / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        is_best = (best_val_cer - val_metrics["cer"]) > EARLY_STOP_MIN_DELTA

        if is_best:
            best_val_cer = val_metrics["cer"]
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Luôn lưu latest sau khi đã cập nhật best_val_cer
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            val_metrics=val_metrics,
            best_val_cer=best_val_cer,
            history=history,
            path=CHECKPOINT_DIR / "latest_crnn_ctc.pth",
            no_improve_epochs=no_improve_epochs,
        )

        print("Saved latest checkpoint.")
        print(f"Early stop counter: {no_improve_epochs}/{EARLY_STOP_PATIENCE}")

        # Nếu epoch này tốt nhất thì lưu thêm best
        if is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_metrics=val_metrics,
                best_val_cer=best_val_cer,
                history=history,
                path=CHECKPOINT_DIR / "best_crnn_ctc.pth",
                no_improve_epochs=no_improve_epochs,
            )

            print("Saved best checkpoint.")

        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(
                "Early stopping triggered: "
                f"Val CER did not improve for {EARLY_STOP_PATIENCE} epochs."
            )
            break

    print("\nTraining done.")
    print("Best Val CER:", best_val_cer)
    print("Checkpoint:", CHECKPOINT_DIR / "best_crnn_ctc.pth")
    print("History:", LOG_DIR / "history.json")


if __name__ == "__main__":
    main()
