import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

history_path = "outputs/logs/history.json"

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

df = pd.DataFrame(history)
df.tail()

fig_dir = Path("visualization/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("visualization/figures/loss_curve.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["val_cer"], marker="o", label="Validation CER")
plt.plot(df["epoch"], df["val_wer"], marker="o", label="Validation WER")
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.title("Validation CER and WER")
plt.legend()
plt.grid(True)
plt.savefig("visualization/figures/cer_wer_curve.png", dpi=200, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["val_exact_match"], marker="o", label="Validation Exact Match")
plt.xlabel("Epoch")
plt.ylabel("Exact Match")
plt.title("Validation Exact Match")
plt.legend()
plt.grid(True)
plt.savefig("visualization/figures/exact_match_curve.png", dpi=200, bbox_inches="tight")
plt.show()