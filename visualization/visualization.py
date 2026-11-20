import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HISTORY_PATH = Path("outputs/logs/history.json")
TEST_RESULTS_PATH = Path("outputs/eval/test_results.json")
FIG_DIR = Path("visualization/figures")


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise ValueError(f"Empty history file: {path}")

    df = pd.DataFrame(history).sort_values("epoch").reset_index(drop=True)
    required_cols = ["epoch", "train_loss", "val_loss"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required history columns: {missing_cols}")

    return df


def load_test_results(path: Path):
    if not path.exists():
        print(f"Skip test comparison, file not found: {path}")
        return None

    with path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    required_keys = ["test_cer", "test_wer", "test_exact_match", "num_samples"]
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        print(f"Skip test comparison, missing keys: {missing_keys}")
        return None

    return results


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.figsize": (10, 5),
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
        }
    )


def finish_chart(fig, ax, output_path: Path) -> None:
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_loss_curve(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["epoch"], df["train_loss"], color="#1f77b4", label="Train loss")
    ax.plot(df["epoch"], df["val_loss"], color="#ff7f0e", label="Validation loss")
    ax.set_title("Diễn biến loss trong quá trình huấn luyện", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=True)
    finish_chart(fig, ax, FIG_DIR / "loss_curve.png")


def plot_error_metrics(df: pd.DataFrame) -> None:
    metric_cols = [
        ("val_cer", "Validation CER", "#d62728"),
        ("val_wer", "Validation WER", "#9467bd"),
    ]
    available_metrics = [(col, label, color) for col, label, color in metric_cols if col in df.columns]
    if not available_metrics:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label, color in available_metrics:
        ax.plot(df["epoch"], df[col], color=color, label=label)

    ax.set_title("Diễn biến CER/WER trên tập validation", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error rate")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True)
    finish_chart(fig, ax, FIG_DIR / "cer_wer_curve.png")


def plot_exact_match(df: pd.DataFrame) -> None:
    if "val_exact_match" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["epoch"],
        df["val_exact_match"],
        color="#2ca02c",
        label="Validation exact match",
    )
    ax.set_title("Diễn biến Exact Match trên tập validation", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Exact match")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=True)
    finish_chart(fig, ax, FIG_DIR / "exact_match_curve.png")


def plot_validation_metrics_percent(df: pd.DataFrame) -> None:
    metric_cols = [
        ("val_cer", "CER validation (%)", "#1f77b4"),
        ("val_wer", "WER validation (%)", "#ff7f0e"),
        ("val_exact_match", "Exact Match validation (%)", "#2ca02c"),
    ]
    missing_cols = [col for col, _, _ in metric_cols if col not in df.columns]
    if missing_cols:
        return

    best_cer_row = df.loc[df["val_cer"].idxmin()]
    best_cer_epoch = int(best_cer_row["epoch"])

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label, color in metric_cols:
        ax.plot(df["epoch"], df[col] * 100, color=color, label=label)

    ax.axvline(
        best_cer_epoch,
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label=f"Best CER: epoch {best_cer_epoch}",
    )
    ax.set_title("Diễn biến metrics validation theo epoch", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)
    finish_chart(fig, ax, FIG_DIR / "validation_metrics_percent.png")


def plot_validation_test_comparison(df: pd.DataFrame, test_results) -> None:
    history_cols = ["val_cer", "val_wer", "val_exact_match"]
    if test_results is None or any(col not in df.columns for col in history_cols):
        return

    best_cer_row = df.loc[df["val_cer"].idxmin()]
    best_cer_epoch = int(best_cer_row["epoch"])
    num_samples = int(test_results["num_samples"])

    group_labels = [
        f"Val best\n(epoch {best_cer_epoch})",
        f"Test set\n({num_samples} mẫu)",
    ]
    metrics = [
        ("CER", "val_cer", "test_cer", "#1f77b4"),
        ("WER", "val_wer", "test_wer", "#ff7f0e"),
        ("Exact Match", "val_exact_match", "test_exact_match", "#2ca02c"),
    ]

    x = np.arange(len(group_labels))
    width = 0.25
    max_value = 0.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for index, (label, val_col, test_key, color) in enumerate(metrics):
        values = [
            float(best_cer_row[val_col]) * 100,
            float(test_results[test_key]) * 100,
        ]
        max_value = max(max_value, *values)
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}%" for value in values],
            padding=3,
            fontsize=9,
        )

    ax.set_title("Validation tốt hơn test: cần đánh giá trung thực", pad=10, weight="bold")
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, max(30, max_value * 1.2))
    ax.legend(loc="upper right", frameon=True)
    finish_chart(fig, ax, FIG_DIR / "validation_test_comparison.png")


def plot_epoch_time(df: pd.DataFrame) -> None:
    if "time_sec" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["epoch"],
        df["time_sec"] / 60,
        color="#17becf",
        label="Time per epoch",
    )
    ax.set_title("Thời gian huấn luyện theo epoch", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Minutes")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True)
    finish_chart(fig, ax, FIG_DIR / "epoch_time_curve.png")


def plot_metrics_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    axes[0].plot(df["epoch"], df["train_loss"], color="#1f77b4", label="Train loss")
    axes[0].plot(df["epoch"], df["val_loss"], color="#ff7f0e", label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=True)

    if "val_cer" in df.columns:
        axes[1].plot(df["epoch"], df["val_cer"], color="#d62728", label="Validation CER")
    if "val_wer" in df.columns:
        axes[1].plot(df["epoch"], df["val_wer"], color="#9467bd", label="Validation WER")
    axes[1].set_title("CER / WER")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error rate")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=True)

    if "val_exact_match" in df.columns:
        axes[2].plot(
            df["epoch"],
            df["val_exact_match"],
            color="#2ca02c",
            label="Validation exact match",
        )
    axes[2].set_title("Exact Match")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(frameon=True)

    if "time_sec" in df.columns:
        axes[3].plot(df["epoch"], df["time_sec"] / 60, color="#17becf", label="Minutes")
    axes[3].set_title("Time per Epoch")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Minutes")
    axes[3].set_ylim(bottom=0)
    axes[3].legend(frameon=True)

    for ax in axes:
        ax.grid(True, alpha=0.35)
        ax.margins(x=0.02)

    fig.suptitle("Tổng quan metrics trong quá trình huấn luyện", fontsize=17, y=1.02)
    fig.tight_layout()
    output_path = FIG_DIR / "metrics_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def format_float(value: float) -> str:
    return f"{value:.4f}"


def build_summary(df: pd.DataFrame) -> dict:
    last_row = df.iloc[-1]
    summary = {
        "so_epoch": int(len(df)),
        "epoch_cuoi": int(last_row["epoch"]),
        "train_loss_cuoi": format_float(last_row["train_loss"]),
        "val_loss_cuoi": format_float(last_row["val_loss"]),
    }

    best_val_loss = df.loc[df["val_loss"].idxmin()]
    summary["best_val_loss"] = format_float(best_val_loss["val_loss"])
    summary["best_val_loss_epoch"] = int(best_val_loss["epoch"])

    optional_metrics = [
        ("val_cer", "best_val_cer", "min"),
        ("val_wer", "best_val_wer", "min"),
        ("val_exact_match", "best_val_exact_match", "max"),
    ]
    for source_col, target_name, mode in optional_metrics:
        if source_col not in df.columns:
            continue

        row = df.loc[df[source_col].idxmin()] if mode == "min" else df.loc[df[source_col].idxmax()]
        summary[target_name] = format_float(row[source_col])
        summary[f"{target_name}_epoch"] = int(row["epoch"])
        summary[f"{source_col}_cuoi"] = format_float(last_row[source_col])

    if "time_sec" in df.columns:
        summary["tong_thoi_gian_phut"] = format_float(df["time_sec"].sum() / 60)
        summary["thoi_gian_tb_moi_epoch_giay"] = format_float(df["time_sec"].mean())

    return summary


def print_and_save_summary(summary: dict) -> None:
    print("\n===== METRICS SUMMARY =====")
    for key, value in summary.items():
        print(f"{key:<32}: {value}")

    summary_path = FIG_DIR / "metrics_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    df = load_history(HISTORY_PATH)
    test_results = load_test_results(TEST_RESULTS_PATH)
    plot_loss_curve(df)
    plot_error_metrics(df)
    plot_exact_match(df)
    plot_validation_metrics_percent(df)
    plot_validation_test_comparison(df, test_results)
    plot_epoch_time(df)
    plot_metrics_overview(df)
    print_and_save_summary(build_summary(df))


if __name__ == "__main__":
    main()
