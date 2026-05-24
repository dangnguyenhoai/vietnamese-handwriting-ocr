from src.evaluate.metrics import cer, wer, exact_match, compute_batch_metrics


def main():
    print("=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    pred = "cho nguoi nghèo"
    target = "cho người nghèo"

    print("Pred  :", pred)
    print("Target:", target)

    print("CER:", cer(pred, target))
    print("WER:", wer(pred, target))
    print("Exact:", exact_match(pred, target))

    predictions = [
        "cho nguoi nghèo",
        "hôm nay trời đẹp",
        "abc"
    ]

    targets = [
        "cho người nghèo",
        "hôm nay trời đẹp",
        "abcd"
    ]

    metrics = compute_batch_metrics(predictions, targets)

    print("\nBatch metrics:")
    print(metrics)


if __name__ == "__main__":
    main()