import argparse
import json
from pathlib import Path

import torch

from scripts.predict_line import predict_line
from scripts.predict_line_autocrop import predict_line_autocrop


DEFAULT_IMAGES = [
    "src/data/Real_line/1.jpg",
    "src/data/Real_line/2.jpg",
    "src/data/Real_line/3.jpg",
    "src/data/Real_line/4.jpg",
]

OUTPUT_PATH = Path("outputs/real_line_mode_results.json")


def safe_predict(mode_name, func, **kwargs):
    try:
        pred = func(**kwargs)
        return {
            "mode": mode_name,
            "prediction": pred,
            "error": None,
        }
    except Exception as e:
        return {
            "mode": mode_name,
            "prediction": None,
            "error": str(e),
        }


def run_all_modes_for_image(image_path):
    image_path = str(image_path)

    results = []

    # 1. Baseline cũ: predict_line.py
    results.append(
        safe_predict(
            mode_name="baseline_predict_line",
            func=predict_line,
            image_path=image_path,
        )
    )

    # 2. Autocrop mặc định: crop + làm trắng nền
    results.append(
        safe_predict(
            mode_name="autocrop_white_pad20",
            func=predict_line_autocrop,
            image_path=image_path,
            pad=20,
            white_bg=True,
            white_thresh=210,
            debug_dir="outputs/debug_autocrop/pad20_white",
        )
    )

    # 3. Autocrop nhưng không làm trắng nền
    results.append(
        safe_predict(
            mode_name="autocrop_no_white_pad20",
            func=predict_line_autocrop,
            image_path=image_path,
            pad=20,
            white_bg=False,
            white_thresh=210,
            debug_dir="outputs/debug_autocrop/pad20_no_white",
        )
    )

    # 4. Autocrop pad lớn hơn
    results.append(
        safe_predict(
            mode_name="autocrop_white_pad30",
            func=predict_line_autocrop,
            image_path=image_path,
            pad=30,
            white_bg=True,
            white_thresh=210,
            debug_dir="outputs/debug_autocrop/pad30_white",
        )
    )

    # 5. Autocrop pad lớn, không làm trắng nền
    results.append(
        safe_predict(
            mode_name="autocrop_no_white_pad30",
            func=predict_line_autocrop,
            image_path=image_path,
            pad=30,
            white_bg=False,
            white_thresh=210,
            debug_dir="outputs/debug_autocrop/pad30_no_white",
        )
    )

    # 6. Nền trắng nhẹ hơn: chỉ đẩy pixel rất sáng thành trắng
    results.append(
        safe_predict(
            mode_name="autocrop_white_pad20_thresh230",
            func=predict_line_autocrop,
            image_path=image_path,
            pad=20,
            white_bg=True,
            white_thresh=230,
            debug_dir="outputs/debug_autocrop/pad20_white230",
        )
    )

    # 7. Nền trắng mạnh hơn
    results.append(
        safe_predict(
            mode_name="autocrop_white_pad20_thresh190",
            func=predict_line_autocrop,
            image_path=image_path,
            pad=20,
            white_bg=True,
            white_thresh=190,
            debug_dir="outputs/debug_autocrop/pad20_white190",
        )
    )

    return results


def print_results_for_image(image_path, results):
    print("\n" + "=" * 100)
    print(f"IMAGE: {image_path}")
    print("=" * 100)

    for item in results:
        print("-" * 100)
        print("MODE:", item["mode"])

        if item["error"] is not None:
            print("ERROR:", item["error"])
        else:
            print("PRED:", item["prediction"])


def main():
    parser = argparse.ArgumentParser(
        description="Test multiple OCR preprocessing modes on real line images."
    )

    parser.add_argument(
        "--images",
        nargs="*",
        default=DEFAULT_IMAGES,
        help="Danh sách ảnh cần test. Nếu bỏ trống sẽ test 1.jpg đến 4.jpg."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help="File JSON lưu kết quả."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("TEST AUTOCROP MODES")
    print("=" * 100)
    print("Using device:", device)
    print("Images:")
    for img in args.images:
        print("-", img)

    all_results = []

    for image_path in args.images:
        image_path_obj = Path(image_path)

        if not image_path_obj.exists():
            result = {
                "image_path": image_path,
                "error": f"Image not found: {image_path}",
                "results": [],
            }
            all_results.append(result)
            print("\nMissing image:", image_path)
            continue

        results = run_all_modes_for_image(image_path)

        print_results_for_image(image_path, results)

        all_results.append(
            {
                "image_path": image_path,
                "error": None,
                "results": results,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved results to:", output_path)
    print("Debug images saved under: outputs/debug_autocrop")


if __name__ == "__main__":
    main()