import json
from src.models.decoder import CTCGreedyDecoder


def main():
    with open("outputs/ocr_outputs/char2idx.json", "r", encoding="utf-8") as f:
        char2idx = json.load(f)

    with open("outputs/ocr_outputs/idx2char.json", "r", encoding="utf-8") as f:
        idx2char = json.load(f)

    decoder = CTCGreedyDecoder(idx2char=idx2char, blank_idx=0)

    # Lấy id của vài ký tự chắc chắn có trong charset
    c = char2idx["c"]
    h = char2idx["h"]
    o = char2idx["o"]
    blank = 0

    fake_indices = [
        blank, blank,
        c, c, c,
        blank,
        h, h,
        blank,
        o, o, o,
        blank
    ]

    decoded = decoder.decode_indices(fake_indices)

    print("Fake indices:", fake_indices)
    print("Decoded text:", decoded)


if __name__ == "__main__":
    main()