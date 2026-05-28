import json

out = "outputs/ocr_outputs"

with open(f"{out}/train_folders.json", encoding="utf-8") as f:
    train_folders = set(json.load(f))

with open(f"{out}/val_folders.json", encoding="utf-8") as f:
    val_folders = set(json.load(f))

overlap = train_folders & val_folders
print("Train folders:", len(train_folders))
print("Val folders:", len(val_folders))
print("Overlap:", len(overlap))

if overlap:
    print("LEAKAGE:", sorted(overlap)[:20])
else:
    print("OK: không có folder leakage.")