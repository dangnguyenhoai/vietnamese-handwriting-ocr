import torch


def ocr_collate_fn(batch):
    """
    Collate function cho OCR line dataset.

    Nhiệm vụ:
    - Pad ảnh trong batch về cùng width
    - Ghép labels thành một tensor 1D cho CTC Loss
    - Lưu label_lengths
    - Lưu texts và image_paths để debug/evaluate
    """

    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    label_lengths = [item["label_length"] for item in batch]
    texts = [item["text"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    batch_size = len(images)

    channels = images[0].shape[0]
    height = images[0].shape[1]
    widths = [img.shape[2] for img in images]
    max_width = max(widths)

    # Tạo tensor batch toàn màu trắng sau normalize.
    # Ảnh nền thường trắng, pixel trắng sau normalize gần 1.
    padded_images = torch.ones(
        batch_size,
        channels,
        height,
        max_width,
        dtype=torch.float32
    )

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img

    # CTC Loss cần target labels dạng concat 1D:
    # label 1 + label 2 + label 3 + ...
    labels_concat = torch.cat(labels, dim=0)

    label_lengths = torch.stack(label_lengths)

    image_widths = torch.tensor(widths, dtype=torch.long)

    return {
        "images": padded_images,
        "labels": labels_concat,
        "label_lengths": label_lengths,
        "image_widths": image_widths,
        "texts": texts,
        "image_paths": image_paths,
    }