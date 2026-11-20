import torch


class CTCGreedyDecoder:
    """
    Greedy decoder cho CTC.

    Quy tắc:
    1. Lấy class có xác suất cao nhất tại mỗi timestep.
    2. Gộp các ký tự lặp liên tiếp.
    3. Bỏ <blank>.
    4. Ghép lại thành chuỗi text.
    """

    def __init__(self, idx2char, blank_idx=0):
        """
        idx2char:
            dict dạng:
            {
                "0": "<blank>",
                "1": " ",
                "2": "a",
                ...
            }

        blank_idx:
            index của CTC blank token, mặc định là 0.
        """
        self.idx2char = idx2char
        self.blank_idx = blank_idx

    def decode_indices(self, indices):
        """
        Decode một sequence index thành text.

        Ví dụ:
            [0, 5, 5, 0, 8, 8, 0, 9]
        Sau gộp lặp + bỏ blank:
            [5, 8, 9]
        """
        decoded_chars = []
        prev_idx = None

        for idx in indices:
            idx = int(idx)

            # Bỏ ký tự lặp liên tiếp
            if idx == prev_idx:
                continue

            # Bỏ blank
            if idx != self.blank_idx:
                char = self.idx2char.get(str(idx), "")
                decoded_chars.append(char)

            prev_idx = idx

        return "".join(decoded_chars)

    def decode_logits(self, logits):
        """
        Decode logits từ model.

        Input:
            logits: [T, B, C]

        Output:
            list[str] độ dài B
        """
        # Lấy class có logit lớn nhất tại mỗi timestep
        # pred_indices: [T, B]
        pred_indices = torch.argmax(logits, dim=2)

        # Chuyển sang [B, T]
        pred_indices = pred_indices.permute(1, 0)

        results = []

        for sequence in pred_indices:
            text = self.decode_indices(sequence.tolist())
            results.append(text)

        return results