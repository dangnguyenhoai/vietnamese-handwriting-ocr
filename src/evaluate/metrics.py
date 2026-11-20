def levenshtein_distance(a, b):
    """
    Tính edit distance giữa 2 sequence.

    a, b có thể là:
    - string: tính theo ký tự
    - list words: tính theo từ
    """

    n = len(a)
    m = len(b)

    # dp[i][j] = khoảng cách edit giữa a[:i] và b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete
                dp[i][j - 1] + 1,       # insert
                dp[i - 1][j - 1] + cost # substitute
            )

    return dp[n][m]


def cer(pred, target):
    """
    Character Error Rate.

    CER = edit_distance(pred_chars, target_chars) / len(target_chars)
    """
    pred = str(pred)
    target = str(target)

    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    distance = levenshtein_distance(pred, target)
    return distance / len(target)


def wer(pred, target):
    """
    Word Error Rate.

    WER = edit_distance(pred_words, target_words) / len(target_words)
    """
    pred_words = str(pred).split()
    target_words = str(target).split()

    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    distance = levenshtein_distance(pred_words, target_words)
    return distance / len(target_words)


def exact_match(pred, target):
    """
    Dòng dự đoán có khớp hoàn toàn với ground truth không.
    """
    return 1.0 if str(pred) == str(target) else 0.0


def compute_batch_metrics(predictions, targets):
    """
    Tính CER/WER/Exact Match trung bình cho một batch.
    """
    assert len(predictions) == len(targets)

    cer_scores = []
    wer_scores = []
    exact_scores = []

    for pred, target in zip(predictions, targets):
        cer_scores.append(cer(pred, target))
        wer_scores.append(wer(pred, target))
        exact_scores.append(exact_match(pred, target))

    return {
        "cer": sum(cer_scores) / len(cer_scores),
        "wer": sum(wer_scores) / len(wer_scores),
        "exact_match": sum(exact_scores) / len(exact_scores),
    }