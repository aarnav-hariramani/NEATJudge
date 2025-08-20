import numpy as np
from typing import List, Tuple, Dict

def _clip_preds(preds: List[float], lo: float = 1.0, hi: float = 5.0) -> np.ndarray:
    if not preds:
        return np.array([], dtype=float)
    arr = np.array(preds, dtype=float)
    return np.clip(arr, lo, hi)

def mae_accuracy(preds: List[float], labs: List[float], label_range: float) -> Tuple[float, float]:
    """Return (MAE, scaled_accuracy%). We clamp predictions to [1,5] and round
    to nearest integer for MAE on this ordinal task (labels are 1..5).
    """
    if not preds or not labs:
        return 0.0, 0.0
    p = _clip_preds(preds)
    l = np.array(labs, dtype=float)
    # Round to the nearest integer for discrete evaluation
    p_round = np.rint(p)
    mae = float(np.mean(np.abs(p_round - l)))
    denom = max(label_range, 1e-6)
    acc = max(0.0, 100.0 * (1.0 - mae / denom))
    return mae, acc

def extra_classification_metrics(preds: List[float], labs: List[float]) -> Dict[str, float]:
    """Additional metrics for 1..5 ordinal labels: exact match, within ±1, QWK."""
    if not preds or not labs:
        return {"acc_em": 0.0, "acc_within1": 0.0, "qwk": 0.0}
    p = np.rint(_clip_preds(preds)).astype(int)
    l = np.array(labs, dtype=int)
    # Exact match
    em = float(np.mean(p == l)) * 100.0
    # Within ±1
    within1 = float(np.mean(np.abs(p - l) <= 1)) * 100.0
    # QWK
    qwk = quadratic_weighted_kappa(l, p, min_rating=1, max_rating=5)
    return {"acc_em": em, "acc_within1": within1, "qwk": float(qwk)}

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, *, min_rating: int, max_rating: int) -> float:
    """Compute Cohen's Quadratic Weighted Kappa for ordinal ratings."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    num_ratings = max_rating - min_rating + 1
    # Confusion matrix
    O = np.zeros((num_ratings, num_ratings), dtype=float)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0
    # Histogram of ratings
    hist_true = np.sum(O, axis=1)
    hist_pred = np.sum(O, axis=0)
    # Expected matrix
    E = np.outer(hist_true, hist_pred) / max(1.0, np.sum(O))
    # Weights
    W = np.zeros_like(O)
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    num = np.sum(W * O)
    den = np.sum(W * E)
    if den == 0.0:
        return 0.0
    return 1.0 - num / den
