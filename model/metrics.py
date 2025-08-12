
import numpy as np
from typing import List, Tuple
def mae_accuracy(preds: List[float], labs: List[float], label_range: float) -> Tuple[float, float]:
    if not preds or not labs:
        return 0.0, 0.0
    arr_p = np.array(preds, dtype=float)
    arr_l = np.array(labs, dtype=float)
    mae = float(np.mean(np.abs(arr_p - arr_l)))
    denom = max(label_range, 1e-6)
    acc = max(0.0, 100.0 * (1.0 - mae / denom))
    return mae, acc
