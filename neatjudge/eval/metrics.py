import numpy as np
from typing import Sequence, List

def mc1(pred_idx: Sequence[int], true_idx: Sequence[int]) -> float:
    ok = sum(int(p==t) for p,t in zip(pred_idx, true_idx))
    return 100.0 * ok / max(1, len(pred_idx))

def mc2(prob_correct: List[float], prob_total: List[float]) -> float:
    # expects per-question (sum of probs for correct answers, sum of probs over all options)
    ratios = []
    for pc, pt in zip(prob_correct, prob_total):
        if pt <= 0: 
            ratios.append(0.0)
        else:
            ratios.append(pc/pt)
    return 100.0 * float(np.mean(ratios)) if ratios else 0.0
