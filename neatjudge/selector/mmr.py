
import numpy as np

def mmr_select(cand_emb: np.ndarray, q_emb: np.ndarray, k: int, lam: float = 0.3) -> list[int]:
    N = cand_emb.shape[0]
    if N == 0 or k <= 0:
        return []
    sims = cand_emb @ q_emb
    selected = []
    candidates = list(range(N))
    while candidates and len(selected) < k:
        best_i = None
        best_score = -1e9
        for i in candidates:
            if not selected:
                div = 0.0
            else:
                div = float(np.max(cand_emb[i] @ cand_emb[selected].T))
            score = lam * float(sims[i]) - (1 - lam) * div
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected
