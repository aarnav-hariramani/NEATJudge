
import numpy as np
def mmr_select(cands_emb, query_emb, K, lam=0.3):
    sims = (cands_emb @ query_emb)
    selected = []
    pool = list(range(len(cands_emb)))
    while pool and len(selected) < K:
        if not selected:
            i = int(np.argmax(sims[pool]))
            selected.append(pool.pop(i))
            continue
        S = np.array(selected, dtype=int)
        red = np.max((cands_emb[pool] @ cands_emb[S].T), axis=1)
        score = lam * sims[pool] - (1.0 - lam) * red
        i = int(np.argmax(score))
        selected.append(pool.pop(i))
    return selected
