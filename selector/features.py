
import numpy as np

def build_features(q_emb: np.ndarray, cand_emb: np.ndarray) -> np.ndarray:
    """Feature map: [cand, q, |cand-q|, cand*q] => 4D per candidate."""
    q = q_emb.reshape(1, -1)
    N, D = cand_emb.shape
    diff = np.abs(cand_emb - q)
    prod = cand_emb * q
    feats = np.concatenate([cand_emb, np.repeat(q, N, axis=0), diff, prod], axis=1)
    return feats  # (N, 4D)
