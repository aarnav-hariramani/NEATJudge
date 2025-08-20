
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / n

class BankIndex:
    def __init__(self, bank_rows: List[Dict[str, Any]], embed_model: str):
        self.embedder = SentenceTransformer(embed_model)
        self.bank_rows = bank_rows
        texts = [f'{r["query"]} {r["response"]}' for r in bank_rows]
        self.bank_emb = _norm(self.embedder.encode(texts, convert_to_numpy=True))
        q_texts = [r["query"] for r in bank_rows]
        self.bank_qemb = _norm(self.embedder.encode(q_texts, convert_to_numpy=True))

    def emb_dim(self) -> int:
        return int(self.bank_emb.shape[1])

    def encode_query(self, query: str) -> np.ndarray:
        return _norm(self.embedder.encode([query], convert_to_numpy=True))[0]

    def shortlist(self, q_emb: np.ndarray, topk: int = 128) -> np.ndarray:
        sims = (self.bank_emb @ q_emb)
        idx = np.argpartition(-sims, min(topk, len(sims)-1))[:topk]
        idx = idx[np.argsort(-sims[idx])]
        return idx

    def filter_similar(self, cand_idx: np.ndarray, query_emb: np.ndarray, cap: float) -> np.ndarray:
        """Drop candidates whose BANK QUERY is too similar to current query."""
        sims = self.bank_qemb[cand_idx] @ query_emb
        keep = cand_idx[sims < cap]
        if keep.size == 0:
            order = np.argsort(sims)
            keep = cand_idx[order[: min(5, len(order))]]
        return keep

    def examples(self, idxs: List[int]) -> List[Dict[str, Any]]:
        return [self.bank_rows[i] for i in idxs]

    def diversity(self, idxs: List[int], bins: int = 5) -> float:
        """Simple label-bin coverage of selected examples, scaled to 0..100."""
        if not idxs:
            return 0.0
        labels = [self.bank_rows[i]["label"] for i in idxs]
        lo, hi = min(labels), max(labels)
        if hi == lo:
            return 0.0
        arr = np.array(labels)
        cats = np.floor((arr - lo) / (hi - lo + 1e-9) * bins).astype(int)
        cats = np.clip(cats, 0, bins-1)
        cov = len(set(cats.tolist())) / float(bins)
        return cov * 100.0
