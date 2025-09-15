import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from .loaders import QAExample

def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / n

class BankIndex:
    def __init__(self, bank_rows: List[QAExample], embed_model: str):
        self.embedder = SentenceTransformer(embed_model)
        self.bank_rows = bank_rows
        texts = [ex.question for ex in bank_rows]
        self.bank_emb = _norm(self.embedder.encode(texts, convert_to_numpy=True))
    def encode_query(self, q: str) -> np.ndarray:
        return _norm(self.embedder.encode([q], convert_to_numpy=True))[0]
    def shortlist(self, q_emb: np.ndarray, k: int) -> np.ndarray:
        sims = (self.bank_emb @ q_emb)
        idx = np.argsort(-sims)[:k]
        return idx
    def filter_similar(self, idxs: np.ndarray, q_emb: np.ndarray, cap: int) -> List[int]:
        if len(idxs) <= cap:
            return idxs.tolist()
        chosen = []
        for i in idxs:
            if not chosen:
                chosen.append(int(i)); 
                if len(chosen)>=cap: break
                continue
            # penalize redundancy by cosine sim to chosen
            S = np.array(chosen, dtype=int)
            red = np.max(self.bank_emb[i] @ self.bank_emb[S].T)
            if red < 0.95:  # keep diversity
                chosen.append(int(i))
                if len(chosen)>=cap: break
        # fallback fill
        j = 0
        while len(chosen) < min(cap, len(idxs)):
            i = int(idxs[j]); j+=1
            if i not in chosen:
                chosen.append(i)
        return chosen
