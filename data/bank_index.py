
from typing import List, Dict, Any, Sequence
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

class BankIndex:
    def __init__(self, bank_rows: List[Dict[str, Any]], embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.bank_rows = bank_rows
        self.embed_model_name = embed_model
        self.model = SentenceTransformer(embed_model) if SentenceTransformer else None
        texts = [f"Q: {r['question']}\nA: {r.get('choice','')}" for r in bank_rows]
        self.bank_emb = self._encode(texts)
        self.bank_texts = texts

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            rng = np.random.RandomState(0)
            return rng.randn(len(texts), 384).astype("float32")
        embs = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        return (embs / norms).astype("float32")

    def embed_dim(self) -> int:
        return int(self.bank_emb.shape[1])

    def encode_query(self, q: str) -> np.ndarray:
        return self._encode([q])[0]

    def shortlist(self, q_emb: np.ndarray, top_k: int = 64) -> List[int]:
        sims = self.bank_emb @ q_emb
        order = np.argsort(-sims)[:top_k]
        return [int(i) for i in order]

    def filter_similar(self, idxs: List[int], q_emb: np.ndarray, cap: int = 32) -> List[int]:
        sims = self.bank_emb[idxs] @ q_emb
        order = np.argsort(-sims)[:cap]
        return [idxs[int(i)] for i in order]

    def examples(self, idxs: List[int]) -> List[str]:
        return [self.bank_texts[i] for i in idxs]
