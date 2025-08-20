# data/bank_index.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _ensure_model(name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required for selectors. Please install it.")
    return SentenceTransformer(name)


def embed_sentence_transformers(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = _ensure_model(model_name)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


@dataclass
class Bank:
    examples: List[Dict[str, str]]  # keys: query, title, label


class BankIndex:
    def __init__(self, bank: Bank, embedder: Optional[str] = None):
        self.bank = bank
        self.model_name = embedder or "sentence-transformers/all-MiniLM-L6-v2"
        self.texts = [f"{ex['query']} [SEP] {ex['title']}" for ex in self.bank.examples]
        self.emb = embed_sentence_transformers(self.texts, self.model_name)

    def topk(self, query: str, title: str, k: int = 6) -> List[Dict[str, str]]:
        qtext = f"{query} [SEP] {title}"
        qemb = embed_sentence_transformers([qtext], self.model_name)[0]
        sims = self.emb @ qemb
        idx = sims.argsort()[-k:][::-1]
        return [self.bank.examples[i] for i in idx]
