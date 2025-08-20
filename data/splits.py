from typing import List, Dict, Any
import random
import numpy as np

def make_splits(rows: List[Dict[str, Any]], cfg) -> Dict[str, List[Dict[str, Any]]]:
    random.seed(cfg["data"]["seed"])
    rows = rows[:]
    random.shuffle(rows)
    n = len(rows)
    n_test = int(cfg["data"]["test_frac"] * n)
    test = rows[:n_test]
    remain = rows[n_test:]
    n_val = int(cfg["data"]["val_frac"] * len(remain))
    val = remain[:n_val]
    train = remain[n_val:]
    n_bank = int(cfg["data"]["bank_frac"] * len(train))
    bank = train[:n_bank]
    fitness_pool = train[n_bank:]
    return {"bank": bank, "fitness_pool": fitness_pool, "val": val, "test": test}

def _label_bins(rows: List[Dict[str, Any]], bins: int = 5):
    if not rows:
        return [[] for _ in range(bins)]
    labels = [r["label"] for r in rows]
    lo, hi = min(labels), max(labels)
    if hi == lo:
        # all the same; single bin
        return [list(range(len(rows)))] + [[] for _ in range(bins-1)]
    arr = np.array(labels, dtype=float)
    cats = np.floor((arr - lo) / (hi - lo + 1e-9) * bins).astype(int)
    cats = np.clip(cats, 0, bins-1)
    buckets = [[] for _ in range(bins)]
    for i, c in enumerate(cats.tolist()):
        buckets[c].append(i)
    return buckets

def resample_microbatches(pool: List[Dict[str, Any]], batch_size: int, micro_batches: int, *, seed: int):
    """Stratified resampling over label bins for stability across micro-batches."""
    rnd = random.Random(seed)
    batches = []
    idxs = list(range(len(pool)))
    buckets = _label_bins(pool, bins=5)
    for _ in range(micro_batches):
        chosen = []
        # round-robin sample from each bucket
        per_bucket = max(1, batch_size // max(1, len(buckets)))
        for b in buckets:
            if not b:
                continue
            rnd.shuffle(b)
            take = b[:per_bucket]
            chosen.extend(take)
        # fill if short
        if len(chosen) < batch_size:
            rnd.shuffle(idxs)
            for i in idxs:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= batch_size:
                    break
        batch = [pool[i] for i in chosen[:batch_size]]
        batches.append(batch)
    return batches
