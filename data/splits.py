
from typing import List, Dict, Any
import random

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

def resample_microbatches(pool: List[Dict[str, Any]], batch_size: int, micro_batches: int, *, seed: int):
    rnd = random.Random(seed)
    batches = []
    for _ in range(micro_batches):
        rnd.shuffle(pool)
        batches.append(pool[:batch_size])
    return batches
