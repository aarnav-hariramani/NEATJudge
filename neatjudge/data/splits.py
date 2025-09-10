
import random
from typing import Dict, List, Any

def make_splits(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    seed = int(cfg.get("data", {}).get("seed", 42))
    rng = random.Random(seed)
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question"], []).append(r)
    qs = list(by_q.keys())
    rng.shuffle(qs)
    bank_frac = float(cfg.get("data", {}).get("bank_frac", 0.6))
    val_frac  = float(cfg.get("data", {}).get("val_frac", 0.1))
    test_frac = float(cfg.get("data", {}).get("test_frac", 0.1))
    n = len(qs)
    n_bank = int(n * bank_frac)
    n_val  = int(n * val_frac)
    n_test = int(n * test_frac)
    bank_qs = set(qs[:n_bank])
    val_qs  = set(qs[n_bank:n_bank+n_val])
    test_qs = set(qs[n_bank+n_val:n_bank+n_val+n_test])
    train_qs = set(qs) - bank_qs - val_qs - test_qs
    def collect(keys):
        out = []
        for k in keys:
            out.extend(by_q[k])
        return out
    return {"bank": collect(bank_qs), "val": collect(val_qs), "test": collect(test_qs), "train": collect(train_qs)}
