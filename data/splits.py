from typing import List, Dict, Any, Tuple
import random
from .loaders import QAExample

def split_bank_val(rows: List[QAExample], val_frac: float, seed: int):
    rnd = random.Random(seed)
    rows = rows[:]
    rnd.shuffle(rows)
    n = len(rows)
    n_val = int(val_frac * n)
    val = rows[:n_val]
    bank = rows[n_val:]
    return bank, val
