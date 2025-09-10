
from collections import defaultdict
from typing import List, Dict, Any

def mc1_accuracy(flat_rows: List[Dict[str, Any]], scores: List[float]) -> float:
    by_q = defaultdict(list)
    for r, s in zip(flat_rows, scores):
        by_q[r["question"]].append((float(s), int(r["label"])))
    correct, total = 0, 0
    for cand in by_q.values():
        if not cand: continue
        best = max(cand, key=lambda x: x[0])
        correct += 1 if best[1] == 1 else 0
        total += 1
    return (100.0 * correct / total) if total else 0.0
