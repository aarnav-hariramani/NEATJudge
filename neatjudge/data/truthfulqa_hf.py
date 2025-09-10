
from typing import Dict, List, Any
from datasets import load_dataset

def load_truthfulqa_hf(cfg: dict | None = None) -> List[Dict[str, Any]]:
    ds = load_dataset("EleutherAI/truthful_qa_mc", "multiple_choice")
    rows: List[Dict[str, Any]] = []
    for ex in ds["validation"]:
        q = ex["question"]
        choices = list(ex["choices"])
        gold_idx = int(ex["label"])
        for i, ans in enumerate(choices):
            rows.append({"question": q, "choice": ans, "label": 1 if i == gold_idx else 0})
    return rows
