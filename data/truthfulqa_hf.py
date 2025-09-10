
from typing import Dict, List, Any
from datasets import load_dataset

DATASET_ID = "rahmanidashti/truthful-qa"
DATASET_CONFIG = "multiple-choice"

def load_truthfulqa_hf(cfg: dict | None = None) -> List[Dict[str, Any]]:
    ds = load_dataset(DATASET_ID, DATASET_CONFIG)
    rows: List[Dict[str, Any]] = []
    for ex in ds["validation"]:
        q = ex["question"]
        mc1 = ex.get("mc1_targets") or {}
        choices = list(mc1.get("choices", []))
        labels  = list(mc1.get("labels", []))
        if not choices or not labels or len(choices) != len(labels):
            continue
        for ans, lab in zip(choices, labels):
            rows.append({"question": q, "choice": ans, "label": int(lab)})
    return rows
