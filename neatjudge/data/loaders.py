from typing import List
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class QAExample:
    question: str
    options: List[str]
    correct_idx: int
    incorrect_idxes: List[int]

def _flatten_int(x):
    # Coerce labels like 1, [1], [[1]] → 1
    while isinstance(x, list):
        x = x[0] if x else 0
    return int(x)

def load_truthfulqa_hf(split: str = "validation", subset: str = "multiple_choice") -> List[QAExample]:
    """
    Loads the catalog 'truthful_qa' dataset with subset 'multiple_choice',
    handling mc1_targets / choices, and normalizing odd label shapes.
    """
    ds = load_dataset("truthful_qa", subset, split=split)
    out: List[QAExample] = []

    for row in ds:
        q = row["question"]

        # Normalize choices and labels
        if "choices" in row and row["choices"]:
            options = list(row["choices"])
            mc1 = row.get("mc1_targets", {})
            labels = []
            for ans in options:
                val = mc1.get(ans, 0)
                labels.append(_flatten_int(val))
        else:
            mc1 = row.get("mc1_targets", {})
            options = list(mc1.keys())
            labels = [_flatten_int(v) for v in mc1.values()]

        if not options:
            continue

        # MC1 requires exactly one correct = 1
        if not all(v in (0, 1) for v in labels) or sum(labels) != 1:
            # Skip rows that aren't single-answer MC1
            continue

        correct_idx = labels.index(1)
        incorrect_idxes = [i for i, v in enumerate(labels) if i != correct_idx]
        out.append(QAExample(q, options, correct_idx, incorrect_idxes))

    return out

def load_dataset_from_cfg(cfg) -> List[QAExample]:
    data_cfg = (cfg.get("data", {}) or {})
    ds = (data_cfg.get("dataset", "") or "").lower()
    split = data_cfg.get("split", "validation")
    subset = data_cfg.get("subset", "multiple_choice")

    if ds in ("truthfulqa_hf", "truthful_qa", "truthfulqa"):
        return load_truthfulqa_hf(split, subset)

    raise ValueError(f"Unsupported dataset '{ds}'. Use 'truthfulqa_hf' with subset 'multiple_choice'.")
