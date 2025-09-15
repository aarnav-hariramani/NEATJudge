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
    Loads the catalog 'truthful_qa' dataset (subset='multiple_choice') and normalizes
    MC1 targets across schema variants:
      A) row.choices + row.mc1_targets.labels
      B) row.mc1_targets.choices + row.mc1_targets.labels
      C) row.mc1_targets is a mapping {answer_str: 0/1}
    """
    ds = load_dataset("truthful_qa", subset, split=split)
    out: List[QAExample] = []

    for row in ds:
        q = row["question"]
        mc1 = row.get("mc1_targets", {})

        options: List[str] = []
        labels: List[int] = []

        # Case A: top-level choices
        if isinstance(row.get("choices"), list) and row["choices"]:
            options = list(row["choices"])
            lab = (mc1 or {}).get("labels")
            if isinstance(lab, list) and lab:
                labels = [_flatten_int(v) for v in lab]

        # Case B: mc1_targets contains choices/labels
        elif isinstance(mc1, dict) and "choices" in mc1 and "labels" in mc1:
            if isinstance(mc1["choices"], list):
                options = list(mc1["choices"])
            lab = mc1.get("labels")
            if isinstance(lab, list) and lab:
                labels = [_flatten_int(v) for v in lab]

        # Case C: legacy mapping answer->0/1
        elif isinstance(mc1, dict) and mc1:
            # Only if values look like 0/1 (or nested ints)
            try:
                keys = list(mc1.keys())
                vals = [_flatten_int(v) for v in mc1.values()]
                # Heuristic: labels must be only 0/1 to accept this path
                if all(v in (0, 1) for v in vals):
                    options = keys
                    labels = vals
            except Exception:
                pass  # fall through to skip

        # Guard: must have options and labels of same length
        if not options or not labels or len(options) != len(labels):
            continue

        # MC1 requires exactly one correct answer (one-hot)
        if not all(v in (0, 1) for v in labels) or sum(labels) != 1:
            continue

        correct_idx = labels.index(1)
        incorrect_idxes = [i for i in range(len(options)) if i != correct_idx]
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
