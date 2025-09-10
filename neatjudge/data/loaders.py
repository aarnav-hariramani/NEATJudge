from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class QAExample:
    question: str
    options: List[str]
    correct_idx: int
    incorrect_idxes: List[int]

def load_truthfulqa_hf(split: str = "validation", subset: str = "multiple_choice") -> List[QAExample]:
    ds = load_dataset("truthful_qa", subset, split=split)
    out: List[QAExample] = []
    for row in ds:
        q = row["question"]
        # 'mc1_targets' is a dict of {answer: 1 or 0}
        # 'choices' may also exist depending on HF version; we normalize to options[]
        if "choices" in row and row["choices"]:
            options = list(row["choices"])
            labels = [1 if ans in row.get("mc1_targets", {}) and row["mc1_targets"][ans]==1 else 0 for ans in options]
        else:
            options = list(row["mc1_targets"].keys())
            labels = list(row["mc1_targets"].values())
        if not options:
            continue
        # MC1: exactly one correct
        if sum(labels) != 1:
            # skip non-standard rows
            continue
        correct_idx = labels.index(1)
        incorrect_idxes = [i for i,l in enumerate(labels) if l==0]
        out.append(QAExample(question=q, options=options, correct_idx=correct_idx, incorrect_idxes=incorrect_idxes))
    return out

def load_dataset_from_cfg(cfg) -> List[QAExample]:
    ds = (cfg.get("data", {}) or {}).get("dataset", "")
    if ds.lower() == "truthfulqa_hf":
        return load_truthfulqa_hf(cfg["data"]["split"], cfg["data"]["subset"])
    raise ValueError(f"Unsupported dataset {ds}")
