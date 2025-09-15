from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class QAExample:
    question: str
    options: List[str]
    correct_idx: int
    incorrect_idxes: List[int]

# --- NEW: EleutherAI/truthful_qa_mc loader (recommended) ----------------------

def load_truthfulqa_eleutherai_mc(split: str = "validation") -> List[QAExample]:
    """
    Loads the multiple-choice TruthfulQA variant published by EleutherAI:
      - dataset: EleutherAI/truthful_qa_mc
      - fields:  question (str), choices (List[str]), label (int; index into choices)
    """
    ds = load_dataset("EleutherAI/truthful_qa_mc", split=split)
    out: List[QAExample] = []
    for ex in ds:
        q = ex["question"]
        options = list(ex["choices"])
        label = ex["label"]

        # Normalize label to an int index
        if isinstance(label, (list, tuple)):
            # Some mirrors can wrap it; take the first if present
            label = int(label[0]) if len(label) > 0 else -1
        else:
            label = int(label)

        # Guard against malformed rows
        if not options or label < 0 or label >= len(options):
            continue

        correct_idx = label
        incorrect_idxes = [i for i in range(len(options)) if i != correct_idx]
        out.append(QAExample(question=q, options=options,
                             correct_idx=correct_idx, incorrect_idxes=incorrect_idxes))
    return out

# --- (Optional) legacy HF loader kept for reference/compat --------------------

def load_truthfulqa_hf(split: str = "validation", subset: str = "multiple_choice") -> List[QAExample]:
    """
    Legacy loader for various TruthfulQA mirrors that expose mc1_targets/choices.
    You can keep this for experimentation, but for clean MC1 baselines prefer
    load_truthfulqa_eleutherai_mc above.
    """
    ds = load_dataset("truthful_qa", subset, split=split)
    out: List[QAExample] = []
    for row in ds:
        q = row["question"]

        # Normalize choices/labels into options + one-hot labels
        if "choices" in row and row["choices"]:
            options = list(row["choices"])
            mc1 = row.get("mc1_targets", {})
            labels = []
            for ans in options:
                val = mc1.get(ans, 0)
                if isinstance(val, list):
                    # flatten any nesting (e.g., [1])
                    val = val[0] if len(val) else 0
                labels.append(int(val))
        else:
            mc1 = row.get("mc1_targets", {})
            options = list(mc1.keys())
            labels = []
            for v in mc1.values():
                if isinstance(v, list):
                    v = v[0] if len(v) else 0
                labels.append(int(v))

        if not options:
            continue

        # MC1 requires exactly one correct
        if sum(labels) != 1:
            continue

        correct_idx = labels.index(1)
        incorrect_idxes = [i for i, l in enumerate(labels) if l == 0]
        out.append(QAExample(question=q, options=options,
                             correct_idx=correct_idx, incorrect_idxes=incorrect_idxes))
    return out

# --- Config routing -----------------------------------------------------------

def load_dataset_from_cfg(cfg) -> List[QAExample]:
    """
    Config keys (example):
      data:
        dataset: truthfulqa_eleutherai_mc   # <-- use this
        split: validation
        # subset: (ignored for EleutherAI MC)
    """
    data_cfg = (cfg.get("data", {}) or {})
    ds = (data_cfg.get("dataset", "") or "").lower()
    split = data_cfg.get("split", "validation")
    subset = data_cfg.get("subset", "multiple_choice")

    if ds in ("truthfulqa_eleutherai_mc", "truthfulqa_mc", "eleutherai_truthfulqa_mc"):
        return load_truthfulqa_eleutherai_mc(split)
    elif ds in ("truthfulqa_hf", "truthfulqa"):
        return load_truthfulqa_hf(split, subset)
    else:
        raise ValueError(f"Unsupported dataset '{ds}'. "
                         f"Use 'truthfulqa_eleutherai_mc' for EleutherAI/truthful_qa_mc, "
                         f"or 'truthfulqa_hf' for legacy loaders.")
