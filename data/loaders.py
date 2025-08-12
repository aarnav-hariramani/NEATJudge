
from pathlib import Path
import os, json, pandas as pd
from typing import List, Dict, Any

def load_ebay(cfg) -> List[Dict[str, Any]]:
    path = Path(os.getenv("EBAY_XLSX", cfg["data"]["paths"]["ebay"])).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"eBay file not found: {path}")
    cols = cfg["data"]["columns"]
    df = pd.read_excel(path)
    df = df[[cols["history"], cols["product"], cols["label"]]].dropna()
    rows = []
    for h, t, l in zip(df[cols["history"]], df[cols["product"]], df[cols["label"]]):
        rows.append({"query": str(h), "response": str(t), "label": float(l)})
    return rows

def load_truthfulqa(cfg) -> List[Dict[str, Any]]:
    path = Path(cfg["data"]["paths"]["truthfulqa"]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"TruthfulQA file not found: {path}")
    data = json.loads(Path(path).read_text())
    rows = []
    for entry in data:
        q = entry["question"]
        for ans, lab in entry["mc1_targets"].items():
            rows.append({"query": str(q), "response": str(ans), "label": float(lab)})
    return rows

def load_dataset(cfg) -> List[Dict[str, Any]]:
    ds = cfg["data"]["dataset"].lower()
    if ds == "ebay":
        return load_ebay(cfg)
    elif ds == "truthfulqa":
        return load_truthfulqa(cfg)
    else:
        raise ValueError(f"Unknown dataset: {ds}")
