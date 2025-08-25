# scripts/debug_mutation.py
from __future__ import annotations
import argparse, os, random, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.io import read_yaml
from utils.io import read_yaml
from data.loaders import load_dataset
from data.splits import make_splits
from model.prompt import assemble_prompt, DEFAULT_HEADER
from model.metrics import mae_accuracy
from llms.judge import Judge
try:
    from llms.mutate import mutate_prompt
except Exception:
    mutate_prompt = None

def _eval_header(header: str, rows: List[Dict[str, Any]], judge: Judge, repeats: int, label_range: float) -> Tuple[float, float, float]:
    """Return (mae, acc, avg_len) on these rows (no few-shots; isolates header)."""
    preds, labs = [], []
    for r in rows:
        prompt = assemble_prompt(header, r["query"], r["response"], None)
        scores = judge.score(prompt, repeats=repeats)  # list[float]
        if scores:
            preds.append(sum(scores)/len(scores))
            labs.append(r["label"])
    mae, acc = mae_accuracy(preds, labs, label_range)
    avg_len = len(header.split())
    return mae, acc, avg_len

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n_iters", type=int, default=6)
    ap.add_argument("--val_size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--accept_margin", type=float, default=0.25, help="min accuracy gain (percentage points) to accept")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--start_header", type=str, default="", help="optional path to a starting header .txt")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    random.seed(args.seed)

    # Data
    rows = load_dataset(cfg)
    splits = make_splits(rows, cfg)
    val = splits["val"][:]
    random.shuffle(val)
    slice_rows = val[:args.val_size]

    # LLM Judge (same config you already use)
    jcfg = cfg["judge"]
    judge = Judge(model=jcfg["model"], base_url=jcfg["base_url"], temperature=jcfg.get("temperature", 0.0))

    # Header
    if args.start_header and Path(args.start_header).exists():
        header = Path(args.start_header).read_text(encoding="utf-8")
    else:
        header = DEFAULT_HEADER

    # Baseline
    mae, acc, avg_len = _eval_header(header, slice_rows, judge, repeats=args.repeats, label_range=cfg["data"]["label_range"])
    print(f"[BASELINE] acc={acc:.2f} mae={mae:.2f} words={avg_len}")

    out_dir = Path("runs/mutation_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "mutation_trace.tsv"
    with trace_path.open("w", encoding="utf-8") as f:
        f.write("iter\told_acc\tnew_acc\tdelta_acc\told_words\tnew_words\taccepted\n")

    for i in range(1, args.n_iters + 1):
        if mutate_prompt is None:
            print("[warn] mutate_prompt unavailable; skipping.")
            break

        # Propose
        candidate = mutate_prompt(header)

        # Validate invariants quickly; if invalid, skip
        def ok(h: str) -> bool:
            t = (h or "").lower()
            # must include 1..5 lines (loosely)
            ok_scale = all((f"{k} =" in t or f"{k}=" in t) for k in "12345")
            # must mention output/rating JSON somewhere
            has_rating = ("rating" in t)
            return ok_scale and has_rating

        if not ok(candidate):
            accepted = False
            new_mae, new_acc, new_len = 0.0, 0.0, len(candidate.split()) if candidate else 0
        else:
            # Evaluate both on SAME slice to get a clean delta
            old_mae, old_acc, old_len = mae, acc, avg_len
            new_mae, new_acc, new_len = _eval_header(candidate, slice_rows, judge, repeats=args.repeats, label_range=cfg["data"]["label_range"])
            d_acc = new_acc - old_acc

            accepted = d_acc >= args.accept_margin and new_len <= int(old_len * 1.25)

            # Accept or revert
            if accepted:
                header, mae, acc, avg_len = candidate, new_mae, new_acc, new_len

        # Log
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(f"{i}\t{acc:.2f if 'old_acc' in locals() else 0.0}\t{new_acc:.2f}\t{(new_acc - (locals().get('old_acc', acc))):.2f}\t{avg_len}\t{new_len}\t{int(accepted)}\n")

        # Save snapshots for quick eyeballing
        (out_dir / f"iter{i:02d}_old.txt").write_text(header, encoding="utf-8")
        (out_dir / f"iter{i:02d}_new.txt").write_text(candidate or "", encoding="utf-8")

        print(f"[iter {i}] new_acc={new_acc:.2f} {'ACCEPTED' if accepted else 'rejected'} (old_acc={acc:.2f})")

    print(f"Trace written to {trace_path}")

if __name__ == "__main__":
    main()
