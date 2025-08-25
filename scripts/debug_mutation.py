# scripts/debug_mutation.py
from __future__ import annotations
import argparse, os, random, sys
from pathlib import Path

# --- make repo root importable even when running from scripts/ ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.io import read_yaml
from data.loaders import load_dataset
from data.splits import make_splits
from model.prompt import assemble_prompt, DEFAULT_HEADER
from model.metrics import mae_accuracy
from model.prompt_genome import _is_valid_header
from llms.judge import Judge

try:
    from llms.mutate import mutate_prompt
except Exception:
    mutate_prompt = None

def _eval_header(header, rows, judge, repeats, label_range):
    preds, labs = [], []
    for r in rows:
        prompt = assemble_prompt(header, r["query"], r["response"], None)
        scores = judge.score(prompt, repeats=repeats)  # list[float]
        if scores:
            preds.append(sum(scores)/len(scores))
            labs.append(r["label"])
    mae, acc = mae_accuracy(preds, labs, label_range)
    return mae, acc, len(header.split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--n_iters", type=int, default=6)
    ap.add_argument("--val_size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--accept_margin", type=float, default=0.25)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--start_header", type=str, default="")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    random.seed(args.seed)

    rows = load_dataset(cfg)
    splits = make_splits(rows, cfg)
    val = splits["val"][:]
    random.shuffle(val)
    slice_rows = val[:args.val_size]

    jcfg = cfg["judge"]
    judge = Judge(model=jcfg["model"], base_url=jcfg["base_url"], temperature=jcfg.get("temperature", 0.0))

    header = Path(args.start_header).read_text(encoding="utf-8") if args.start_header and Path(args.start_header).exists() else DEFAULT_HEADER

    cur_mae, cur_acc, cur_len = _eval_header(header, slice_rows, judge, repeats=args.repeats, label_range=cfg["data"]["label_range"])
    print(f"[BASELINE] acc={cur_acc:.2f} mae={cur_mae:.2f} words={cur_len}")

    out_dir = REPO_ROOT / "runs" / "mutation_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "mutation_trace.tsv"
    trace_path.write_text("iter\told_acc\tnew_acc\tdelta_acc\told_words\tnew_words\taccepted\n", encoding="utf-8")

    if mutate_prompt is None:
        print("[warn] mutate_prompt unavailable; exiting.")
        return

    for i in range(1, args.n_iters + 1):
        candidate = mutate_prompt(header)

        def ok(h: str) -> bool:
            if not h:
                return False
            t = h.lower()
            # must include 1..5 scale and a "rating" output reference
            return all((f"{k} =" in t or f"{k}=" in t) for k in "12345") and ("rating" in t)

        if not ok(candidate):
            new_mae, new_acc, new_len = float("nan"), float("nan"), len(candidate.split()) if candidate else 0
            accepted = False
            delta = float("nan")
        else:
            new_mae, new_acc, new_len = _eval_header(candidate, slice_rows, judge, repeats=args.repeats, label_range=cfg["data"]["label_range"])
            delta = new_acc - cur_acc
            # simple acceptance: require margin and anti-bloat
            accepted = (delta >= args.accept_margin) and (new_len <= int(cur_len * 1.25))

        with trace_path.open("a", encoding="utf-8") as f:
            f.write(f"{i}\t{cur_acc:.2f}\t{new_acc if new_acc==new_acc else 0.0:.2f}\t{delta if delta==delta else 0.0:.2f}\t{cur_len}\t{new_len}\t{int(accepted)}\n")

        (out_dir / f"iter{i:02d}_old.txt").write_text(header, encoding="utf-8")
        (out_dir / f"iter{i:02d}_new.txt").write_text(candidate or "", encoding="utf-8")

        print(f"[iter {i}] new_acc={new_acc if new_acc==new_acc else float('nan'):.2f} {'ACCEPTED' if accepted else 'rejected'} (old_acc={cur_acc:.2f})")

        if accepted:
            header, cur_mae, cur_acc, cur_len = candidate, new_mae, new_acc, new_len

    print(f"Trace written to {trace_path}")

if __name__ == "__main__":
    main()
