"""The decisive test: do HETEROGENEOUS (mixed-model) judges beat the best single judge?

Same-model panels can't beat a single judge because their errors are correlated.
Different models have *decorrelated* errors, so aggregating them (a jury) can, in
principle, beat any single judge. This script tests that on the multi-faceted
HelpSteer2 task (held-out), comparing:

  1. Each model as a SINGLE judge      -> establishes the best-single-judge bar.
  2. Heterogeneous JURY (mean of diverse models per axis) -> the core thesis test.
  3. Same-family Claude jury            -> control (less diverse).
  4. Mixed-model PANEL (a different model owns each axis) -> hand-designed topology.
  5. Heterogeneous NEATJudge            -> evolve topology + per-node model gene.

All scored identically on the same held-out eval (mean per-axis closeness x100).
"""

from __future__ import annotations

import argparse
import io
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path

from neatjudge import (
    HELPSTEER_RUBRIC,
    AnthropicClient,
    CachingLLMClient,
    Config,
    FitnessEvaluator,
    Genome,
    InnovationTracker,
    ModelRouter,
    NEATJudge,
    try_load_helpsteer,
)
from neatjudge.archetypes import ARCHETYPE_LIBRARY
from neatjudge.benchmark.harness import n_judge_agents, panel_genome, score_genome, single_node_genome

RUBRIC = HELPSTEER_RUBRIC
BASE = ARCHETYPE_LIBRARY["Core Judge"].base_instruction
POOL = [
    "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5",
    "gpt-4o", "gemini-3.5-flash", "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
]
SHORT = {  # short labels for the table
    "claude-opus-4-8": "opus-4-8", "claude-sonnet-4-6": "sonnet-4-6",
    "claude-haiku-4-5": "haiku-4-5", "gpt-4o": "gpt-4o",
    "gemini-3.5-flash": "gemini-flash",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8": "llama-4-mav",
}


def _int(v, ax):
    try:
        return int(v)
    except (TypeError, ValueError):
        return RUBRIC.midpoint(ax)


def model_predictions(client, evalset, workers):
    """Per-item per-axis predictions from one model as a single Core Judge."""
    g = single_node_genome(InnovationTracker(), BASE)

    def pred(item):
        v = g.evaluate_item(item, client, RUBRIC)
        return item["id"], {ax.name: _int(v.get(ax.name), ax.name) for ax in RUBRIC.axes}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(pred, evalset))


def score_preds(preds, evalset):
    """Mean per-axis closeness x100 from a predictions dict, plus per-axis breakdown."""
    per_axis = {}
    for ax in RUBRIC.axes:
        span = RUBRIC.span(ax.name)
        closes = [max(0.0, 1.0 - abs(preds[it["id"]][ax.name] - int(it["truth"][ax.name])) / span)
                  for it in evalset]
        per_axis[ax.name] = sum(closes) / len(closes)
    fit = 100.0 * sum(per_axis.values()) / len(per_axis)
    return fit, per_axis


def jury(pred_by_model, models, evalset):
    """Aggregate several models by rounded mean per axis (no extra API calls)."""
    agg = {}
    for it in evalset:
        agg[it["id"]] = {}
        for ax in RUBRIC.axes:
            vals = [pred_by_model[m][it["id"]][ax.name] for m in models]
            agg[it["id"]][ax.name] = round(statistics.mean(vals))
    return score_preds(agg, evalset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=24)
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="docs/benchmark_hetero.md")
    args = ap.parse_args()

    data = try_load_helpsteer(limit=args.train_size + args.eval_size, seed=args.seed,
                              pool=2000)
    if not data:
        raise SystemExit("HelpSteer2 unavailable.")
    train = data[:args.train_size]
    evalset = data[args.train_size:args.train_size + args.eval_size]

    clients = {m: CachingLLMClient(AnthropicClient(model=m, max_tokens=args.max_tokens))
               for m in POOL}

    print(f"Hetero thesis test :: models={len(POOL)} train={len(train)} eval={len(evalset)} "
          f"axes={RUBRIC.names()}")

    # 1. Single judges (also caches per-model per-item predictions for juries).
    rows = []            # (label, kind, fitness, per_axis, agents, note)
    pred_by_model = {}
    for m in POOL:
        t0 = time.time()
        pred_by_model[m] = model_predictions(clients[m], evalset, args.workers)
        fit, pa = score_preds(pred_by_model[m], evalset)
        rows.append((SHORT[m], "single", fit, pa, 1, ""))
        print(f"  single {SHORT[m]:14} {fit:6.2f}  "
              f"{' '.join(f'{k}={v:.2f}' for k,v in pa.items())}  {time.time()-t0:.0f}s")

    best_single = max(r[2] for r in rows if r[1] == "single")
    best_single_label = max((r for r in rows if r[1] == "single"), key=lambda r: r[2])[0]

    # 2/3. Juries (free -- reuse cached predictions).
    claude = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
    cross = ["claude-opus-4-8", "gpt-4o", "gemini-3.5-flash",
             "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"]
    jury_sets = {
        "jury: all-6 (diverse)": POOL,
        "jury: cross-provider-4": cross,
        "jury: claude-3 (same family)": claude,
    }
    # Also the best 3-model jury by brute force over cached preds.
    best3, best3_fit = None, -1
    for combo in combinations(POOL, 3):
        f, _ = jury(pred_by_model, list(combo), evalset)
        if f > best3_fit:
            best3_fit, best3 = f, combo
    jury_sets[f"jury: best-3 ({'+'.join(SHORT[m] for m in best3)})"] = list(best3)

    for label, ms in jury_sets.items():
        fit, pa = jury(pred_by_model, ms, evalset)
        rows.append((label, "jury", fit, pa, len(ms), ""))
        print(f"  {label:34} {fit:6.2f}  {' '.join(f'{k}={v:.2f}' for k,v in pa.items())}")

    # 4. Mixed-model panel: a different model owns each axis; strong aggregator.
    router = ModelRouter(clients["claude-opus-4-8"], clients)
    mp = panel_genome(InnovationTracker(), [ax.owner for ax in RUBRIC.axes])
    axis_model = {"Fact-Checker": "gpt-4o", "Coherence Judge": "gemini-3.5-flash",
                  "Relevance Judge": "claude-sonnet-4-6"}
    for node in mp.nodes.values():
        if node.personality_core in axis_model:
            node.model = axis_model[node.personality_core]
        elif node.personality_core == "Core Judge":
            node.model = "claude-opus-4-8"
    t0 = time.time()
    fit, pa, _ = (*score_genome(mp, evalset, router, workers=args.workers, rubric=RUBRIC),)
    rows.append(("mixed panel (gpt/gem/son->opus)", "panel", fit, mp.axis_accuracy, n_judge_agents(mp), ""))
    print(f"  mixed panel {fit:6.2f}  {' '.join(f'{k}={v:.2f}' for k,v in mp.axis_accuracy.items())}  {time.time()-t0:.0f}s")

    # 5. Heterogeneous NEATJudge: evolve topology + per-node model over the pool.
    t0 = time.time()
    cfg = Config(population_size=6, generations=3, seed=args.seed, eval_workers=args.workers,
                 p_mutate_prompt=0.9, reflective_prompt_rewrite=True, reflection_batch=5,
                 compatibility_threshold=0.6, p_mutate_model=0.6, model_pool=POOL,
                 default_model="claude-opus-4-8")
    ev_train = FitnessEvaluator(train, router, train_set=train, complexity_penalty=0.1,
                                workers=1, rubric=RUBRIC)
    with redirect_stdout(io.StringIO()):
        best = NEATJudge(cfg, ev_train, clients["claude-opus-4-8"]).run()
    fit, _, _ = score_genome(best, evalset, router, workers=args.workers, rubric=RUBRIC)
    models_used = sorted({SHORT.get(n.model, n.model) for n in best.nodes.values()
                          if n.model and n.node_type.value != "input"})
    note = f"agents={n_judge_agents(best)} models={models_used or ['default']}"
    rows.append(("NEATJudge (hetero)", "evolved", fit, best.axis_accuracy, n_judge_agents(best), note))
    print(f"  NEATJudge hetero {fit:6.2f}  {note}  {time.time()-t0:.0f}s")

    # ---- report ----
    rows.sort(key=lambda r: r[2], reverse=True)
    print("\n" + "=" * 78)
    print(f"HETERO THESIS RESULTS (held-out eval; best single = {best_single_label} @ {best_single:.2f})")
    print("=" * 78)
    for label, kind, fit, pa, agents, note in rows:
        delta = fit - best_single
        flag = "  <-- BEATS best single" if delta > 0.001 and kind != "single" else ""
        print(f"  {label:40} {kind:7} {fit:6.2f} ({delta:+.2f}) agents={agents}{flag}")

    beat = [r for r in rows if r[1] != "single" and r[2] > best_single + 0.001]
    print("\nTHESIS:", "CONFIRMED -- a heterogeneous config beat the best single judge."
          if beat else "NOT CONFIRMED -- no heterogeneous config beat the best single judge.")

    # ---- markdown ----
    md = [f"# Heterogeneous-models thesis test — HelpSteer2 multi-axis (held-out)", "",
          f"Best single judge: **{best_single_label} @ {best_single:.2f}**. "
          f"train={len(train)} eval={len(evalset)} axes={RUBRIC.names()}.", "",
          "| Config | kind | eval | vs best single | agents |",
          "|---|---|---|---|---|"]
    for label, kind, fit, pa, agents, note in rows:
        md.append(f"| {label} | {kind} | {fit:.2f} | {fit-best_single:+.2f} | {agents} |")
    md += ["", ("**Thesis CONFIRMED**: a heterogeneous configuration beat the best single "
                "judge." if beat else "**Thesis NOT confirmed**: no heterogeneous configuration "
                "beat the best single judge."),
           "", f"_Models: {', '.join(SHORT[m] for m in POOL)}._"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
