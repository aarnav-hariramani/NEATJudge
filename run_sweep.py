"""Seed-averaged benchmark sweep: NEATJudge vs. baselines, mean +/- std.

Small evals are noisy, so a single split can mis-rank methods. This runs the full
benchmark across several seeds (each seed = a different train/eval split AND a
different optimizer RNG) on the larger 40-item split and reports mean +/- std of
the held-out eval metrics -- a far more reliable comparison.

    python run_sweep.py                          # live on Opus, seeds 1..5
    python run_sweep.py --mock --seeds 1,2,3
    python run_sweep.py --limit 40 --budget 240 --seeds 1,2,3,4,5

Writes docs/benchmark_sweep.md with the aggregated table.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from neatjudge import (
    AnthropicClient,
    CachingLLMClient,
    MockLLMClient,
    build_public_safety_dataset,
    train_eval_split,
)
from neatjudge.benchmark import METHODS, CountingLLMClient
from neatjudge.benchmark.harness import n_judge_agents, score_genome

PARAMS = {
    "evoprompt_ga": dict(pop=5, gens=2),
    "opro": dict(steps=8),
    "gepa_prompt": dict(steps=6),
    "neatjudge": dict(pop=5, gens=3),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed-averaged NEATJudge benchmark sweep.")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--eval-fraction", type=float, default=0.5)
    p.add_argument("--budget", type=int, default=240)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--max-tokens", type=int, default=400)
    return p.parse_args()


def _mean_std(xs):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    base = MockLLMClient() if args.mock else AnthropicClient(
        model=args.model, max_tokens=args.max_tokens)
    shared = CachingLLMClient(base)   # dedupe real API cost across all seeds/methods

    # per method -> lists across seeds
    fit = defaultdict(list)
    sacc = defaultdict(list)
    qacc = defaultdict(list)
    agents = defaultdict(list)
    calls = defaultdict(list)
    label_of = {m[0]: m[1] for m in METHODS}
    cite_of = {m[0]: m[2] for m in METHODS}
    topo_of = {m[0]: m[3] for m in METHODS}

    print(f"Sweep :: {'MOCK' if args.mock else args.model}  limit={args.limit}  "
          f"budget={args.budget}  seeds={seeds}")
    t_start = time.time()

    for seed in seeds:
        dataset = build_public_safety_dataset(limit=args.limit)
        train, evalset = train_eval_split(dataset, eval_fraction=args.eval_fraction,
                                          seed=seed)
        print(f"\n--- seed {seed} (train={len(train)}, eval={len(evalset)}) ---")
        for key, label, citation, topo, fn in METHODS:
            counting = CountingLLMClient(shared)
            rng = random.Random(seed)
            t0 = time.time()
            genome, notes = fn(train, counting, rng, args.budget, **PARAMS.get(key, {}))
            f, sa, qa = score_genome(genome, evalset, shared)
            fit[key].append(f); sacc[key].append(sa); qacc[key].append(qa)
            agents[key].append(n_judge_agents(genome)); calls[key].append(counting.calls)
            print(f"  {label:<22} eval={f:6.2f} safety={sa:.2f} quality={qa:.2f} "
                  f"agents={n_judge_agents(genome)} calls={counting.calls} "
                  f"{time.time()-t0:.0f}s")

    # ---- aggregate ----
    rows = []
    for key in [m[0] for m in METHODS]:
        fm, fs = _mean_std(fit[key])
        sm, _ = _mean_std(sacc[key])
        qm, _ = _mean_std(qacc[key])
        rows.append((key, fm, fs, sm, qm,
                     statistics.mean(agents[key]), statistics.mean(calls[key])))
    rows.sort(key=lambda r: r[1], reverse=True)

    total_s = time.time() - t_start
    header = (f"{'method':<22}{'topo':<5}{'eval mean':>10}{'std':>7}"
              f"{'safety':>8}{'quality':>8}{'agents':>7}{'calls':>7}")
    lines = [header, "-" * len(header)]
    for key, fm, fs, sm, qm, ag, ca in rows:
        lines.append(f"{label_of[key]:<22}{('yes' if topo_of[key] else 'no'):<5}"
                     f"{fm:>10.2f}{fs:>7.2f}{sm:>8.2f}{qm:>8.2f}{ag:>7.1f}{ca:>7.0f}")
    table = "\n".join(lines)

    print("\n" + "=" * 78)
    print(f"SWEEP RESULTS over seeds {seeds} (held-out eval; mean +/- std)")
    print("=" * 78)
    print(table)
    print(f"\n{shared.stats()}   total {total_s:.0f}s")

    # ---- write markdown doc ----
    md = [f"# Seed-averaged benchmark sweep — {'MOCK' if args.mock else args.model}",
          "",
          f"Seeds: `{seeds}` · dataset limit {args.limit} "
          f"(≈{int(args.limit*(1-args.eval_fraction))} train / "
          f"{int(args.limit*args.eval_fraction)} eval per seed) · "
          f"budget {args.budget} optimization calls/method · scored on held-out eval.",
          "", "Each seed is a different train/eval split and optimizer RNG; values are "
          "mean ± std across seeds (higher eval is better).", "",
          "| Method | Cite | Topo | eval mean | ±std | safety | quality | agents | calls |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, fm, fs, sm, qm, ag, ca in rows:
        md.append(f"| {label_of[key]} | {cite_of[key]} | "
                  f"{'yes' if topo_of[key] else 'no'} | {fm:.2f} | {fs:.2f} | "
                  f"{sm:.2f} | {qm:.2f} | {ag:.1f} | {ca:.0f} |")
    md += ["", "## Per-seed eval fitness", "",
           "| Method | " + " | ".join(f"seed {s}" for s in seeds) + " |",
           "|---|" + "|".join("---" for _ in seeds) + "|"]
    for key in [m[0] for m in METHODS]:
        md.append(f"| {label_of[key]} | " +
                  " | ".join(f"{v:.2f}" for v in fit[key]) + " |")
    md += ["", f"_Shared cache: {shared.stats()}. Total wall-clock {total_s:.0f}s._"]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/benchmark_sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nwrote docs/benchmark_sweep.md")


if __name__ == "__main__":
    main()
