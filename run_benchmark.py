"""Benchmark: NEATJudge vs. non-NEAT LLM-as-a-judge optimizers.

All methods share one harness -- same base model, same train/eval split, same
scoring, per-method LLM-call budget -- so the comparison is apples-to-apples.
Methods optimize on TRAIN and are scored on the held-out EVAL split.

    python run_benchmark.py                 # live on Opus (needs ANTHROPIC_API_KEY)
    python run_benchmark.py --mock          # deterministic offline smoke run
    python run_benchmark.py --limit 24 --budget 240 --eval-fraction 0.5

See docs/benchmark_results.md for a recorded live run and citations.
"""

from __future__ import annotations

import argparse
import random
import time

from neatjudge import (
    AnthropicClient,
    CachingLLMClient,
    MockLLMClient,
    build_public_safety_dataset,
    train_eval_split,
)
from neatjudge.benchmark import METHODS, CountingLLMClient, format_table
from neatjudge.benchmark.harness import BenchmarkResult, n_judge_agents, score_genome

# Per-method optimizer scale (kept modest so a live run is affordable; the
# per-method LLM-call budget is the hard cap that keeps the comparison fair).
PARAMS = {
    "evoprompt_ga": dict(pop=5, gens=2),
    "opro": dict(steps=8),
    "gepa_prompt": dict(steps=6),
    "neatjudge": dict(pop=5, gens=3),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark NEATJudge vs baselines.")
    p.add_argument("--mock", action="store_true", help="use deterministic MockLLMClient")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--limit", type=int, default=24, help="dataset items used")
    p.add_argument("--eval-fraction", type=float, default=0.5)
    p.add_argument("--budget", type=int, default=240,
                   help="max optimization LLM requests per method")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--methods", default="", help="comma-separated subset of method keys")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = MockLLMClient() if args.mock else AnthropicClient(
        model=args.model, max_tokens=args.max_tokens)
    shared = CachingLLMClient(base)   # dedupes real API cost across all methods

    dataset = build_public_safety_dataset(limit=args.limit)
    train, evalset = train_eval_split(dataset, eval_fraction=args.eval_fraction,
                                      seed=args.seed)

    wanted = set(filter(None, args.methods.split(","))) if args.methods else None
    methods = [m for m in METHODS if (wanted is None or m[0] in wanted)]

    print(f"Benchmark :: {'MOCK' if args.mock else args.model}  "
          f"dataset={len(dataset)} (train={len(train)}, eval={len(evalset)})  "
          f"budget={args.budget}/method")
    print(f"methods: {', '.join(m[0] for m in methods)}\n")

    results = []
    for key, label, citation, evolves_topology, fn in methods:
        counting = CountingLLMClient(shared)   # counts this method's optimization work
        rng = random.Random(args.seed)
        t0 = time.time()
        genome, notes = fn(train, counting, rng, args.budget, **PARAMS.get(key, {}))
        # Score on the held-out eval split with the shared (uncounted) client, so
        # `requests` reflects optimization only and is identical scoring for all.
        fit, sacc, qacc = score_genome(genome, evalset, shared)
        secs = time.time() - t0
        res = BenchmarkResult(
            method=label, citation=citation, evolves_topology=evolves_topology,
            eval_fitness=fit, safety_acc=sacc, quality_acc=qacc,
            n_agents=n_judge_agents(genome), requests=counting.calls,
            seconds=secs, notes=notes)
        results.append(res)
        print(f"  done {label:<22} eval={fit:6.2f}  safety={sacc:.2f}  "
              f"quality={qacc:.2f}  agents={res.n_agents}  calls={res.requests}  "
              f"{secs:.0f}s  [{notes}]")

    print("\n" + "=" * 78)
    print("BENCHMARK RESULTS (scored on held-out eval split; higher is better)")
    print("=" * 78)
    print(format_table(results))
    print(f"\n{shared.stats()}")
    print("\nCitations:")
    for r in sorted(results, key=lambda x: x.eval_fitness, reverse=True):
        print(f"  {r.method:<22} {r.citation}")


if __name__ == "__main__":
    main()
