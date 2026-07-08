"""Live NEATJudge evolution against real Claude models via the Anthropic API.

Requires ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_BASE_URL`` for a
gateway) in the environment, plus ``pip install anthropic``.

The judges run on the CLEAN golden dataset (real prompt/response text, hidden
labels) so the model must actually reason -- nothing about the ground truth is
visible in the prompt. Agents are dispatched through a ModelRouter, exercising
the model-mutating gene's routing path even when pinned to a single model.

    python run_live.py                          # defaults: opus-4-8, pop 12, 3 gens
    python run_live.py --model claude-haiku-4-5 --pop 6 --generations 2 --workers 8
    python run_live.py --evolve-models          # let the model gene explore the pool
"""

from __future__ import annotations

import argparse

from neatjudge import (
    AnthropicClient,
    Config,
    FitnessEvaluator,
    ModelRouter,
    NEATJudge,
    build_golden_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NEATJudge live on Claude.")
    p.add_argument("--model", default="claude-opus-4-8",
                   help="default model every agent runs on (default: claude-opus-4-8)")
    p.add_argument("--pop", type=int, default=12, help="population size")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--workers", type=int, default=8,
                   help="parallel genome evaluations (real API is IO-bound)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--evolve-models", action="store_true",
                   help="enable the model-mutating gene over an Opus/Sonnet/Haiku pool "
                        "with cost-aware fitness")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # One shared client per model id (SDK clients are safe to share across threads).
    default_client = AnthropicClient(model=args.model, max_tokens=args.max_tokens)
    clients = {args.model: default_client}

    model_pool = [args.model]
    p_mutate_model = 0.0
    cost_weight = 0.0

    if args.evolve_models:
        model_pool = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
        p_mutate_model = 0.5
        cost_weight = 0.05
        for m in model_pool:
            clients.setdefault(m, AnthropicClient(model=m, max_tokens=args.max_tokens))

    router = ModelRouter(default_client, clients)          # judges route per model gene
    critic = AnthropicClient(model=args.model, max_tokens=args.max_tokens)

    cfg = Config(
        population_size=args.pop,
        generations=args.generations,
        seed=args.seed,
        eval_workers=args.workers,
        p_mutate_model=p_mutate_model,
        model_pool=model_pool,
        default_model=args.model,
        model_cost_weight=cost_weight,
    )

    dataset = build_golden_dataset()   # clean text, hidden labels -> real judging
    evaluator = FitnessEvaluator(
        dataset, router,
        default_model=args.model, model_cost_weight=cost_weight,
    )

    print(f"Live run :: default model={args.model}  pop={args.pop}  "
          f"gens={args.generations}  workers={args.workers}  "
          f"evolve_models={args.evolve_models}")
    engine = NEATJudge(cfg, evaluator, critic)
    best = engine.run()

    print("\nFinal verdicts from the best evolved judge (clean, unlabeled items):")
    correct = 0
    for sample in dataset:
        verdict = best.evaluate_item(sample, router)
        truth = sample["truth"]
        ok = verdict["safety"] == truth["safety"] and verdict["quality"] == truth["quality"]
        correct += ok
        print(f"  {'OK ' if ok else '-- '}{sample['id']}: "
              f"verdict={verdict}  truth={truth}")
    print(f"\nExact-match items: {correct}/{len(dataset)}")


if __name__ == "__main__":
    main()
