"""Live NEATJudge evolution against real Claude models via the Anthropic API.

Requires ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_BASE_URL`` for a
gateway) in the environment, plus ``pip install anthropic``.

Judges run on the bundled public safety dataset (clean prompt/response text,
hidden labels) split into disjoint train / eval halves:

  * reflective prompt mutation optimizes each agent's instruction on the TRAIN
    split (the critic reads the agent's real mistakes and rewrites its prompt);
  * fitness is scored on the held-out EVAL split.

All judge calls go through a CachingLLMClient so duplicate prompts across similar
genomes hit cache instead of the API.

    python run_live.py                        # opus-4-8, pop 10, 4 gens, reflective
    python run_live.py --no-reflect           # ablation: reflection off
    python run_live.py --evolve-models        # also evolve per-agent model choice
"""

from __future__ import annotations

import argparse

from neatjudge import (
    AnthropicClient,
    CachingLLMClient,
    Config,
    FitnessEvaluator,
    ModelRouter,
    NEATJudge,
    build_public_safety_dataset,
    train_eval_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run NEATJudge live on Claude.")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--pop", type=int, default=10)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--limit", type=int, default=None, help="cap dataset size")
    p.add_argument("--eval-fraction", type=float, default=0.5)
    p.add_argument("--complexity-penalty", type=float, default=0.1,
                   help="per-hidden-agent penalty (low so specialists can earn keep)")
    p.add_argument("--safety-weight", type=float, default=0.75,
                   help="weight on the objective safety axis vs quality")
    p.add_argument("--no-reflect", action="store_true",
                   help="disable reflective prompt rewriting (ablation)")
    p.add_argument("--evolve-models", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reflective = not args.no_reflect

    # One cached client per model id (SDK clients are thread-safe to share).
    def make(model):
        return CachingLLMClient(AnthropicClient(model=model, max_tokens=args.max_tokens))

    default_client = make(args.model)
    clients = {args.model: default_client}
    model_pool, p_mutate_model, cost_weight = [args.model], 0.0, 0.0
    if args.evolve_models:
        model_pool = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]
        p_mutate_model, cost_weight = 0.5, 0.05
        for m in model_pool:
            clients.setdefault(m, make(m))

    router = ModelRouter(default_client, clients)
    critic = default_client   # reuse the cached client for reflection calls

    dataset = build_public_safety_dataset(limit=args.limit)
    train, evalset = train_eval_split(dataset, eval_fraction=args.eval_fraction,
                                      seed=args.seed)

    cfg = Config(
        population_size=args.pop, generations=args.generations, seed=args.seed,
        eval_workers=args.workers,
        p_mutate_prompt=0.9,
        reflective_prompt_rewrite=reflective, reflection_batch=5,
        p_mutate_model=p_mutate_model, model_pool=model_pool,
        default_model=args.model, model_cost_weight=cost_weight,
    )
    evaluator = FitnessEvaluator(
        evalset, router, train_set=train,
        default_model=args.model, model_cost_weight=cost_weight,
        complexity_penalty=args.complexity_penalty, safety_weight=args.safety_weight,
    )

    print(f"Live run :: model={args.model}  pop={args.pop}  gens={args.generations}  "
          f"workers={args.workers}  reflective={reflective}  "
          f"evolve_models={args.evolve_models}")
    print(f"dataset={len(dataset)} (train={len(train)}, eval={len(evalset)})  "
          f"safety_weight={args.safety_weight}  complexity_penalty={args.complexity_penalty}")

    engine = NEATJudge(cfg, evaluator, critic)
    best = engine.run()

    print(f"\n{default_client.stats()}")
    print(f"Best on EVAL split: fitness={best.fitness:.2f}  "
          f"safety_acc={best.safety_accuracy:.2f}  quality_acc={best.quality_accuracy:.2f}")

    print("\nEvolved agent instructions (best genome):")
    for nid in sorted(best.nodes):
        node = best.nodes[nid]
        if node.node_type.value == "input":
            continue
        print(f"  [{node.personality_core}] {node.system_instruction[:200]}")


if __name__ == "__main__":
    main()
