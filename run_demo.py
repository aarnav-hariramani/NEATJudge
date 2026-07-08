"""Offline, reproducible demo of NEATJudge using the deterministic MockLLMClient.

Runs a full 3-generation evolution with no API keys required::

    python run_demo.py
"""

from __future__ import annotations

from neatjudge import (
    Config,
    FitnessEvaluator,
    MockLLMClient,
    NEATJudge,
    build_simulated_dataset,
)


def main() -> None:
    config = Config(population_size=12, generations=3, seed=7)

    llm = MockLLMClient()      # deterministic offline judge
    critic = MockLLMClient()   # deterministic offline prompt critic

    dataset = build_simulated_dataset()   # cue-embedded items for the mock
    evaluator = FitnessEvaluator(dataset, llm)

    engine = NEATJudge(config, evaluator, critic)
    best = engine.run()

    print("\nExample verdicts from the best evolved judge:")
    for sample in dataset[:4]:
        verdict = best.evaluate_item(sample, llm)
        truth = sample["truth"]
        ok = verdict["safety"] == truth["safety"] and verdict["quality"] == truth["quality"]
        print(f"  {'OK ' if ok else '-- '}{sample['id']}: "
              f"verdict={verdict}  truth={truth}")


if __name__ == "__main__":
    main()
