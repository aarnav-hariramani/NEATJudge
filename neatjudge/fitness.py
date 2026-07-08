"""Fitness evaluation -- scoring an evolved judge topology against a golden set."""

from __future__ import annotations

import random
from typing import List

from .genes import NodeType
from .genome import Genome
from .llm import DEFAULT_MODEL_COST, MODEL_COST


class FitnessEvaluator:
    """Scores a genome by running it over a tiny golden validation set.

    Each item carries ground-truth ``safety`` and ``quality`` targets. The graph
    produces a verdict per item; fitness rewards safety accuracy and quality
    closeness, with a small parsimony penalty per hidden agent to discourage bloat
    (mirroring NEAT's bias toward minimal structure).

    When ``model_cost_weight`` > 0, a per-agent model-cost term is also subtracted
    (each agent priced by :data:`~neatjudge.llm.MODEL_COST` for its model gene),
    so evolution is nudged toward the cheapest models that still hold accuracy --
    the selection pressure that makes the model-mutating gene meaningful.
    """

    COMPLEXITY_PENALTY = 0.4   # fitness points shaved per hidden agent

    def __init__(self, dataset: List[dict], llm, *, train_set: List[dict] = None,
                 default_model: str = "", model_cost_weight: float = 0.0):
        self.dataset = dataset
        # Reflection samples from train_set (disjoint from the scored `dataset` when
        # provided) so prompts are not tuned on the exact items they are graded on.
        self.train_set = train_set if train_set is not None else dataset
        self.llm = llm
        self.default_model = default_model
        self.model_cost_weight = model_cost_weight

    def _model_cost(self, genome: Genome) -> float:
        """Summed per-call cost of the agents that actually run (non-input nodes)."""
        total = 0.0
        for node in genome.nodes.values():
            if node.node_type == NodeType.INPUT:
                continue
            model = node.model or self.default_model
            total += MODEL_COST.get(model, DEFAULT_MODEL_COST)
        return total

    def sample_batch(self, rng: random.Random, k: int) -> List[dict]:
        return rng.sample(self.dataset, k) if k < len(self.dataset) else list(self.dataset)

    def sample_train(self, rng: random.Random, k: int) -> List[dict]:
        pool = self.train_set
        return rng.sample(pool, k) if k < len(pool) else list(pool)

    def evaluate(self, genome: Genome) -> float:
        safety_hits = 0.0
        quality_score = 0.0
        for item in self.dataset:
            verdict = genome.evaluate_item(item, self.llm)
            truth = item["truth"]
            if verdict["safety"] == truth["safety"]:
                safety_hits += 1.0
            try:
                pred_quality = int(verdict["quality"])
            except (TypeError, ValueError):
                pred_quality = 3   # neutral fallback for a malformed judge output
            err = abs(pred_quality - int(truth["quality"]))
            quality_score += max(0.0, 1.0 - err / 4.0)

        n = len(self.dataset)
        safety_acc = safety_hits / n
        quality_acc = quality_score / n
        raw = 100.0 * (0.5 * safety_acc + 0.5 * quality_acc)

        hidden = sum(1 for node in genome.nodes.values()
                     if node.node_type == NodeType.HIDDEN)
        cost_penalty = self.model_cost_weight * self._model_cost(genome)
        fitness = max(0.0, raw - self.COMPLEXITY_PENALTY * hidden - cost_penalty)
        genome.fitness = fitness
        return fitness
