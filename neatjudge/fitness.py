"""Fitness evaluation -- scoring an evolved judge topology against a golden set."""

from __future__ import annotations

import random
from typing import List

from .genes import NodeType
from .genome import Genome
from .llm import LLMClient


class FitnessEvaluator:
    """Scores a genome by running it over a tiny golden validation set.

    Each item carries ground-truth ``safety`` and ``quality`` targets. The graph
    produces a verdict per item; fitness rewards safety accuracy and quality
    closeness, with a small parsimony penalty per hidden agent to discourage bloat
    (mirroring NEAT's bias toward minimal structure).
    """

    COMPLEXITY_PENALTY = 0.4   # fitness points shaved per hidden agent

    def __init__(self, dataset: List[dict], llm: LLMClient):
        self.dataset = dataset
        self.llm = llm

    def sample_batch(self, rng: random.Random, k: int) -> List[dict]:
        return rng.sample(self.dataset, k) if k < len(self.dataset) else list(self.dataset)

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
        fitness = max(0.0, raw - self.COMPLEXITY_PENALTY * hidden)
        genome.fitness = fitness
        return fitness
