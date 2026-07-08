"""Fitness evaluation -- scoring an evolved judge topology against a golden set."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
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

    COMPLEXITY_PENALTY = 0.4   # default fitness points shaved per hidden agent

    def __init__(self, dataset: List[dict], llm, *, train_set: List[dict] = None,
                 default_model: str = "", model_cost_weight: float = 0.0,
                 complexity_penalty: float = None, safety_weight: float = 0.5,
                 workers: int = 1, rubric=None):
        self.dataset = dataset
        # When set, judging uses the rubric's numeric axes (mean per-axis closeness)
        # instead of the default safety/quality scheme.
        self.rubric = rubric
        # Item-level parallelism for one genome's evaluation (LLM calls are IO-bound
        # and genome.evaluate_item mutates no shared state). 1 = sequential.
        self.workers = workers
        # Reflection samples from train_set (disjoint from the scored `dataset` when
        # provided) so prompts are not tuned on the exact items they are graded on.
        self.train_set = train_set if train_set is not None else dataset
        self.llm = llm
        self.default_model = default_model
        self.model_cost_weight = model_cost_weight
        # Per-hidden-agent parsimony penalty. Lower it when specialists must be
        # allowed to earn their keep (e.g. a strong generalist baseline). Defaults
        # to the historical 0.4 for backward compatibility.
        self.complexity_penalty = (self.COMPLEXITY_PENALTY if complexity_penalty is None
                                   else complexity_penalty)
        # Weight on the safety axis (quality gets 1 - safety_weight).
        self.safety_weight = safety_weight

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

    def _item_scores(self, genome: Genome, item: dict) -> tuple:
        """Return (safety_hit, quality_closeness) for one item -- pure, thread-safe."""
        verdict = genome.evaluate_item(item, self.llm)
        truth = item["truth"]
        safety_hit = 1.0 if verdict["safety"] == truth["safety"] else 0.0
        try:
            pred_quality = int(verdict["quality"])
        except (TypeError, ValueError):
            pred_quality = 3   # neutral fallback for a malformed judge output
        err = abs(pred_quality - int(truth["quality"]))
        return safety_hit, max(0.0, 1.0 - err / 4.0)

    def _axis_closeness(self, genome: Genome, item: dict) -> dict:
        """Per-axis closeness in [0,1] for one item under the rubric."""
        verdict = genome.evaluate_item(item, self.llm, self.rubric)
        truth = item["truth"]
        out = {}
        for ax in self.rubric.axes:
            try:
                pred = int(verdict.get(ax.name))
            except (TypeError, ValueError):
                pred = self.rubric.midpoint(ax.name)
            err = abs(pred - int(truth[ax.name]))
            out[ax.name] = max(0.0, 1.0 - err / self.rubric.span(ax.name))
        return out

    def _penalized(self, genome: Genome, raw: float) -> float:
        hidden = sum(1 for node in genome.nodes.values()
                     if node.node_type == NodeType.HIDDEN)
        cost_penalty = self.model_cost_weight * self._model_cost(genome)
        return max(0.0, raw - self.complexity_penalty * hidden - cost_penalty)

    def evaluate(self, genome: Genome) -> float:
        if self.rubric is not None:
            return self._evaluate_rubric(genome)
        if self.workers > 1 and len(self.dataset) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                pairs = list(pool.map(lambda it: self._item_scores(genome, it),
                                      self.dataset))
        else:
            pairs = [self._item_scores(genome, it) for it in self.dataset]
        n = len(self.dataset)
        safety_acc = sum(p[0] for p in pairs) / n
        quality_acc = sum(p[1] for p in pairs) / n
        sw = self.safety_weight
        raw = 100.0 * (sw * safety_acc + (1.0 - sw) * quality_acc)
        fitness = self._penalized(genome, raw)
        genome.fitness = fitness
        genome.safety_accuracy = safety_acc
        genome.quality_accuracy = quality_acc
        return fitness

    def _evaluate_rubric(self, genome: Genome) -> float:
        if self.workers > 1 and len(self.dataset) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                rows = list(pool.map(lambda it: self._axis_closeness(genome, it),
                                     self.dataset))
        else:
            rows = [self._axis_closeness(genome, it) for it in self.dataset]
        n = len(self.dataset)
        axis_acc = {ax.name: sum(r[ax.name] for r in rows) / n for ax in self.rubric.axes}
        raw = 100.0 * (sum(axis_acc.values()) / len(axis_acc))
        fitness = self._penalized(genome, raw)
        genome.fitness = fitness
        genome.axis_accuracy = axis_acc
        return fitness
