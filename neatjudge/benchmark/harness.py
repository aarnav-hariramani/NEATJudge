"""Shared benchmark machinery: call counting, genome builders, uniform scoring.

Every optimizer in ``baselines.py`` ultimately produces a :class:`Genome`, and
every method is scored by the SAME function (:func:`score_genome`) on the SAME
held-out eval split with the SAME client, so results are directly comparable.
Optimization LLM calls are counted per method via :class:`CountingLLMClient`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

from ..fitness import FitnessEvaluator
from ..genes import ConnectionGene, NodeGene, NodeType, default_weight_text
from ..genome import Genome
from ..innovation import INPUT_NODE_ID, OUTPUT_NODE_ID, InnovationTracker
from ..llm import LLMClient
from ..archetypes import ARCHETYPE_LIBRARY


class CountingLLMClient(LLMClient):
    """Wrapper that counts every completion request routed through it.

    Placed *above* a shared cache so it measures the logical work a method issues,
    while the shared cache still deduplicates real API cost across methods.
    """

    def __init__(self, wrapped: LLMClient):
        self.wrapped = wrapped
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system_instruction: str, user_content: str) -> str:
        with self._lock:
            self.calls += 1
        return self.wrapped.complete(system_instruction, user_content)


@dataclass
class BenchmarkResult:
    method: str
    citation: str
    evolves_topology: bool
    eval_fitness: float
    safety_acc: float
    quality_acc: float
    n_agents: int          # judge agents in the final graph (excludes the ingestor)
    requests: int          # logical LLM completion requests during optimization
    seconds: float
    notes: str = ""


# ---- genome builders ----------------------------------------------------------------


def single_node_genome(tracker: InnovationTracker, instruction: str,
                       core: str = "Core Judge", model: Optional[str] = None) -> Genome:
    """A one-judge graph INPUT -> Core Judge(OUTPUT) with a given instruction.

    This is the substrate the prompt-only optimizers evolve: they change only the
    OUTPUT node's ``system_instruction``; topology is fixed.
    """
    g = Genome(tracker, 0)
    g.nodes[INPUT_NODE_ID] = NodeGene(
        INPUT_NODE_ID, NodeType.INPUT, "Ingestor",
        "You are the intake node. Pass the item through unchanged.", 0.0)
    g.nodes[OUTPUT_NODE_ID] = NodeGene(
        OUTPUT_NODE_ID, NodeType.OUTPUT, core, instruction, 0.30, model)
    innov = tracker.get_edge_innovation(INPUT_NODE_ID, OUTPUT_NODE_ID)
    g.connections[innov] = ConnectionGene(
        innov, INPUT_NODE_ID, OUTPUT_NODE_ID, default_weight_text(1.0), 1.0, True)
    return g


def panel_genome(tracker: InnovationTracker, specialists: List[str]) -> Genome:
    """A fixed multi-agent panel: INPUT -> [specialists] -> Core Judge(OUTPUT).

    Models an LLM-as-a-jury / panel-of-evaluators design (fixed topology, no
    evolution): each specialist judges the item and feeds its verdict to the Core
    Judge, which aggregates.
    """
    g = single_node_genome(tracker, ARCHETYPE_LIBRARY["Core Judge"].base_instruction)
    next_id = 2
    for core in specialists:
        spec = ARCHETYPE_LIBRARY[core]
        nid = next_id
        next_id += 1
        g.nodes[nid] = NodeGene(nid, NodeType.HIDDEN, spec.core, spec.base_instruction, 0.30)
        i_in = tracker.get_edge_innovation(INPUT_NODE_ID, nid)
        g.connections[i_in] = ConnectionGene(
            i_in, INPUT_NODE_ID, nid, default_weight_text(1.0), 1.0, True)
        i_out = tracker.get_edge_innovation(nid, OUTPUT_NODE_ID)
        g.connections[i_out] = ConnectionGene(
            i_out, nid, OUTPUT_NODE_ID, default_weight_text(1.0), 1.0, True)
    g.enforce_acyclic()
    return g


# ---- uniform scoring ----------------------------------------------------------------


def score_genome(genome: Genome, dataset: List[dict], client: LLMClient,
                 safety_weight: float = 0.75, workers: int = 8, rubric=None) -> tuple:
    """Score a genome on a dataset with NO complexity penalty (pure accuracy).

    The parsimony penalty is excluded here so methods are compared on judgment
    quality alone; agent count is reported separately. Returns
    (fitness, primary_acc, secondary_acc): for the default scheme these are
    (safety_acc, quality_acc); for a rubric they are (mean axis closeness, 0.0)
    and the per-axis breakdown is available on ``genome.axis_accuracy``.
    """
    ev = FitnessEvaluator(dataset, client, complexity_penalty=0.0,
                          safety_weight=safety_weight, workers=workers, rubric=rubric)
    fit = ev.evaluate(genome)
    if rubric is not None:
        acc = genome.axis_accuracy
        mean_acc = (sum(acc.values()) / len(acc)) if acc else 0.0
        return fit, mean_acc, 0.0
    return fit, genome.safety_accuracy, genome.quality_accuracy


def n_judge_agents(genome: Genome) -> int:
    return sum(1 for n in genome.nodes.values() if n.node_type != NodeType.INPUT)


# ---- reporting ----------------------------------------------------------------------


def format_table(results: List[BenchmarkResult]) -> str:
    header = (f"{'method':<22} {'topo':<5} {'eval_fit':>8} {'safety':>7} "
              f"{'quality':>7} {'agents':>6} {'calls':>6} {'secs':>6}")
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda x: x.eval_fitness, reverse=True):
        lines.append(
            f"{r.method:<22} {('yes' if r.evolves_topology else 'no'):<5} "
            f"{r.eval_fitness:>8.2f} {r.safety_acc:>7.2f} {r.quality_acc:>7.2f} "
            f"{r.n_agents:>6} {r.requests:>6} {r.seconds:>6.0f}")
    return "\n".join(lines)
