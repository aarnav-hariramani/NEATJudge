"""Genes -- the nodes (agents) and edges (synapses) of a judge topology."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .llm import OUTPUT_CONTRACT


class NodeType(Enum):
    INPUT = "input"     # the item ingestor (id 0)
    OUTPUT = "output"   # the final aggregator (id 1)
    HIDDEN = "hidden"   # an inserted specialist sub-judge


@dataclass
class NodeGene:
    """A single Judge Agent in the graph.

    ``personality_core`` is immutable (the archetype name). ``system_instruction``
    is the live, mutable prompt that GEPA-style reflective mutation rewrites.
    ``calibration`` is a simulation hook (see :class:`~neatjudge.llm.MockLLMClient`)
    tracking how much reflective refinement the prompt has accumulated.

    ``model`` is the *model-mutating gene*: the id of the LLM this agent runs on
    (e.g. ``"claude-opus-4-8"``). ``None`` means "use the router's default model".
    It is heritable through crossover and altered by ``Genome.mutate_model``, so
    evolution can search over *which model runs each agent*, not just topology and
    prompts -- letting selection discover the cheapest model that holds fitness.
    """
    node_id: int
    node_type: NodeType
    personality_core: str
    system_instruction: str
    calibration: float = 0.30
    model: Optional[str] = None

    def rendered_system_instruction(self, rubric=None) -> str:
        """The system prompt actually sent to the LLM.

        Composed of the evolved instruction, the (simulation-only) calibration
        marker, and the output contract that makes real models emit parseable JSON.
        When a ``rubric`` is given, its multi-axis contract replaces the default
        safety/quality one.
        """
        contract = rubric.contract() if rubric is not None else OUTPUT_CONTRACT
        return (
            f"{self.system_instruction} [[calibration:{self.calibration:.2f}]]\n\n"
            f"{contract}"
        )

    def clone(self) -> "NodeGene":
        return NodeGene(
            self.node_id, self.node_type, self.personality_core,
            self.system_instruction, self.calibration, self.model,
        )


@dataclass
class ConnectionGene:
    """A directed context pathway (synapse) between two agents.

    The weight is expressed *textually* -- the routing / priority instruction the
    receiving agent is given about this upstream signal -- with a numeric
    ``priority`` companion used for compatibility distance and tie-breaking.
    Following NEAT, edges are *disabled* (not deleted) when split, preserving
    history for crossover.
    """
    innovation: int
    in_node: int
    out_node: int
    weight_text: str
    priority: float = 1.0
    enabled: bool = True

    def clone(self) -> "ConnectionGene":
        return ConnectionGene(
            self.innovation, self.in_node, self.out_node,
            self.weight_text, self.priority, self.enabled,
        )


def default_weight_text(priority: float) -> str:
    """Generate a textual routing instruction for a new edge from its priority."""
    if priority >= 0.75:
        band = "CRITICAL upstream signal; weigh it heavily"
    elif priority >= 0.4:
        band = "supporting upstream signal; consider it"
    else:
        band = "advisory upstream signal; use only to break ties"
    return f"Treat the following as a {band}."
