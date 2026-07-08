"""NEATJudge -- evolving multi-agent LLM-as-a-Judge communication topologies.

A conceptual translation of NEAT (NeuroEvolution of Augmenting Topologies) to
LLM-as-a-Judge architectures: nodes are specialized judge agents, directed edges
are context pathways, innovation numbers track structural history for safe
crossover, and speciation protects young topologies while their prompts optimize.
"""

from __future__ import annotations

from .archetypes import ARCHETYPE_LIBRARY, SPECIALIST_POOL, Archetype
from .config import Config
from .datasets import (
    build_golden_dataset,
    build_public_safety_dataset,
    build_simulated_dataset,
    train_eval_split,
    try_load_beavertails,
)
from .engine import NEATJudge
from .fitness import FitnessEvaluator
from .genes import ConnectionGene, NodeGene, NodeType, default_weight_text
from .genome import Genome
from .innovation import INPUT_NODE_ID, OUTPUT_NODE_ID, InnovationTracker
from .llm import (
    DEFAULT_MODEL_COST,
    MODEL_COST,
    OUTPUT_CONTRACT,
    AnthropicClient,
    LLMClient,
    MockLLMClient,
    ModelRouter,
    OpenAIClient,
)
from .speciation import Species

__version__ = "1.0.0"

__all__ = [
    "Archetype", "ARCHETYPE_LIBRARY", "SPECIALIST_POOL",
    "Config",
    "build_golden_dataset", "build_simulated_dataset",
    "build_public_safety_dataset", "train_eval_split", "try_load_beavertails",
    "NEATJudge",
    "FitnessEvaluator",
    "ConnectionGene", "NodeGene", "NodeType", "default_weight_text",
    "Genome",
    "InnovationTracker", "INPUT_NODE_ID", "OUTPUT_NODE_ID",
    "LLMClient", "MockLLMClient", "AnthropicClient", "OpenAIClient", "OUTPUT_CONTRACT",
    "ModelRouter", "MODEL_COST", "DEFAULT_MODEL_COST",
    "Species",
]
