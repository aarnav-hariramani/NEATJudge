"""Configuration -- all evolutionary hyperparameters in one place."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    population_size: int = 12
    generations: int = 3
    seed: int = 7

    # Parallelism for fitness evaluation (LLM calls are IO-bound). 1 = sequential
    # and fully reproducible; >1 speeds up live runs against real APIs.
    eval_workers: int = 1

    # Compatibility distance coefficients + speciation threshold.
    c1_excess: float = 1.0
    c2_disjoint: float = 1.0
    c3_weight: float = 0.4
    compatibility_threshold: float = 0.6

    # Mutation probabilities.
    p_add_node: float = 0.25
    p_add_edge: float = 0.35
    p_mutate_prompt: float = 0.80

    # Reproduction.
    survival_threshold: float = 0.5       # top fraction of a species that may breed
    elitism_min_size: int = 3             # species this big or larger copy their champ
    interspecies_mate_rate: float = 0.05
    stale_species_limit: int = 15         # cull species stagnant this long
