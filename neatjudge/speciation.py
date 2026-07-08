"""Speciation -- clustering topologically-similar genomes to protect innovation."""

from __future__ import annotations

from typing import List

from .genome import Genome


class Species:
    """A cluster of topologically-similar genomes sharing a fitness pool.

    Fitness sharing (dividing each member's fitness by species size) prevents any
    single structural strategy from dominating the whole population before its
    prompts have had time to optimize.
    """

    _counter = 0

    def __init__(self, representative: Genome):
        Species._counter += 1
        self.species_id = Species._counter
        self.representative = representative
        self.members: List[Genome] = []
        self.best_fitness: float = 0.0
        self.staleness: int = 0

    def reset(self) -> None:
        self.members = []

    def add(self, genome: Genome) -> None:
        self.members.append(genome)

    def share_fitness(self) -> None:
        size = max(1, len(self.members))
        for genome in self.members:
            genome.adjusted_fitness = genome.fitness / size

    def total_adjusted_fitness(self) -> float:
        return sum(g.adjusted_fitness for g in self.members)

    def champion(self) -> Genome:
        return max(self.members, key=lambda g: g.fitness)

    def update_staleness(self) -> None:
        top = self.champion().fitness
        if top > self.best_fitness + 1e-9:
            self.best_fitness = top
            self.staleness = 0
        else:
            self.staleness += 1
