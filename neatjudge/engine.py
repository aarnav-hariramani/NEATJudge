"""NEATJudge engine -- the evolutionary loop that grows judge topologies."""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .config import Config
from .fitness import FitnessEvaluator
from .genome import Genome
from .innovation import InnovationTracker
from .llm import LLMClient
from .speciation import Species


class NEATJudge:
    """Top-level engine: initialize -> evaluate -> speciate -> reproduce -> repeat."""

    def __init__(self, config: Config, evaluator: FitnessEvaluator, critic: LLMClient):
        self.config = config
        self.evaluator = evaluator
        self.critic = critic
        self.rng = random.Random(config.seed)
        self.tracker = InnovationTracker()
        self.species: List[Species] = []
        self._genome_counter = 0
        self.population: List[Genome] = self._initial_population()
        self.best_ever: Optional[Genome] = None

    # ---- population bootstrap --------------------------------------------------------

    def _new_genome_id(self) -> int:
        self._genome_counter += 1
        return self._genome_counter

    def _initial_population(self) -> List[Genome]:
        return [Genome.base_genome(self.tracker, self._new_genome_id())
                for _ in range(self.config.population_size)]

    # ---- speciation ------------------------------------------------------------------

    def _speciate(self) -> None:
        for sp in self.species:
            sp.reset()
        cfg = self.config
        for genome in self.population:
            placed = False
            for sp in self.species:
                dist = genome.compatibility_distance(
                    sp.representative, cfg.c1_excess, cfg.c2_disjoint, cfg.c3_weight)
                if dist < cfg.compatibility_threshold:
                    sp.add(genome)
                    placed = True
                    break
            if not placed:
                self.species.append(Species(genome.clone()))
                self.species[-1].add(genome)
        self.species = [sp for sp in self.species if sp.members]
        for sp in self.species:
            sp.representative = self.rng.choice(sp.members).clone()
            sp.share_fitness()
            sp.update_staleness()

    # ---- reproduction ----------------------------------------------------------------

    def _mutate(self, genome: Genome) -> None:
        cfg = self.config
        if self.rng.random() < cfg.p_add_node:
            genome.mutate_add_node(self.rng)
        if self.rng.random() < cfg.p_add_edge:
            genome.mutate_add_edge(self.rng)
        if self.rng.random() < cfg.p_mutate_prompt:
            genome.mutate_prompt(self.rng, self.critic, self.evaluator,
                                 reflective=cfg.reflective_prompt_rewrite,
                                 batch_size=cfg.reflection_batch,
                                 rubric=getattr(self.evaluator, "rubric", None))
        if self.rng.random() < cfg.p_mutate_model:
            genome.mutate_model(self.rng, cfg.model_pool)

    def _breed(self, species: Species) -> Genome:
        cfg = self.config
        ranked = sorted(species.members, key=lambda g: g.fitness, reverse=True)
        cutoff = max(1, int(math.ceil(len(ranked) * cfg.survival_threshold)))
        pool = ranked[:cutoff]

        if len(pool) >= 2 and self.rng.random() > 0.25:
            parent_a = self.rng.choice(pool)
            parent_b = self.rng.choice(pool)
            if self.rng.random() < cfg.interspecies_mate_rate and len(self.species) > 1:
                other = self.rng.choice(self.species)
                if other.members:
                    parent_b = self.rng.choice(other.members)
            child = Genome.crossover(
                parent_a, parent_b, self.tracker, self._new_genome_id(), self.rng)
        else:
            child = self.rng.choice(pool).clone()
            child.genome_id = self._new_genome_id()

        self._mutate(child)
        return child

    def _reproduce(self) -> List[Genome]:
        cfg = self.config
        alive = [sp for sp in self.species if sp.staleness < cfg.stale_species_limit]
        self.species = alive if alive else self.species

        total_adj = sum(sp.total_adjusted_fitness() for sp in self.species)
        next_pop: List[Genome] = []

        for sp in self.species:
            if len(sp.members) >= cfg.elitism_min_size:
                champ = sp.champion().clone()
                champ.genome_id = self._new_genome_id()
                next_pop.append(champ)

        slots = cfg.population_size - len(next_pop)
        for sp in self.species:
            if slots <= 0:
                break
            share = (sp.total_adjusted_fitness() / total_adj) if total_adj > 0 \
                else 1.0 / len(self.species)
            for _ in range(int(round(share * slots))):
                if len(next_pop) >= cfg.population_size:
                    break
                next_pop.append(self._breed(sp))

        while len(next_pop) < cfg.population_size:
            next_pop.append(self._breed(self.rng.choice(self.species)))
        return next_pop[: cfg.population_size]

    # ---- main loop -------------------------------------------------------------------

    def _evaluate_population(self) -> None:
        workers = max(1, self.config.eval_workers)
        if workers == 1:
            for genome in self.population:
                self.evaluator.evaluate(genome)
        else:
            # Fitness evaluation is pure LLM I/O and consumes no shared RNG, so it
            # parallelizes safely; per-genome fitness is deterministic regardless.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self.evaluator.evaluate, self.population))

        for genome in self.population:
            if self.best_ever is None or genome.fitness > self.best_ever.fitness:
                self.best_ever = genome.clone()
                self.best_ever.fitness = genome.fitness

    def run(self) -> Genome:
        print("=" * 78)
        print("NEATJudge :: evolving multi-agent judge topologies")
        print("=" * 78)

        for generation in range(self.config.generations):
            self._evaluate_population()
            self._speciate()

            fits = [g.fitness for g in self.population]
            best = max(self.population, key=lambda g: g.fitness)
            print(f"\n--- Generation {generation} ---")
            print(f"species={len(self.species)}  "
                  f"best={max(fits):.2f}  mean={sum(fits) / len(fits):.2f}")
            for sp in self.species:
                print(f"  species {sp.species_id:>2}: size={len(sp.members):>2} "
                      f"| champ {sp.champion().describe()}")
            print(f"  >> gen-best: {best.describe()}")

            if generation < self.config.generations - 1:
                self.population = self._reproduce()

        print("\n" + "=" * 78)
        print(f"BEST EVOLVED JUDGE: {self.best_ever.describe()}")
        print("=" * 78)
        self.print_topology(self.best_ever)
        return self.best_ever

    @staticmethod
    def print_topology(genome: Genome) -> None:
        print("Topology (enabled context pathways):")
        for conn in sorted(genome.enabled_connections(), key=lambda c: c.innovation):
            src = genome.nodes[conn.in_node]
            dst = genome.nodes[conn.out_node]
            print(f"  [innov {conn.innovation:>2}] {src.personality_core:>16} "
                  f"--(prio {conn.priority:.2f})--> {dst.personality_core:<16}")
        print("Agent roster:")
        for nid in sorted(genome.nodes):
            node = genome.nodes[nid]
            model = node.model or "(default)"
            print(f"  node#{nid} [{node.node_type.value:>6}] {node.personality_core} "
                  f"(calibration={node.calibration:.2f}, model={model})")
