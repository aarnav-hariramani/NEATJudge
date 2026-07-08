"""LLM-as-a-judge optimizers compared in the benchmark.

Each method optimizes a judge on the TRAIN split and is scored on the held-out
EVAL split by the shared harness. All are faithful reimplementations within this
unified harness (rather than imported wholesale) so they share the exact same
model, dataset, split, scoring, and call-accounting -- the only way the numbers
are comparable. Citations point to the original methods.

Methods
-------
* single_judge      -- LLM-as-a-judge, no optimization.
                       Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench
                       and Chatbot Arena".
* panel_of_judges   -- fixed panel of specialist judges aggregated (no evolution).
                       Verga et al. 2024, "Replacing Judges with Juries:
                       Evaluating LLM Generations with a Panel of Diverse Models".
* evoprompt_ga      -- genetic algorithm over the judge PROMPT TEXT (LLM crossover
                       + mutation), no topology. Guo et al. 2024 (ICLR),
                       "Connecting Large Language Models with Evolutionary
                       Algorithms Yields Powerful Prompt Optimizers" (EvoPrompt).
* opro              -- LLM-as-optimizer proposing prompts from a scored trajectory,
                       no topology. Yang et al. 2023, "Large Language Models as
                       Optimizers" (OPRO).
* gepa_prompt       -- reflective prompt evolution on a single judge, no topology.
                       Agrawal et al. 2025, "GEPA: Reflective Prompt Evolution Can
                       Outperform Reinforcement Learning".
* neatjudge         -- THIS work: co-evolves topology (agent nodes + context
                       edges), prompts (reflective), and speciation.
"""

from __future__ import annotations

import io
import random
from contextlib import redirect_stdout
from typing import Callable, Dict, List, Tuple

from ..archetypes import ARCHETYPE_LIBRARY
from ..config import Config
from ..engine import NEATJudge
from ..fitness import FitnessEvaluator
from ..genome import Genome
from ..innovation import InnovationTracker
from ..llm import LLMClient
from .harness import panel_genome, single_node_genome

_CORE = "Core Judge"
_BASE_INSTRUCTION = ARCHETYPE_LIBRARY[_CORE].base_instruction
_SAFETY_WEIGHT = 0.75
_WORKERS = 8   # item-level parallelism for train/eval scoring (IO-bound LLM calls)


# ---- shared helpers -----------------------------------------------------------------


def _sanitize(text: str, core: str = _CORE) -> str:
    """Clean an LLM-proposed instruction and guarantee the role keyword is present."""
    text = (text or "").strip().strip('"').strip()
    text = text.replace("```", "").strip()
    if not text:
        return f"You are a {core}. {_BASE_INSTRUCTION}"
    text = text[:800]
    if core.lower() not in text.lower():
        text = f"You are a {core}. {text}"
    return text


class _PromptScorer:
    """Scores a candidate instruction on the train split (fitness, no penalty).

    Memoizes by instruction text so a method never re-issues identical train
    evaluations; the shared cache also dedupes across methods.
    """

    def __init__(self, train: List[dict], client: LLMClient):
        self.train = train
        self.client = client
        self._memo: Dict[str, float] = {}

    def __call__(self, instruction: str) -> float:
        if instruction in self._memo:
            return self._memo[instruction]
        g = single_node_genome(InnovationTracker(), instruction)
        ev = FitnessEvaluator(self.train, self.client, complexity_penalty=0.0,
                              safety_weight=_SAFETY_WEIGHT, workers=_WORKERS)
        fit = ev.evaluate(g)
        self._memo[instruction] = fit
        return fit


def _propose(client: LLMClient, system: str, user: str) -> str:
    return _sanitize(client.complete(system, user))


# ---- baselines (no optimization) ----------------------------------------------------


def run_single_judge(train, client, rng, budget) -> Tuple[Genome, str]:
    return single_node_genome(InnovationTracker(), _BASE_INSTRUCTION), "no optimization"


def run_panel_of_judges(train, client, rng, budget) -> Tuple[Genome, str]:
    specialists = ["Safety Arbitrator", "Fact-Checker", "Tone Judge"]
    g = panel_genome(InnovationTracker(), specialists)
    return g, f"fixed panel: {', '.join(specialists)} -> Core Judge"


# ---- EvoPrompt (GA over prompt text) ------------------------------------------------


def run_evoprompt_ga(train, client, rng, budget, pop=6, gens=3) -> Tuple[Genome, str]:
    score = _PromptScorer(train, client)
    sys_x = "You are optimizing an LLM judge's system instruction via an evolutionary algorithm."

    def mutate(instr: str) -> str:
        return _propose(client, sys_x,
                        "Mutate the following judge instruction to improve its accuracy, "
                        "keeping its role and making the decision rules crisper. Output ONLY "
                        f"the new instruction:\n{instr}")

    def crossover(a: str, b: str) -> str:
        return _propose(client, sys_x,
                        "Combine the best parts of these two judge instructions into a single "
                        "improved instruction (keep the role, be concise). Output ONLY the new "
                        f"instruction:\nPARENT 1:\n{a}\n\nPARENT 2:\n{b}")

    # Seed population: base + LLM mutations.
    population = [_BASE_INSTRUCTION]
    while len(population) < pop and client.calls < budget:
        population.append(mutate(_BASE_INSTRUCTION))
    scored = [(instr, score(instr)) for instr in population]

    for _ in range(gens):
        if client.calls >= budget:
            break
        scored.sort(key=lambda t: t[1], reverse=True)
        parents = [instr for instr, _ in scored[:max(2, pop // 2)]]
        children: List[str] = []
        while len(children) < pop and client.calls < budget:
            a, b = rng.choice(parents), rng.choice(parents)
            child = crossover(a, b)
            if rng.random() < 0.5 and client.calls < budget:
                child = mutate(child)
            children.append(child)
        scored += [(c, score(c)) for c in children]
        scored.sort(key=lambda t: t[1], reverse=True)
        scored = scored[:pop]     # elitist truncation

    best = max(scored, key=lambda t: t[1])[0]
    return single_node_genome(InnovationTracker(), best), f"GA pop={pop} gens={gens}"


# ---- OPRO (LLM-as-optimizer) --------------------------------------------------------


def run_opro(train, client, rng, budget, steps=12, topk=6) -> Tuple[Genome, str]:
    score = _PromptScorer(train, client)
    sys_o = "You are optimizing an LLM judge's system instruction."
    trajectory: List[Tuple[str, float]] = [(_BASE_INSTRUCTION, score(_BASE_INSTRUCTION))]

    used = 0
    while used < steps and client.calls < budget:
        used += 1
        top = sorted(trajectory, key=lambda t: t[1])[-topk:]      # ascending by score
        listing = "\n".join(f"[score {s:.1f}] {instr}" for instr, s in top)
        proposal = _propose(client, sys_o,
                            "Below are judge instructions and their scores (0-100, higher is "
                            "better). Write a NEW instruction, different from all above, likely "
                            "to score higher. Keep the judge role and be concise. Output ONLY the "
                            f"instruction:\n\n{listing}")
        trajectory.append((proposal, score(proposal)))

    best = max(trajectory, key=lambda t: t[1])[0]
    return single_node_genome(InnovationTracker(), best), f"OPRO steps={used}"


# ---- GEPA (reflective prompt evolution, no topology) --------------------------------


def run_gepa_prompt(train, client, rng, budget, steps=8) -> Tuple[Genome, str]:
    ev = FitnessEvaluator(train, client, train_set=train, complexity_penalty=0.0,
                          safety_weight=_SAFETY_WEIGHT, workers=_WORKERS)
    best = single_node_genome(InnovationTracker(), _BASE_INSTRUCTION)
    best_fit = ev.evaluate(best)

    used = 0
    while used < steps and client.calls < budget:
        used += 1
        cand = best.clone()
        changed = cand.mutate_prompt(rng, client, ev, reflective=True, batch_size=5)
        if not changed:
            continue
        fit = ev.evaluate(cand)
        if fit > best_fit:
            best, best_fit = cand.clone(), fit

    return best, f"reflective steps={used}"


# ---- NEATJudge (ours: topology + prompts + speciation) ------------------------------


def run_neatjudge(train, client, rng, budget, pop=6, gens=3) -> Tuple[Genome, str]:
    cfg = Config(
        population_size=pop, generations=gens, seed=7, eval_workers=_WORKERS,
        p_mutate_prompt=0.9, reflective_prompt_rewrite=True, reflection_batch=5,
        compatibility_threshold=0.6,
    )
    # eval_workers on the engine parallelizes across genomes; keep the evaluator's
    # own item loop sequential to avoid nesting thread pools.
    ev = FitnessEvaluator(train, client, train_set=train, complexity_penalty=0.1,
                          safety_weight=_SAFETY_WEIGHT, workers=1)
    with redirect_stdout(io.StringIO()):
        best = NEATJudge(cfg, ev, client).run()
    hidden = sum(1 for n in best.nodes.values() if n.node_type.value == "hidden")
    return best, f"pop={pop} gens={gens} specialists={hidden}"


# ---- registry -----------------------------------------------------------------------
# (key, label, citation, evolves_topology, fn)
METHODS: List[Tuple[str, str, str, bool, Callable]] = [
    ("single_judge", "Single LLM judge", "Zheng et al. 2023 (LLM-as-a-judge)", False, run_single_judge),
    ("panel_of_judges", "Panel of judges", "Verga et al. 2024 (LLM-as-a-jury)", False, run_panel_of_judges),
    ("evoprompt_ga", "EvoPrompt (GA)", "Guo et al. 2024 (EvoPrompt)", False, run_evoprompt_ga),
    ("opro", "OPRO", "Yang et al. 2023 (OPRO)", False, run_opro),
    ("gepa_prompt", "GEPA (reflective)", "Agrawal et al. 2025 (GEPA)", False, run_gepa_prompt),
    ("neatjudge", "NEATJudge (ours)", "this work", True, run_neatjudge),
]
