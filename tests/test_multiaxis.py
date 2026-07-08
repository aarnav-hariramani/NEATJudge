"""Multi-axis (rubric) judging tests, using deterministic fake clients."""

from __future__ import annotations

import random

from neatjudge import FitnessEvaluator, Genome, HELPSTEER_RUBRIC, InnovationTracker, LLMClient
from neatjudge.benchmark import METHODS
from neatjudge.benchmark.harness import (
    CountingLLMClient,
    n_judge_agents,
    panel_genome,
    score_genome,
    single_node_genome,
)
from neatjudge.rubric import Axis, Rubric


class _FlatJudge(LLMClient):
    """Rates every axis a fixed constant (parsed generically from the contract)."""

    def __init__(self, value=3):
        self.value = value

    def complete(self, system, user):
        import re
        axes = re.findall(r'"(\w+)": \{"value"', system)
        cells = ", ".join(f'"{a}": {{"value": {self.value}, "confidence": 0.8}}' for a in axes)
        return '{"per_axis": {' + cells + "}}"


def _data():
    return [{"id": f"x{i}", "prompt": "p", "response": "r",
             "truth": {"correctness": c, "coherence": 4, "helpfulness": 2}}
            for i, c in enumerate([0, 2, 4, 4, 3, 1])]


def test_rubric_contract_lists_all_axes():
    c = HELPSTEER_RUBRIC.contract()
    for name in ("correctness", "coherence", "helpfulness"):
        assert name in c


def test_rubric_scoring_is_mean_axis_closeness():
    g = single_node_genome(InnovationTracker(), "You are a Core Judge.")
    fit, mean_acc, _ = score_genome(g, _data(), _FlatJudge(3), rubric=HELPSTEER_RUBRIC)
    assert 0.0 <= fit <= 100.0
    assert set(g.axis_accuracy) == {"correctness", "coherence", "helpfulness"}
    # coherence truth is always 4; a judge outputting 3 is off by 1 -> closeness 0.75.
    assert abs(g.axis_accuracy["coherence"] - 0.75) < 1e-6


def test_ownership_maps_axes_to_specialists():
    assert HELPSTEER_RUBRIC.owner_of("Fact-Checker") == ["correctness"]
    assert HELPSTEER_RUBRIC.owner_of("Coherence Judge") == ["coherence"]
    assert HELPSTEER_RUBRIC.owner_of("Relevance Judge") == ["helpfulness"]
    assert HELPSTEER_RUBRIC.owner_of("Core Judge") == ["correctness", "coherence", "helpfulness"]


def test_panel_uses_axis_owner_specialists_under_rubric():
    # The benchmark panel for a rubric should be exactly the axis owners.
    from neatjudge.benchmark.baselines import run_panel_of_judges
    g, notes = run_panel_of_judges(_data(), _FlatJudge(), random.Random(0), 100,
                                   rubric=HELPSTEER_RUBRIC)
    cores = {n.personality_core for n in g.nodes.values()}
    assert {"Fact-Checker", "Coherence Judge", "Relevance Judge"} <= cores
    assert n_judge_agents(g) == 4   # 3 specialists + Core Judge aggregator


def test_all_methods_run_under_rubric():
    train = _data()
    evalset = _data()
    shared = _FlatJudge(3)
    for key, label, citation, topo, fn in METHODS:
        kwargs = {}
        if key == "evoprompt_ga":
            kwargs = dict(pop=2, gens=1)
        elif key == "opro":
            kwargs = dict(steps=1)
        elif key == "gepa_prompt":
            kwargs = dict(steps=1)
        elif key == "neatjudge":
            kwargs = dict(pop=3, gens=1)
        genome, notes = fn(train, CountingLLMClient(shared), random.Random(0),
                           budget=80, rubric=HELPSTEER_RUBRIC, **kwargs)
        fit, mean_acc, _ = score_genome(genome, evalset, shared, rubric=HELPSTEER_RUBRIC)
        assert 0.0 <= fit <= 100.0
        assert n_judge_agents(genome) >= 1


def test_specialist_focus_beats_generalist_when_axes_conflict():
    # A synthetic rubric where a generalist (flat guesser) is mediocre on both axes,
    # but axis-owning specialists that each nail their axis (via routing) do better.
    rub = Rubric(axes=(
        Axis("correctness", 0, 4, "x", "Fact-Checker"),
        Axis("coherence", 0, 4, "y", "Coherence Judge"),
    ))
    data = [{"id": f"d{i}", "prompt": "p", "response": "r",
             "truth": {"correctness": 4, "coherence": 0}} for i in range(4)]

    class _RoutingJudge(LLMClient):
        # A Fact-Checker nails correctness(=4); a Coherence Judge nails coherence(=0);
        # anyone else (generalist) guesses the midpoint 2 on both.
        def complete(self, system, user):
            if "Fact-Checker" in system:
                cor, coh = 4, 2
            elif "Coherence Judge" in system:
                cor, coh = 2, 0
            else:
                cor, coh = 2, 2
            # If it received upstream specialist findings, adopt them (aggregator).
            import re
            if "UPSTREAM_JSON" in user:
                if '"correctness": {"value": 4' in user:
                    cor = 4
                if '"coherence": {"value": 0' in user:
                    coh = 0
            return ('{"per_axis": {"correctness": {"value": %d, "confidence": 0.9}, '
                    '"coherence": {"value": %d, "confidence": 0.9}}}' % (cor, coh))

    j = _RoutingJudge()
    single = single_node_genome(InnovationTracker(), "You are a Core Judge.")
    panel = panel_genome(InnovationTracker(), ["Fact-Checker", "Coherence Judge"])
    s_fit, *_ = score_genome(single, data, j, rubric=rub)
    p_fit, *_ = score_genome(panel, data, j, rubric=rub)
    assert p_fit > s_fit   # specialists that own an axis beat the generalist
