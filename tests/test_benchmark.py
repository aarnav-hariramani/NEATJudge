"""Smoke tests for the benchmark harness (mock, deterministic, offline)."""

from __future__ import annotations

import random

from neatjudge import MockLLMClient, build_public_safety_dataset, train_eval_split
from neatjudge.benchmark import METHODS, CountingLLMClient
from neatjudge.benchmark.harness import (
    n_judge_agents,
    panel_genome,
    score_genome,
    single_node_genome,
)
from neatjudge.innovation import InnovationTracker


def _split(limit=16):
    ds = build_public_safety_dataset(limit=limit)
    return train_eval_split(ds, eval_fraction=0.5, seed=0)


def test_counting_client_counts():
    c = CountingLLMClient(MockLLMClient())
    c.complete("You are a Core Judge. [[calibration:0.30]]", "PROMPT: p\nRESPONSE: r")
    assert c.calls == 1


def test_panel_genome_is_valid_multi_agent_dag():
    g = panel_genome(InnovationTracker(), ["Safety Arbitrator", "Fact-Checker"])
    assert n_judge_agents(g) == 3          # 2 specialists + Core Judge aggregator
    order = {n: i for i, n in enumerate(g.topological_order())}
    for c in g.enabled_connections():
        assert order[c.in_node] < order[c.out_node]


def test_all_methods_run_and_produce_scores():
    train, evalset = _split(16)
    shared = MockLLMClient()
    for key, label, citation, topo, fn in METHODS:
        counting = CountingLLMClient(shared)
        rng = random.Random(0)
        kwargs = {}
        if key == "evoprompt_ga":
            kwargs = dict(pop=3, gens=1)
        elif key == "opro":
            kwargs = dict(steps=2)
        elif key == "gepa_prompt":
            kwargs = dict(steps=2)
        elif key == "neatjudge":
            kwargs = dict(pop=4, gens=2)
        genome, notes = fn(train, counting, rng, budget=200, **kwargs)
        fit, sacc, qacc = score_genome(genome, evalset, shared)
        assert 0.0 <= fit <= 100.0
        assert 0.0 <= sacc <= 1.0 and 0.0 <= qacc <= 1.0
        assert n_judge_agents(genome) >= 1


def test_neatjudge_topology_beats_single_in_mock():
    # In the mock landscape, safety needs a specialist, so NEAT (topology) should
    # score at least as well as an un-optimized single judge.
    train, evalset = _split(16)
    shared = MockLLMClient()

    single_fn = dict((m[0], m[4]) for m in METHODS)["single_judge"]
    neat_fn = dict((m[0], m[4]) for m in METHODS)["neatjudge"]

    sg, _ = single_fn(train, CountingLLMClient(shared), random.Random(0), 200)
    ng, _ = neat_fn(train, CountingLLMClient(shared), random.Random(0), 200, pop=6, gens=3)

    single_fit, _, _ = score_genome(sg, evalset, shared)
    neat_fit, _, _ = score_genome(ng, evalset, shared)
    assert neat_fit >= single_fit
