"""Verification suite for NEATJudge.

Covers the NEAT correctness invariants (innovation memoization, crossover
alignment, DAG safety, compatibility distance, speciation) and the robustness of
graph evaluation against malformed judge output. Runs entirely offline.
"""

from __future__ import annotations

import io
import random
from contextlib import redirect_stdout

import pytest

from neatjudge import (
    Config,
    FitnessEvaluator,
    Genome,
    InnovationTracker,
    MockLLMClient,
    NEATJudge,
    NodeGene,
    NodeType,
    build_golden_dataset,
    build_simulated_dataset,
)
from neatjudge.genes import ConnectionGene, default_weight_text
from neatjudge.innovation import INPUT_NODE_ID, OUTPUT_NODE_ID


# ---------------------------------------------------------------------------------
# Innovation tracking
# ---------------------------------------------------------------------------------

def test_edge_innovation_is_memoized_by_structure():
    tr = InnovationTracker()
    a = tr.get_edge_innovation(0, 5)
    b = tr.get_edge_innovation(0, 5)
    c = tr.get_edge_innovation(5, 0)      # different structure -> different number
    assert a == b
    assert a != c


def test_split_node_is_stable_per_split_edge():
    tr = InnovationTracker()
    pool = ["Safety Arbitrator", "Fact-Checker"]
    n1 = tr.get_split_node(7, pool)
    n2 = tr.get_split_node(7, pool)       # same split edge -> same node id
    n3 = tr.get_split_node(8, pool)       # different split edge -> different id
    assert n1 == n2
    assert n1 != n3


def test_first_split_yields_first_specialist():
    tr = InnovationTracker()
    node_id = tr.get_split_node(0, ["Safety Arbitrator", "Fact-Checker"])
    assert tr.archetype_of(node_id) == "Safety Arbitrator"


# ---------------------------------------------------------------------------------
# Structural mutation
# ---------------------------------------------------------------------------------

def test_add_node_splits_and_disables_original_edge():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    assert len(g.enabled_connections()) == 1
    g.mutate_add_node(random.Random(0))
    # Original INPUT->OUTPUT edge disabled; two new enabled edges added.
    assert len(g.enabled_connections()) == 2
    assert any(not c.enabled for c in g.connections.values())
    assert any(n.node_type == NodeType.HIDDEN for n in g.nodes.values())


def test_add_edge_never_creates_cycle():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    rng = random.Random(1)
    for _ in range(30):
        g.mutate_add_node(rng)
        g.mutate_add_edge(rng)
    order = {n: i for i, n in enumerate(g.topological_order())}
    for c in g.enabled_connections():
        assert order[c.in_node] < order[c.out_node]


# ---------------------------------------------------------------------------------
# DAG repair after crossover
# ---------------------------------------------------------------------------------

def test_enforce_acyclic_breaks_a_hand_built_cycle():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    for nid in (2, 3):
        g.nodes[nid] = NodeGene(nid, NodeType.HIDDEN, "Tone Judge", "t", 0.3)

    def add(a, b):
        i = tr.get_edge_innovation(a, b)
        g.connections[i] = ConnectionGene(i, a, b, default_weight_text(1.0), 1.0, True)

    add(INPUT_NODE_ID, 2); add(2, 3); add(3, 2); add(3, OUTPUT_NODE_ID)
    assert g.enforce_acyclic() >= 1
    order = {n: i for i, n in enumerate(g.topological_order())}
    for c in g.enabled_connections():
        assert order[c.in_node] < order[c.out_node]


def test_crossover_of_opposing_edges_stays_acyclic():
    tr = InnovationTracker()
    A, B = Genome.base_genome(tr, 1), Genome.base_genome(tr, 2)
    for G in (A, B):
        for nid in (2, 3):
            G.nodes[nid] = NodeGene(nid, NodeType.HIDDEN, "Tone Judge", "t", 0.3)

    def link(G, a, b):
        i = tr.get_edge_innovation(a, b)
        G.connections[i] = ConnectionGene(i, a, b, default_weight_text(1.0), 1.0, True)

    link(A, 2, 3)
    link(B, 3, 2)
    A.fitness = B.fitness = 50.0          # tie => both disjoint genes inherited
    child = Genome.crossover(A, B, tr, 3, random.Random(0))
    order = {n: i for i, n in enumerate(child.topological_order())}
    for c in child.enabled_connections():
        assert order[c.in_node] < order[c.out_node]


# ---------------------------------------------------------------------------------
# Compatibility distance + crossover history
# ---------------------------------------------------------------------------------

def test_identical_genomes_have_zero_distance():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    assert g.compatibility_distance(g.clone(), 1.0, 1.0, 0.4) == 0.0


def test_structural_divergence_increases_distance():
    tr = InnovationTracker()
    base = Genome.base_genome(tr, 1)
    grown = base.clone()
    grown.mutate_add_node(random.Random(0))
    d = base.compatibility_distance(grown, 1.0, 1.0, 0.4)
    assert d > 0.0


def test_crossover_has_no_orphan_edges():
    tr = InnovationTracker()
    A = Genome.base_genome(tr, 1)
    rng = random.Random(3)
    for _ in range(5):
        A.mutate_add_node(rng)
        A.mutate_add_edge(rng)
    B = A.clone()
    B.mutate_add_node(rng)
    A.fitness, B.fitness = 10.0, 20.0
    child = Genome.crossover(A, B, tr, 99, rng)
    for conn in child.connections.values():
        assert conn.in_node in child.nodes
        assert conn.out_node in child.nodes


# ---------------------------------------------------------------------------------
# Robustness of graph evaluation to malformed judge output
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("5", {}), ("true", {}), ("null", {}), ('"safe"', {}), ("garbage", {}),
    ('{"per_axis": {}}', {"per_axis": {}}),
    ('```json\n{"per_axis": {"safety": {"value": "unsafe"}}}\n```',
     {"per_axis": {"safety": {"value": "unsafe"}}}),
])
def test_safe_parse_normalizes_non_dict_and_fenced(raw, expected):
    assert Genome._safe_parse(raw) == expected


@pytest.mark.parametrize("out,expected", [
    ({"per_axis": {"safety": "safe"}}, {"safety": "safe", "quality": 3}),      # bare label
    ({"per_axis": "safe"}, {"safety": "safe", "quality": 3}),                   # non-dict
    ({}, {"safety": "safe", "quality": 3}),                                     # empty
    ({"per_axis": {"safety": {"value": "unsafe"}, "quality": {"value": 1}}},
     {"safety": "unsafe", "quality": 1}),                                       # normal
])
def test_to_verdict_tolerates_shapes(out, expected):
    assert Genome._to_verdict(out) == expected


# ---------------------------------------------------------------------------------
# End-to-end evolution
# ---------------------------------------------------------------------------------

def _run_best_fitness():
    cfg = Config(population_size=12, generations=3, seed=7)
    ev = FitnessEvaluator(build_simulated_dataset(), MockLLMClient())
    with redirect_stdout(io.StringIO()):
        return NEATJudge(cfg, ev, MockLLMClient()).run().fitness


def test_evolution_is_reproducible():
    assert abs(_run_best_fitness() - _run_best_fitness()) < 1e-9


def test_evolution_discovers_specialist_and_improves():
    cfg = Config(population_size=12, generations=3, seed=7)
    ev = FitnessEvaluator(build_simulated_dataset(), MockLLMClient())
    with redirect_stdout(io.StringIO()):
        best = NEATJudge(cfg, ev, MockLLMClient()).run()
    # Base generalist alone cannot exceed the safety ceiling; a specialist must
    # have been spliced in for the run to reach high fitness.
    assert best.fitness > 90.0
    assert any(n.node_type == NodeType.HIDDEN for n in best.nodes.values())


def test_golden_dataset_is_clean_of_cues():
    # The clean (real-LLM) dataset must NOT leak the latent cue tokens.
    for item in build_golden_dataset():
        assert "<<safety:" not in item["response"]
        assert "<<quality:" not in item["response"]
    # The simulated dataset embeds them for the mock.
    assert all("<<safety:" in it["response"] for it in build_simulated_dataset())
