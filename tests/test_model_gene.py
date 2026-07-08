"""Tests for the model-mutating gene: mutation, inheritance, routing, and cost."""

from __future__ import annotations

import io
import random
from contextlib import redirect_stdout

from neatjudge import (
    Config,
    FitnessEvaluator,
    Genome,
    InnovationTracker,
    LLMClient,
    MockLLMClient,
    ModelRouter,
    NEATJudge,
    NodeType,
    build_simulated_dataset,
)

POOL = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"]


def test_mutate_model_assigns_from_pool():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    assert g.mutate_model(random.Random(0), POOL) is True
    assigned = [n.model for n in g.nodes.values() if n.model is not None]
    assert len(assigned) == 1
    assert assigned[0] in POOL


def test_mutate_model_never_touches_input_node():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    for _ in range(20):
        g.mutate_model(random.Random(_), POOL)
    assert g.nodes[0].node_type == NodeType.INPUT
    assert g.nodes[0].model is None   # the ingestor never gets a model gene


def test_model_gene_is_cloned_and_inherited():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    g.nodes[1].model = "claude-opus-4-8"
    assert g.clone().nodes[1].model == "claude-opus-4-8"

    a, b = g.clone(), g.clone()
    a.fitness, b.fitness = 10.0, 5.0
    child = Genome.crossover(a, b, tr, 2, random.Random(0))
    assert child.nodes[1].model == "claude-opus-4-8"   # inherited through crossover


class _TaggedClient(LLMClient):
    """Returns a verdict tagging which model handled the call (for routing tests)."""

    def __init__(self, tag):
        self.tag = tag
        self.calls = 0

    def complete(self, system_instruction, user_content):
        self.calls += 1
        return ('{"per_axis": {"safety": {"value": "safe", "confidence": 0.9}, '
                f'"quality": {{"value": 3, "confidence": 0.9}}}}, "by": "{self.tag}"')


def test_model_router_dispatches_by_node_gene():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    g.mutate_add_node(random.Random(0))     # adds a hidden specialist (node 2)
    hidden = next(n for n in g.nodes.values() if n.node_type == NodeType.HIDDEN)
    hidden.model = "claude-haiku-4-5"

    default_client = _TaggedClient("default")
    haiku_client = _TaggedClient("haiku")
    router = ModelRouter(default_client, {"claude-haiku-4-5": haiku_client})

    g.evaluate_item(build_simulated_dataset()[0], router)
    assert haiku_client.calls == 1          # the haiku-gened node routed to haiku
    assert default_client.calls >= 1        # the default-gened node(s) routed to default


def test_cost_weight_favors_cheaper_models():
    dataset = build_simulated_dataset()
    tr = InnovationTracker()

    expensive = Genome.base_genome(tr, 1)
    expensive.nodes[1].model = "claude-opus-4-8"     # cost 5.0
    cheap = expensive.clone()
    cheap.nodes[1].model = "claude-haiku-4-5"        # cost 1.0

    # Mock is model-agnostic, so accuracy is identical; only the cost term differs.
    ev = FitnessEvaluator(dataset, MockLLMClient(),
                          default_model="claude-opus-4-8", model_cost_weight=1.0)
    ev.evaluate(expensive)
    ev.evaluate(cheap)
    assert cheap.fitness > expensive.fitness

    # With zero cost weight, the model choice must not affect fitness.
    ev0 = FitnessEvaluator(dataset, MockLLMClient(), model_cost_weight=0.0)
    ev0.evaluate(expensive)
    ev0.evaluate(cheap)
    assert abs(cheap.fitness - expensive.fitness) < 1e-9


def test_evolution_with_model_gene_runs_and_assigns_models():
    cfg = Config(
        population_size=12, generations=3, seed=7,
        p_mutate_model=1.0, model_pool=POOL,
        default_model="claude-opus-4-8", model_cost_weight=0.05,
    )
    ev = FitnessEvaluator(build_simulated_dataset(), MockLLMClient(),
                          default_model="claude-opus-4-8", model_cost_weight=0.05)
    with redirect_stdout(io.StringIO()):
        best = NEATJudge(cfg, ev, MockLLMClient()).run()
    # A run with the gene fully on should still produce a valid, high-fitness judge.
    assert best.fitness > 80.0
    assert any(n.model in POOL for n in best.nodes.values())
