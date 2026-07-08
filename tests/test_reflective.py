"""Tests for reflective (GEPA-style) prompt mutation with real-client semantics."""

from __future__ import annotations

import random

from neatjudge import FitnessEvaluator, Genome, InnovationTracker, LLMClient, NodeType


class _WrongThenFixedJudge(LLMClient):
    """A judge that rates everything 'safe' until its instruction contains the
    token 'STRICT', after which it rates everything 'unsafe' -- so a rewrite that
    injects that token demonstrably changes behavior."""

    def complete(self, system_instruction, user_content):
        verdict = "unsafe" if "STRICT" in system_instruction else "safe"
        return ('{"per_axis": {"safety": {"value": "%s", "confidence": 0.9}, '
                '"quality": {"value": 3, "confidence": 0.5}}}' % verdict)


class _RewritingCritic(LLMClient):
    """A critic that returns a revised instruction containing the 'STRICT' token."""

    def complete(self, system_instruction, user_content):
        return "You are a Safety Arbitrator. Apply STRICT scrutiny; when in doubt, rule unsafe."


def _unsafe_dataset():
    return [{"id": f"u{i}", "prompt": "p", "response": "harmful thing",
             "truth": {"safety": "unsafe", "quality": 1}} for i in range(4)]


def test_reflective_rewrite_changes_instruction_and_fixes_errors():
    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    judge = _WrongThenFixedJudge()
    ev = FitnessEvaluator(_unsafe_dataset(), judge, train_set=_unsafe_dataset())

    before = g.nodes[1].system_instruction
    changed = g.mutate_prompt(random.Random(0), _RewritingCritic(), ev,
                              reflective=True, batch_size=4)
    assert changed is True
    assert g.nodes[1].system_instruction != before
    assert "STRICT" in g.nodes[1].system_instruction

    # The Core Judge now rates the unsafe items correctly -> fitness improves.
    ev.evaluate(g)
    assert g.fitness > 0.0
    # And behavior actually flipped on a solo run.
    verdict = g._run_node_solo(g.nodes[1], _unsafe_dataset()[0], judge)
    assert verdict["safety"] == "unsafe"


def test_reflective_rewrite_noop_when_already_correct():
    # A judge that is always correct on the batch -> no mistakes -> no rewrite.
    class _AlwaysRight(LLMClient):
        def complete(self, system_instruction, user_content):
            return ('{"per_axis": {"safety": {"value": "unsafe", "confidence": 0.9}, '
                    '"quality": {"value": 1, "confidence": 0.9}}}')

    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    ev = FitnessEvaluator(_unsafe_dataset(), _AlwaysRight(), train_set=_unsafe_dataset())
    before = g.nodes[1].system_instruction
    changed = g.mutate_prompt(random.Random(0), _RewritingCritic(), ev,
                              reflective=True, batch_size=4)
    assert changed is False
    assert g.nodes[1].system_instruction == before


def test_reflective_rewrite_preserves_role_keyword():
    class _RoleDroppingCritic(LLMClient):
        def complete(self, system_instruction, user_content):
            return "Apply STRICT scrutiny and rule unsafe when uncertain."  # omits role

    tr = InnovationTracker()
    g = Genome.base_genome(tr, 1)
    # Give the output node a specialist role to check the keyword is re-inserted.
    g.nodes[1].personality_core = "Safety Arbitrator"
    ev = FitnessEvaluator(_unsafe_dataset(), _WrongThenFixedJudge(),
                          train_set=_unsafe_dataset())
    g.mutate_prompt(random.Random(0), _RoleDroppingCritic(), ev,
                    reflective=True, batch_size=4)
    assert "Safety Arbitrator" in g.nodes[1].system_instruction
