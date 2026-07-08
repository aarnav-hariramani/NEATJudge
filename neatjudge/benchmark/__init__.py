"""Benchmark harness comparing NEATJudge against non-NEAT LLM-as-a-judge optimizers.

All methods are evaluated under one unified harness -- the same base model, the
same train/eval split, the same scoring function, and per-method LLM-call
accounting -- so the comparison is apples-to-apples. See ``baselines.py`` for the
methods and their citations, and ``harness.py`` for the shared machinery.
"""

from __future__ import annotations

from .baselines import (
    METHODS,
    run_evoprompt_ga,
    run_gepa_prompt,
    run_neatjudge,
    run_opro,
    run_panel_of_judges,
    run_single_judge,
)
from .harness import (
    BenchmarkResult,
    CountingLLMClient,
    format_table,
    panel_genome,
    score_genome,
    single_node_genome,
)

__all__ = [
    "METHODS",
    "run_single_judge", "run_panel_of_judges", "run_evoprompt_ga",
    "run_opro", "run_gepa_prompt", "run_neatjudge",
    "BenchmarkResult", "CountingLLMClient", "format_table",
    "panel_genome", "score_genome", "single_node_genome",
]
