# Seed-averaged benchmark sweep — claude-opus-4-8

Seeds: `[1, 2, 3, 4, 5]` · dataset limit 40 (≈20 train / 20 eval per seed) · budget 240 optimization calls/method · scored on held-out eval.

Each seed is a different train/eval split and optimizer RNG; values are mean ± std across seeds (higher eval is better).

| Method | Cite | Topo | eval mean | ±std | safety | quality | agents | calls |
|---|---|---|---|---|---|---|---|---|
| Panel of judges | Verga et al. 2024 (LLM-as-a-jury) | no | 94.31 | 4.05 | 0.96 | 0.89 | 4.0 | 0 |
| NEATJudge (ours) | this work | yes | 94.12 | 5.51 | 0.95 | 0.92 | 1.6 | 410 |
| Single LLM judge | Zheng et al. 2023 (LLM-as-a-judge) | no | 93.62 | 5.37 | 0.95 | 0.90 | 1.0 | 0 |
| OPRO | Yang et al. 2023 (OPRO) | no | 93.56 | 5.30 | 0.95 | 0.89 | 1.0 | 176 |
| GEPA (reflective) | Agrawal et al. 2025 (GEPA) | no | 93.44 | 2.99 | 0.95 | 0.89 | 1.0 | 143 |
| EvoPrompt (GA) | Guo et al. 2024 (EvoPrompt) | no | 93.31 | 5.25 | 0.95 | 0.88 | 1.0 | 150 |

## Per-seed eval fitness

| Method | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
|---|---|---|---|---|---|
| Single LLM judge | 98.44 | 85.00 | 93.12 | 93.75 | 97.81 |
| Panel of judges | 98.44 | 88.44 | 93.12 | 93.75 | 97.81 |
| EvoPrompt (GA) | 98.12 | 85.00 | 93.12 | 92.81 | 97.50 |
| OPRO | 98.12 | 85.00 | 93.12 | 93.75 | 97.81 |
| GEPA (reflective) | 93.44 | 89.38 | 93.44 | 93.12 | 97.81 |
| NEATJudge (ours) | 99.06 | 85.00 | 94.37 | 94.37 | 97.81 |

_Shared cache: cache: 2729 hits / 5357 calls (51% hit rate). Total wall-clock 1771s._

## Honest analysis

**Top two are a statistical tie: Panel of judges (94.31) and NEATJudge (94.12).**
They differ by 0.19 points with std ≈ 4–5.5 across 5 seeds — indistinguishable.
Both edge the un-optimized single judge (93.62), and the prompt-only optimizers
(OPRO 93.56, GEPA 93.44, EvoPrompt 93.31) land at or just below the baseline.

What actually holds up across seeds:

- **Multi-agent structure is where the (small) gains are.** The two structural
  methods — a fixed Panel and NEATJudge's evolved topology — are the only ones
  above the baseline. Prompt-only optimization did **not** reliably beat a single
  well-prompted Opus judge here, and sometimes hurt (EvoPrompt seed 4: 92.81 <
  93.75; GEPA seed 1: 93.44 < 98.44).
- **NEATJudge has the best quality axis (0.92 vs 0.88–0.90)** and reaches the top
  tier with **1.6 agents on average vs the Panel's fixed 4** — i.e. comparable
  accuracy from a leaner, evolved graph.
- **But NEATJudge is the most expensive** (≈410 optimization calls/seed vs 0 for
  Single/Panel and 143–176 for the prompt optimizers), and its edge over the
  baseline is within noise. It is *competitive with the best*, not a clear winner.

**Why this differs from the earlier single-split run** (`docs/benchmark_results.md`,
where NEATJudge ranked last at 84.90): that used one 12-item split and was dominated
by variance. Averaging over 5 larger (20-item) splits moved NEATJudge from
"unlucky-last" to "co-best," which is exactly the point of seed-averaging — treat any
single small-eval ranking with suspicion.

## Honest bottom line

On this safety-judging task with Opus 4.8, **NEATJudge is among the best methods
(tied with a fixed panel, ahead of prompt-only optimizers and the baseline), with the
best quality scores and fewer agents than the panel — but at the highest compute cost,
and without a statistically significant win.** The result supports NEATJudge's premise
(structure helps) more than a strong "NEAT beats everything" claim. A larger eval
(hundreds of items via `try_load_beavertails()`), more seeds, and a call budget matched
across methods would be needed to separate NEATJudge from a fixed panel with confidence.

## Fairness caveats

- NEATJudge's engine runs its full pop×gens rather than hard-stopping at the shared
  `--budget`, so it spent more calls than the prompt optimizers. A call-matched
  comparison is future work.
- Final scoring uses no parsimony penalty (pure accuracy); agent count is reported
  separately so the Panel's 4 agents vs NEAT's ~1.6 is visible.
