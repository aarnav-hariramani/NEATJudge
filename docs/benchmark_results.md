# Benchmark: NEATJudge vs. non-NEAT LLM-as-a-judge optimizers

> **Note:** this is a single 12-item split and is dominated by variance (NEATJudge
> ranks last here purely by luck of the split). The reliable, seed-averaged result
> is in [`benchmark_sweep.md`](benchmark_sweep.md), where NEATJudge is co-best.
> This page is kept as an illustration of why single small-eval rankings are noisy.


A live, apples-to-apples comparison on **claude-opus-4-8**. Every method optimizes
a judge on the **train** split and is scored on the **held-out eval** split by the
same scoring function (safety-weighted 0.75, no parsimony penalty → pure accuracy),
through a shared cached client with per-method LLM-call accounting.

```bash
python run_benchmark.py --limit 24 --budget 240 --max-tokens 400
# dataset=24 (train=12, eval=12), budget=240 optimization calls/method
```

## Results (held-out eval, higher is better)

| Method | Cite | Evolves topology | eval fitness | safety | quality | agents | opt calls | secs |
|---|---|---|---|---|---|---|---|---|
| **OPRO** | Yang et al. 2023 | no | **91.15** | 0.92 | 0.90 | 1 | 116 | 263 |
| **Panel of judges** | Verga et al. 2024 | no | **90.62** | 0.92 | 0.88 | 4 | 0 | 106 |
| Single LLM judge | Zheng et al. 2023 | no | 84.90 | 0.83 | 0.90 | 1 | 0 | 33 |
| EvoPrompt (GA) | Guo et al. 2024 | no | 84.90 | 0.83 | 0.90 | 1 | 115 | 236 |
| GEPA (reflective) | Agrawal et al. 2025 | no | 84.90 | 0.83 | 0.90 | 1 | 120 | 180 |
| **NEATJudge (ours)** | this work | **yes** | 84.90 | 0.83 | 0.90 | 2 | 246 | 227 |

(Shared cache: 257 hits / 717 calls = 36% dedup.)

## Honest read of these numbers

**NEATJudge did not win this benchmark.** On this 12-item held-out eval it tied the
un-optimized single judge (84.90) — even though it evolved a specialist (2 agents) —
and it spent the *most* optimization calls (246). The strongest methods here were:

- **OPRO (91.15)** — a single optimizer-rewritten Core Judge prompt generalized
  well, lifting safety 0.83 → 0.92 with no extra agents.
- **Panel of judges (90.62)** — a *fixed* 3-specialist panel (no optimization at
  all) reached safety 0.92 purely from structure. This supports NEATJudge's core
  premise that multi-agent structure helps safety — but the *evolved* topology did
  not capture that gain on this split.
- **EvoPrompt and GEPA** matched the baseline: their train-best prompts did not
  transfer to the eval split.

## Why NEATJudge underperformed here (and why it's split-sensitive)

1. **Tiny eval (12 items).** Each item is ~6.25 safety points, so 84.90 vs 91.15 is
   ~1 item. The ranking is *indicative, not definitive* — high variance.
2. **Train→eval generalization.** NEAT selects the train-best genome; on 12 train
   items its evolved specialist route did not change any of the 12 eval verdicts
   (its eval numbers are identical to the single judge). OPRO's single-prompt search
   happened to generalize better on this split.
3. **Selection budget.** With pop 5 / gens 3, NEAT explores topology *and* prompts
   with the same call budget others spend entirely on one prompt — so per-call it
   was less efficient here.
4. **Split sensitivity.** In a separate standalone run on a 20-item eval split
   (`docs/live_reflective_run.md`), NEATJudge *did* improve 92.81 → 94.59 and its
   Safety-Arbitrator species dominated. Different split, different outcome — which is
   exactly why the small eval here should not be over-read.

## Takeaways

- The harness is fair and reproducible: identical model, split, scoring, and
  call-accounting for every method (`neatjudge/benchmark/`).
- On a small safety eval, **strong prompt optimization (OPRO) and fixed panels are
  hard baselines** — NEATJudge's added machinery (topology search + speciation) did
  not pay off at this scale/budget.
- Multi-agent structure clearly helps (Panel), so NEATJudge's premise is sound; the
  open question is whether *evolving* structure beats a well-chosen *fixed* panel
  given a fair budget. This benchmark says: not yet, at this scale.

## Reproduce / strengthen the conclusion

Small evals are noisy. For a reliable ranking, run larger and average over seeds:

```bash
python run_benchmark.py --limit 40 --budget 500 --max-tokens 400   # full bundled set
# swap in thousands of labeled pairs:
python - <<'PY'
from neatjudge import try_load_beavertails
print(len(try_load_beavertails(limit=500) or []))   # needs `pip install datasets`
PY
# average run_benchmark over --seed 1..5 and compare mean eval fitness.
```
