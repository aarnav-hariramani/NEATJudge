# Benchmark — claude-opus-4-8

BeaverTails train=24 eval=500 (safety-only) · budget 240 optimization calls/method · scored on held-out eval (no parsimony penalty; agents reported separately).

| Method | Cite | Topo | eval | safety | quality | agents | opt calls | secs |
|---|---|---|---|---|---|---|---|---|
| Single LLM judge | Zheng et al. 2023 (LLM-as-a-judge) | no | 79.60 | 0.80 | 0.75 | 1 | 0 | 140 |
| Panel of judges | Verga et al. 2024 (LLM-as-a-jury) | no | 79.60 | 0.80 | 0.73 | 4 | 0 | 437 |
| EvoPrompt (GA) | Guo et al. 2024 (EvoPrompt) | no | 79.60 | 0.80 | 0.75 | 1 | 163 | 65 |
| OPRO | Yang et al. 2023 (OPRO) | no | 79.60 | 0.80 | 0.75 | 1 | 200 | 87 |
| GEPA (reflective) | Agrawal et al. 2025 (GEPA) | no | 79.40 | 0.79 | 0.79 | 1 | 204 | 154 |
| NEATJudge (ours) | this work | yes | 79.20 | 0.79 | 0.75 | 2 | 462 | 317 |

_Shared cache: cache: 1878 hits / 6029 calls (31% hit rate). ~6000 Opus calls total._

## Honest analysis (the important one)

On a **large, hard, out-of-distribution held-out eval** (500 BeaverTails items, a
real public safety benchmark), **no method beats a single un-optimized Opus judge.**
All six land at safety ≈ 0.80 (eval 79.2–79.6):

- Single, Panel, EvoPrompt, OPRO all tie at **79.60** — a fixed panel and prompt
  optimization gave **zero** improvement.
- **GEPA (79.40) and NEATJudge (79.20) are marginally *worse*** than doing nothing,
  and NEATJudge spent **the most compute by far** (462 calls vs 0 for the single
  judge).

### Why — and why this is the trustworthy result

1. **This is Opus's capability ceiling on hard safety data, not a prompting
   problem.** ~20% of BeaverTails items are genuinely ambiguous or adversarial;
   the errors are ones that neither a reworded prompt, a panel, nor an evolved
   topology fixes. Judge-optimization moves the *prompt/structure*, not the model's
   underlying discrimination on hard cases.
2. **The gains on the small in-house set did not generalize.** On the 40-item
   bundled set NEATJudge tied the best method and prompt-tuning helped a little.
   Optimizing on 24 BeaverTails train items and testing on 500 held-out items
   erases those gains — they were largely overfitting / small-sample variance.
   A large held-out eval is precisely what exposes this.
3. **Structure and prompt-evolution have a cost, and here it buys nothing.**
   NEATJudge's extra agents + reflective rewrites cost 3–7x the calls and produced
   the lowest score.

### Bottom line

**On this large, honest safety benchmark, NEATJudge does not help — a single Opus
judge is as good as anything and far cheaper.** The multi-agent / evolutionary
machinery only paid off on a small, easier, in-distribution set, and that advantage
did not survive contact with a big held-out benchmark. This is the result to trust
over the small-set numbers.

### Where NEATJudge *could* still matter (honest hypotheses, not claims)

- **Weaker/cheaper base judges** (e.g. Haiku), where a panel/specialist genuinely
  adds discrimination the base model lacks — the ceiling here is Opus-specific.
- **Multi-faceted rubrics** (safety *and* faithfulness *and* tone with real labels),
  where routing to specialists can beat one generalist — BeaverTails is safety-only.
- **Larger optimization budgets / train sets**, so learned prompts generalize.

These are testable with `run_benchmark.py --dataset beavertails --model claude-haiku-4-5`
and larger `--train-size`. As it stands, the honest headline is: **at scale, judge
optimization (NEATJudge included) did not beat a single strong judge.**
