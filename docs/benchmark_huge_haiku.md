# Benchmark — claude-haiku-4-5

BeaverTails train=24 eval=500 (safety-only) · budget 240 optimization calls/method · scored on held-out eval (no parsimony penalty; agents reported separately).

| Method | Cite | Topo | eval | safety | quality | agents | opt calls | secs |
|---|---|---|---|---|---|---|---|---|
| Panel of judges | Verga et al. 2024 (LLM-as-a-jury) | no | 78.80 | 0.79 | 0.68 | 4 | 0 | 465 |
| OPRO | Yang et al. 2023 (OPRO) | no | 78.20 | 0.78 | 0.71 | 1 | 224 | 323 |
| Single LLM judge | Zheng et al. 2023 (LLM-as-a-judge) | no | 78.00 | 0.78 | 0.70 | 1 | 0 | 94 |
| EvoPrompt (GA) | Guo et al. 2024 (EvoPrompt) | no | 78.00 | 0.78 | 0.70 | 1 | 115 | 52 |
| GEPA (reflective) | Agrawal et al. 2025 (GEPA) | no | 78.00 | 0.78 | 0.70 | 1 | 204 | 72 |
| NEATJudge (ours) | this work | yes | 74.40 | 0.74 | 0.68 | 1 | 462 | 304 |

_Shared cache: cache: 1381 hits / 5505 calls (25% hit rate)._

## Analysis — does structure help a weaker judge?

This is the run where multi-agent/evolutionary judging had its best honest shot:
a weaker base model (Haiku) should leave more headroom for structure to add
discrimination the model lacks. On the **same 500 held-out BeaverTails items** as the
Opus run:

- **A fixed Panel is the only clear win: 78.00 → 78.80** (safety 0.78 → 0.79). So
  yes — for a weaker judge, a fixed diverse panel (LLM-as-a-jury) gives a small but
  real lift that it did *not* give Opus.
- **OPRO** nudged +0.20; **EvoPrompt / GEPA** were flat.
- **NEATJudge actively hurt: 78.00 → 74.40** — the *worst* result, at the *highest*
  cost (462 calls), and it evolved **no** surviving specialist (agents=1). Its
  reflective prompt rewrite — the critic here is *Haiku itself* — produced an
  instruction that generalized worse on the held-out set. A weak model is a weak
  self-critic: reflective self-improvement backfired.

### Honest conclusion

The "structure helps weaker judges" hypothesis is **partially confirmed, but not for
NEATJudge**: a *fixed* panel helps Haiku a little; *evolved/reflective* optimization
(NEATJudge, driven by the weak model as its own critic) degrades it. The lever that
worked was hand-designed structure, not search.

## Cross-model summary (500 BeaverTails items, held-out, safety-only)

| Method | Opus 4.8 | Haiku 4.5 |
|---|---|---|
| Single judge | 79.60 | 78.00 |
| Panel of judges | 79.60 | **78.80** |
| EvoPrompt (GA) | 79.60 | 78.00 |
| OPRO | 79.60 | 78.20 |
| GEPA (reflective) | 79.40 | 78.00 |
| NEATJudge (ours) | 79.20 | 74.40 |

Takeaways across both models:
1. **The base judge dominates.** Opus (79.60) only ~1.6 pts above Haiku (78.00) —
   BeaverTails safety judging is not very model-sensitive at this tier, so the ceiling
   is close for both.
2. **A fixed panel is the only method that ever beat the single judge**, and only for
   the weaker model (+0.8 on Haiku; +0.0 on Opus).
3. **NEATJudge is last at scale on both models** and *harmful* on Haiku. Its
   evolutionary + reflective machinery, whose critic is the judge model itself, does
   not pay for its cost on this large held-out safety task; with a weak critic it
   makes things worse.

The defensible recommendation from this whole benchmark suite: **use a single strong
judge, or a small fixed diverse panel for weaker judges; do not spend an
evolutionary/reflective optimization budget for LLM-as-a-judge safety at this scale.**
NEATJudge's evolutionary framing remains a clean, well-engineered research artifact,
but it did not win where it counts.
