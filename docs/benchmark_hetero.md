# Heterogeneous-models thesis test — HelpSteer2 multi-axis (held-out)

Best single judge: **llama-4-mav @ 83.25**. train=24 eval=200 axes=['correctness', 'coherence', 'helpfulness'].

| Config | kind | eval | vs best single | agents |
|---|---|---|---|---|
| llama-4-mav | single | 83.25 | +0.00 | 1 |
| jury: best-3 (opus-4-8+gpt-4o+llama-4-mav) | jury | 82.79 | -0.46 | 3 |
| jury: cross-provider-4 | jury | 81.71 | -1.54 | 4 |
| NEATJudge (hetero) | evolved | 81.42 | -1.83 | 2 |
| jury: all-6 (diverse) | jury | 80.88 | -2.38 | 6 |
| mixed panel (gpt/gem/son->opus) | panel | 80.79 | -2.46 | 4 |
| opus-4-8 | single | 80.67 | -2.58 | 1 |
| gpt-4o | single | 80.38 | -2.88 | 1 |
| jury: claude-3 (same family) | jury | 78.62 | -4.63 | 3 |
| sonnet-4-6 | single | 78.29 | -4.96 | 1 |
| haiku-4-5 | single | 75.96 | -7.29 | 1 |
| gemini-flash | single | 68.04 | -15.21 | 1 |

**Thesis NOT confirmed**: no heterogeneous configuration beat the best single judge.

_Models: opus-4-8, sonnet-4-6, haiku-4-5, gpt-4o, gemini-flash, llama-4-mav._

## Analysis — the decisive negative result

This was the last honest hope for multi-agent judging: genuinely *different* models
(Claude, GPT-4o, Gemini, Llama) whose errors should be decorrelated, so a jury could
beat any single judge. It did not.

**Why the jury lost to the best single judge:**

1. **Averaging beats the *average* member, not the *best* member.** Llama-4-maverick
   alone (83.25) is clearly the strongest judge here; every other model is weaker
   (Opus 80.7, GPT-4o 80.4, and Gemini a distant 68.0). Averaging the best model with
   weaker ones pulls the ensemble *toward the mean* — below the top model. A jury only
   beats its best member when members are **comparably strong** AND have **independent,
   unbiased** errors.
2. **Even a curated jury of the three comparable strong models** (opus+gpt-4o+llama, all
   ~80–83) reached only **82.79 — still under llama's 83.25.** So even after removing the
   weak models, the errors were not independent enough to net a gain: LLM judges share
   *systematic* biases (they tend to agree on the same hard/ambiguous items in the same
   wrong direction), so cross-provider error correlation is higher than the ensemble
   theory assumes.
3. **Heterogeneous NEATJudge correctly discovered llama** (its model gene converged to
   using llama-4-mav) — a nice sign the machinery works — **but its extra agent + reflective
   prompt rewrite still dragged it to 81.42, below llama-alone.** Evolution found the best
   model, then its own scaffolding subtracted value.

**The most useful practical finding of the entire project:** on HelpSteer2,
**Llama-4-maverick is a better judge than Opus 4.8 or GPT-4o** — and picking that single
best model (83.25) beats *every* multi-agent / evolved / jury configuration, including
NEATJudge's best-ever result on this task (81.92 with Opus-only). **Model *selection*
beat model *combination*.**

## Book closed

Across the full arc — small set, seed-averaged, 500-item safety (Opus & Haiku),
multi-faceted rubric, and now cross-provider heterogeneous juries/topologies — **no
multi-agent or evolutionary configuration reliably beats the best single judge.** The
reliable levers, in order of impact:

1. **Pick the best base model for the task** (biggest lever by far — 15+ points here).
2. A light **reflective prompt** tune (small, task-dependent, can backfire with a weak critic).
3. Multi-agent structure / topology search: **not worth its cost** for LLM-as-a-judge.

NEATJudge is a clean, fully-tested, honestly-benchmarked artifact that answered its own
question in the negative. That is a real result.
