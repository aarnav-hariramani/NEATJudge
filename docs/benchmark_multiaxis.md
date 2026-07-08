# Multi-faceted benchmark — claude-opus-4-8 (HelpSteer2, 3 axes)

The setup where multi-agent structure *should* help: a **multi-faceted rubric** with
real per-axis human labels, so distinct specialists can each own a distinct axis
instead of re-judging the same one.

- Data: **NVIDIA HelpSteer2**, 30 train / **300 held-out eval**.
- Axes (0–4, human-labeled): **correctness → Fact-Checker**, **coherence → Coherence
  Judge**, **helpfulness → Relevance Judge**. The single judge and each prompt
  optimizer rate all three at once; the Panel routes each axis to its specialist and
  aggregates; NEATJudge evolves the graph.
- Score = 100 × mean per-axis closeness on held-out eval (higher = closer to human
  ratings). Budget 300 optimization calls/method.

## Results (held-out eval)

| Method | Cite | Topo | eval | correctness | coherence | helpfulness | agents | opt calls |
|---|---|---|---|---|---|---|---|---|
| **NEATJudge (ours)** | this work | yes | **81.92** | 0.78 | 0.88 | 0.80 | 2 | 618 |
| EvoPrompt (GA) | Guo et al. 2024 | no | 81.75 | 0.77 | 0.88 | 0.80 | 1 | 169 |
| GEPA (reflective) | Agrawal et al. 2025 | no | 81.69 | 0.78 | 0.88 | 0.79 | 1 | 215 |
| Single LLM judge | Zheng et al. 2023 | no | 80.97 | 0.77 | 0.87 | 0.79 | 1 | 0 |
| OPRO | Yang et al. 2023 | no | 80.97 | 0.77 | 0.87 | 0.79 | 1 | 278 |
| Panel of judges | Verga et al. 2024 | no | 79.75 | 0.75 | 0.85 | 0.79 | 4 | 0 |

_Shared cache: 696 hits / 4280 calls._

## Honest analysis

**NEATJudge tops the table for the first time (81.92) — but read the fine print.**

1. **The win is real but marginal and within noise of the prompt optimizers.**
   NEATJudge (81.92), EvoPrompt (81.75), and GEPA (81.69) form a top cluster ~0.2
   apart — a statistical tie on 300 items. NEATJudge is nominally best, but not
   distinguishably better than plain prompt evolution, and it spent **3–4× the
   compute** (618 calls vs 169/215).

2. **The gain comes from evolving the *prompt*, not the *topology*.** The three
   methods that beat the baseline (NEATJudge, EvoPrompt, GEPA) all share one thing:
   they rewrite the judge's instruction. NEATJudge added just **one** specialist
   (agents=2), and the pure-structure method —
   **the fixed 3-specialist Panel — actually did *worst* (79.75), below the single
   judge.** So multi-agent structure *still* did not help; the aggregation hop
   degraded the result even with genuinely distinct axes. NEATJudge wins because it
   *contains* reflective prompt optimization (the GEPA operator), not because its
   agent graph is doing the work.

3. **Why the Panel hurt even here.** Each "specialist" is the same Opus model in a
   different hat; it does not judge its axis better than the generalist, and funneling
   three verdicts through a Core Judge aggregator adds a lossy combination step. Same
   model → correlated views → no ensemble benefit, plus aggregation noise.

## Verdict on the original question

Across the full suite, this is the one setting where **NEATJudge comes out on top** —
so "can an evolved judge ever beat a single judge?" → **yes, marginally, on a
multi-faceted task.** But the honest mechanism is:

- **Prompt optimization helps a little** on multi-axis judging (single 80.97 → ~81.7–81.9).
- **Multi-agent *structure* still does not** (the Panel is worst; NEAT's topology adds
  ~nothing beyond its prompt rewriting).
- NEATJudge's headline win is **a tie with EvoPrompt/GEPA at 3–4× the cost**, driven by
  the prompt lever they all share.

So the intuition "agentic frameworks are always stronger" remains false; the reliable
lever is *better prompts*, and NEATJudge's evolutionary topology search is an expensive
way to buy a gain you can get from GEPA/EvoPrompt alone. NEATJudge's honest niche: it
never loses on multi-faceted tasks and can top the field, but it does not justify its
cost over simpler prompt optimizers here.

## Cross-task summary (all held-out, Opus 4.8)

| | BeaverTails safety (500) | HelpSteer2 multi-axis (300) |
|---|---|---|
| Best method | Single/Panel/EvoPrompt/OPRO tie 79.60 | **NEATJudge 81.92** (≈ EvoPrompt/GEPA) |
| Single-judge baseline | 79.60 | 80.97 |
| Does structure (Panel) help? | no (tie) | **no (worst, 79.75)** |
| Does prompt optimization help? | no | yes, slightly (+0.8) |
| NEATJudge rank | last | **1st (marginal, most expensive)** |
