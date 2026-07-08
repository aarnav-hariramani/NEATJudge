# Why the live run didn't improve — diagnosis and fix

The first live run (`docs/live_opus_run.md`) was **flat at 82.81 across all 3
generations** — the bare generalist was never beaten. This documents why, and the
fixes that produced a real improvement on a held-out split.

## Root causes

1. **Prompt mutation did nothing to a real model (the big one).**
   `mutate_prompt` *discarded* the critic's output (`_ = critic.complete(...)`) and
   only (a) bumped a `[[calibration:x]]` marker that **only the mock reads**, and
   (b) appended a fixed canned string. So the single most important optimization
   lever — refining each judge's instruction — had **zero real effect** on Opus.
   With prompts frozen, the only way to change fitness was topology, which was
   itself penalized (below).

2. **The parsimony penalty outweighed the accuracy signal.**
   Adding the one useful structure (a `Safety Arbitrator`) cost `0.4` fitness, but
   on the tiny 8-item set it didn't raise accuracy enough to pay for itself, so
   selection correctly discarded it. Net: no structural improvement was rewardable.

3. **The dataset was too small and noisy.**
   8 items → each is 12.5 fitness points on one axis (coarse, high-variance), with
   little headroom above a strong generalist and subjective quality labels that
   depressed everyone equally.

4. **Real Opus is already a strong generalist judge.**
   82.81 at Gen 0 with no structure. Given (1)–(3), nothing could move it.

## Fixes (each a commit)

| Fix | Commit |
|---|---|
| Reflective `mutate_prompt`: critic reads the node's real errors on a train split and **rewrites the instruction** (GEPA-style) | `fix(genome): make mutate_prompt a real reflective rewrite for live models` |
| Larger **public safety dataset** (40 items, BeaverTails-style taxonomy) + disjoint **train/eval split** | `feat(datasets): add larger public safety evaluation dataset` |
| **Configurable** parsimony penalty (0.4→0.1 live) and **safety-weighted** fitness (0.75) | `feat(eval): make parsimony penalty and safety weight configurable` |
| `CachingLLMClient` so a bigger set/population is affordable | `feat(llm): add thread-safe CachingLLMClient` |
| `run_live` wiring (public data, split, caching, reflection, tuned fitness) | `feat(cli): live runner uses public dataset, caching, reflection, tuned fitness` |

## Evidence — held-out eval split (20 items), Opus 4.8

| | Gen 0 | Gen 1 | Gen 2–4 | Winner |
|---|---|---|---|---|
| **Before** (8-item set, no reflection, penalty 0.4) | 82.81 | 82.81 | 82.81 | bare generalist |
| **After** (40-item split, reflection on, penalty 0.1, safety-weighted) | 92.81 | **94.59** | 94.59 | `Ingestor → Safety Arbitrator → Core Judge` |

After the fixes:
- Fitness **rose 92.81 → 94.59** on the held-out eval split.
- The `Safety Arbitrator` topology was discovered in Gen 1, **protected in its own
  species**, and its species **grew to dominate** (size 1 → 8 by Gen 4) — speciation
  working as intended.
- The Core Judge's instruction was **reflectively rewritten** (it learned its
  failure mode was "under-scoring excellent responses by withholding") → quality_acc
  rose to 0.94; safety_acc 0.95.
- `CachingLLMClient` hit 47% (844/1784 calls), roughly halving API cost.

Full log: `docs/live_reflective_run.md`.

## Why the gain is "only" +1.78 and how to push it further

Opus starts at 92.81 — there is little headroom on this task. To surface larger
gains:
- **Harder / larger data** where a single generalist pass clearly fails
  (`try_load_beavertails()` for thousands of labeled pairs; adversarial jailbreak
  suites).
- **More generations** and larger population (reflection compounds).
- **Lower or zero** complexity penalty while a specialist proves itself.
- **`--evolve-models`**: let the model gene trade Opus for Haiku on easy agents and
  spend Opus only where safety is hard (cost-aware fitness).
- **Ablate** to attribute the gain: `python run_live.py --no-reflect`.
