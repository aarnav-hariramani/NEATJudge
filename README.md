# NEATJudge

**Evolving multi-agent LLM-as-a-Judge communication topologies with NEAT.**

NEATJudge is a conceptual translation of **NEAT** (NeuroEvolution of Augmenting
Topologies) to **LLM-as-a-Judge** architectures. Instead of evolving neural
network weights and layers, it evolves *multi-agent communication topologies* —
directed acyclic graphs where **nodes are specialized judge agents** and
**directed edges are context/data pathways** between them.

```
INPUT ──▶ Safety Arbitrator ──▶ Core Judge (verdict)
        (specialist sub-judge)   (aggregator)
```

## Core concepts

| NEAT concept | NEATJudge realization |
|---|---|
| **Innovation numbers** | Every structural change (new pathway, or a node splitting a pathway) gets a stable, memoized historical id. Identical mutations share an id across genomes → safe crossover. (`innovation.py`) |
| **Nodes** | Judge agents with an **immutable personality core** (Safety Arbitrator, Fact-Checker, Tone/Coherence/Relevance Judge, Core Judge) and a **mutable system instruction**. (`genes.py`, `archetypes.py`) |
| **Edges (synapses)** | Directed context routes. If A→B, then B receives A's judgment in its context window. Weights are expressed **textually** (priority/routing instructions) with a numeric companion. (`genes.py`) |
| **`mutate_add_node`** | Splits an existing edge, inserting a specialist sub-judge (A→B becomes A→C→B; the original edge is *disabled*, not deleted). |
| **`mutate_add_edge`** | Adds a new feed-forward context path between two agents (cycle-guarded). |
| **`mutate_prompt`** | GEPA-style reflective mutation: an LLM *Critic* reviews a node's behavior on a batch and refines its system instruction. |
| **Speciation** | Clusters graphs by topological similarity (shared innovation numbers) and shares fitness, protecting young structural strategies until their prompts optimize. (`speciation.py`) |
| **Crossover** | Innovation-aligned: matching genes random from either parent; disjoint/excess from the fitter parent; disabled-gene rule; cyclic recombinations repaired to a DAG. (`genome.py`) |

## Install

```bash
pip install -e .            # core library (pure standard library)
pip install -e ".[anthropic]"   # + real Claude judges
pip install -e ".[test]"        # + pytest
```

## Run the offline demo (no API key)

```bash
python run_demo.py
```

A deterministic `MockLLMClient` drives a full 3-generation evolution. The fitness
landscape is designed so the generalist Core Judge **cannot** clear the safety bar
by prompt-tuning alone — a `Safety Arbitrator` sub-judge (topology) is required —
so you watch a structural innovation appear, get **protected in its own species**,
and then out-compete the prompt-only lineage:

```
Gen 0  species=1  best=62.50   (all bare Core Judges)
Gen 1  species=2  best=80.85   (Safety Arbitrator split appears, alone in species 2)
Gen 2  species=2  best=99.60   (that species grows and wins)
BEST: INPUT → Safety Arbitrator → Core Judge
```

## Run against a real LLM

Set your key, then use `AnthropicClient` (or `OpenAIClient`) instead of the mock —
and the **clean** golden dataset (no embedded cues):

```python
from neatjudge import (Config, FitnessEvaluator, NEATJudge,
                       AnthropicClient, build_golden_dataset)

cfg = Config(population_size=12, generations=3, seed=7, eval_workers=8)
judge  = AnthropicClient("claude-opus-4-8")
critic = AnthropicClient("claude-opus-4-8")
evaluator = FitnessEvaluator(build_golden_dataset(), judge)
best = NEATJudge(cfg, evaluator, critic).run()
```

> **Datasets.** `build_simulated_dataset()` embeds latent cues (`<<safety:...>>`)
> for the mock to read; `build_golden_dataset()` is clean text with hidden labels
> for real judges. Every node's system prompt carries `OUTPUT_CONTRACT`, which
> tells real models to emit the parseable per-axis verdict JSON.

## The model-mutating gene

Beyond topology and prompts, NEATJudge can evolve **which model runs each agent**.
Every `NodeGene` carries a `model` gene (heritable through crossover); `mutate_model`
reassigns it from `Config.model_pool`, and a `ModelRouter` dispatches each agent to
the matching client at evaluation time. A cost-aware fitness term
(`model_cost_weight * Σ MODEL_COST`) rewards the *cheapest* model that still holds
accuracy — so an Opus→Haiku downgrade that costs no fitness is selected for.

```bash
python run_live.py                 # every agent on claude-opus-4-8 (gene pinned)
python run_live.py --evolve-models # gene explores {opus-4-8, sonnet-4-6, haiku-4-5}
```

```python
from neatjudge import AnthropicClient, ModelRouter
router = (ModelRouter(AnthropicClient("claude-haiku-4-5"))
          .register("claude-opus-4-8", AnthropicClient("claude-opus-4-8")))
# Agents with model gene "claude-opus-4-8" run on Opus; everything else on Haiku.
```

## Results & diagnosis

On real Opus 4.8 over the bundled 40-item public safety set (held-out eval split),
evolution improves **92.81 → 94.59**, selecting a `Safety Arbitrator → Core Judge`
topology and reflectively rewriting the Core Judge's prompt (final safety_acc 0.95,
quality_acc 0.94). See [`docs/live_reflective_run.md`](docs/live_reflective_run.md).

An earlier run was flat — [`DIAGNOSIS.md`](DIAGNOSIS.md) explains why (prompt
mutation was a no-op for real models; parsimony penalty outweighed a tiny/noisy
8-item signal) and the fixes (reflective rewrite, larger public dataset with
train/eval split, configurable penalty + safety weighting, caching).

```bash
python run_live.py                 # reflective evolution on Opus
python run_live.py --no-reflect    # ablation: reflection off
```

## Benchmark vs. other methods

A unified, apples-to-apples harness (`neatjudge/benchmark/`) compares NEATJudge
against non-NEAT LLM-as-a-judge methods — same model, train/eval split, scoring,
and per-method call accounting. Faithful reimplementations, cited in
`neatjudge/benchmark/baselines.py`.

### Large held-out benchmark — the result to trust (500 BeaverTails items)

Held-out eval, safety-only, on the same items for both models:

| Method | topo | Opus 4.8 | Haiku 4.5 | opt calls |
|---|---|---|---|---|
| Single judge (Zheng '23) | no | **79.60** | 78.00 | 0 |
| Panel of judges (Verga '24) | no | 79.60 | **78.80** | 0 |
| EvoPrompt GA (Guo '24) | no | 79.60 | 78.00 | ~140 |
| OPRO (Yang '23) | no | 79.60 | 78.20 | ~210 |
| GEPA (Agrawal '25) | no | 79.40 | 78.00 | ~204 |
| NEATJudge (ours) | yes | 79.20 | 74.40 | 462 |

**Honest headline: at scale on a hard public safety benchmark, no optimizer beats a
single judge — and NEATJudge is last on both models** (marginally on Opus, badly on
Haiku, where its weak-model self-reflection actively hurts). The *only* method that
ever beat the single judge is a **fixed diverse panel, and only for the weaker model**
(+0.8 on Haiku, +0.0 on Opus). The base judge dominates; the small in-house-set gains
did not generalize. Full analysis:
[`docs/benchmark_huge.md`](docs/benchmark_huge.md) (Opus) and
[`docs/benchmark_huge_haiku.md`](docs/benchmark_huge_haiku.md) (Haiku + cross-model).

### Multi-faceted rubric — where NEATJudge finally tops the field (HelpSteer2, 300 items, Opus)

Three human-labeled axes (correctness/coherence/helpfulness), each ownable by a
distinct specialist — the setup most favorable to multi-agent judging:

| Method | topo | eval | agents | opt calls |
|---|---|---|---|---|
| **NEATJudge (ours)** | yes | **81.92** | 2 | 618 |
| EvoPrompt GA | no | 81.75 | 1 | 169 |
| GEPA | no | 81.69 | 1 | 215 |
| Single judge | no | 80.97 | 1 | 0 |
| OPRO | no | 80.97 | 1 | 278 |
| Panel of judges | no | 79.75 | 4 | 0 |

**NEATJudge is nominally #1 here — the one setting where it wins — but honestly:** it's
a statistical tie with plain prompt evolution (EvoPrompt/GEPA, ~81.7) at 3–4× the cost,
and the win comes from *evolving the prompt*, not the topology (the pure-structure
**Panel did worst, below the single judge**). Full analysis:
[`docs/benchmark_multiaxis.md`](docs/benchmark_multiaxis.md).

### Heterogeneous (mixed-model) judges — the decisive test

The last hope for multi-agent judging: *different* models (Claude, GPT-4o, Gemini,
Llama) with decorrelated errors, so a jury could beat any single judge. On HelpSteer2
multi-axis (held-out):

| config | eval | vs best single |
|---|---|---|
| **Llama-4-maverick (single)** | **83.25** | best single (the bar) |
| jury: best-3 (opus+gpt-4o+llama) | 82.79 | −0.46 |
| jury: cross-provider-4 | 81.71 | −1.54 |
| NEATJudge (heterogeneous) | 81.42 | −1.83 |
| mixed-model panel | 80.79 | −2.46 |
| Opus 4.8 (single) | 80.67 | −2.58 |

**Thesis NOT confirmed: no jury, panel, or evolved topology beat the best single judge.**
Averaging a dominant model with weaker ones pulls toward the mean; even a curated jury of
the three strong models (82.79) stayed under Llama alone, because LLM judges share
*systematic* biases (correlated errors even across providers). Full analysis:
[`docs/benchmark_hetero.md`](docs/benchmark_hetero.md).

### Final recommendation (whole suite)

The reliable levers for LLM-as-a-judge, by impact:
1. **Pick the best base model for the task** — by far the biggest lever (Llama-4-mav beat
   Opus/GPT-4o by 3+ pts here; the model spread was 15+ pts). *Model selection > model combination.*
2. A light **reflective prompt** tune — small, task-dependent, can backfire with a weak critic.
3. **Multi-agent structure / topology search (NEATJudge): not worth its cost.** It never
   reliably beat the best single judge across safety, multi-axis, weak-model, or
   heterogeneous settings.

NEATJudge is a clean, fully-tested, honestly-benchmarked artifact that answered its own
question in the negative — a real result.

<details><summary>Smaller in-house runs (why single-scale numbers mislead)</summary>

Seed-averaged over 5 splits of the 40-item bundled set, NEATJudge *tied* the best
method (a fixed panel) with the best quality and fewer agents
([`docs/benchmark_sweep.md`](docs/benchmark_sweep.md)). A single 12-item split ranked
it *last* purely from variance ([`docs/benchmark_results.md`](docs/benchmark_results.md)).
Both are superseded by the large held-out result above — small evals are noisy and
in-distribution tuning does not transfer.
</details>

```bash
python run_benchmark.py --dataset beavertails --train-size 24 --eval-size 500 \
    --safety-only --workers 12 --out docs/benchmark_huge.md   # large held-out (live)
python run_sweep.py                # seed-averaged bundled set (live)
python run_benchmark.py --mock     # deterministic offline smoke run
```

## Package layout

```
neatjudge/
  llm.py          LLMClient + MockLLMClient / AnthropicClient / OpenAIClient, OUTPUT_CONTRACT
  innovation.py   InnovationTracker (historical ids)
  archetypes.py   immutable personality cores
  genes.py        NodeGene, ConnectionGene
  genome.py       Genome: mutation, crossover, DAG-safe feed-forward eval
  fitness.py      FitnessEvaluator
  datasets.py     golden (clean) + simulated (cued) datasets
  speciation.py   Species (fitness sharing)
  config.py       Config
  engine.py       NEATJudge evolution loop
run_demo.py       offline reproducible demo
tests/            verification suite
```

## Tests

```bash
pytest -q
```

## License

MIT.
