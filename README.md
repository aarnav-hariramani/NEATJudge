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
