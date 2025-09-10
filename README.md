# NEATJudge (Refactor)

A clean, modular pipeline for **LLM-as-Judge** with **NEAT**-based example selector and **TruthfulQA** evaluation (MC1/MC2).  
This refactor removes the eBay dataset and standardizes on the Hugging Face `truthful_qa` dataset.

## Highlights
- **Datasets**: loads `truthful_qa` (multiple_choice) directly from Hugging Face (`datasets`).
- **Metrics**: MC1/MC2 using token log-likelihoods (Transformers backend) or a parsing fallback (Chat backends).
- **Evolution**: NEAT selects few-shot examples from a *bank* to condition the judge prompt; prompt header mutates locally.
- **Structure**: one module per responsibility: data, evolution, llms, eval, utils.
- **Testing**: fast unit tests (no model downloads) with mocked backends.

## Quickstart
```bash
# 1) Install
pip install -r requirements.txt

# 2) Evaluate TruthfulQA (MC1) using a local HF model
python -m neatjudge.eval.evaluate_truthfulqa --config config/default.yaml

# 3) Evolve selector + prompt against TruthfulQA validation
python -m neatjudge.train --config config/default.yaml

# 4) Run tests
pytest -q
```

### Config Overview (`config/default.yaml`)
- `data.dataset`: `truthfulqa_hf`
- `llm.backend`: `transformers` (recommended for MC1/MC2) or `ollama`
- `evolution`: NEAT population/generations, bank sizes
- `selector.embed_model`: Sentence-Transformers name for embeddings

See inline comments in the config file.

## Notes on MC1/MC2
For **MC1**, we choose the option with the highest normalized log-probability given `(question + option)` with a consistent prompt template.  
For **MC2**, we sum probabilities for *all* correct options and normalize by total across options.

> If you use `ollama` (chat-style) instead of `transformers`, MC1 uses a *parsing* fallback: we ask the model to output the letter (A/B/C/...). This is not strictly identical to the leaderboard's token-level probability evaluation. For publication-grade numbers, use the **transformers** backend.

## NEAT & Prompt Evolution
We use NEAT to score candidate few-shot banks for a query; the **Selector** net ranks bank examples from embeddings.  
Prompt evolution performs **minimal, local edits** to a sectioned header (role/scale/output/ties/extra) to avoid rubric bloat.

---
