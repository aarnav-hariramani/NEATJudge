# NEATJudge (Refactor) — TruthfulQA (HF) + MC1/MC2

This is a cleaned, modular refactor focused on **TruthfulQA** via the official Hugging Face dataset and the **MC1/MC2** metrics. 
It removes the custom eBay dataset and wraps your NEAT-based selector in a reproducible pipeline.

## Key changes
- **Dataset** → `truthful_qa` (Hugging Face), subset `multiple_choice` (configurable via `config/default.yaml`).
- **Metrics** → MC1/MC2 implemented in `neatjudge/eval/metrics.py` and used end-to-end.
- **NEAT** → Your exact `neat.ini` copied verbatim (sha256: b3f195cccdbac4c38a7d63dbbe3c344ebdfdc0157ed9e2fe01f137880b72e6b7).
- **Structure** → See the tree below.
- **Tests** → `tests/` for metrics and selector; add more as you iterate.

## Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # or: pip install -e .
```

## Quick eval (TruthfulQA)
```
python -m neatjudge.eval.evaluate_truthfulqa --config config/default.yaml
```

## Run prompt-evolution + selection
- The selection network is built with `neat` using your `config/neat.ini`.
- Retrieval/shortlisting uses `sentence-transformers` embeddings (`all-MiniLM-L6-v2` by default).
- The LLM evaluator uses `Transformers` causal LMs and computes option log-probs for MC1/MC2.

## Repo layout
```
NEATJudge-Refactor/
  requirements.txt
config/
  default.yaml
  neat.ini
neatjudge/
  __init__.py
  train.py
  data/
    bank_index.py
    loaders.py
    splits.py
  eval/
    evaluate_truthfulqa.py
    metrics.py
  evolution/
    mutate.py
    prompt.py
    selector.py
  llms/
    judge.py
  utils/
    io.py
    logging.py
    seed.py
scripts/
  run_eval_truthfulqa.sh
  run_train.sh
tests/
  test_metrics.py
  test_selector.py
```

## Notes
- Swap `model_name` in `config/default.yaml` under `judge:` to your preferred HF model (e.g., `meta-llama/Llama-3.1-8B-Instruct` or a local one).
- If you also want Ollama, add a parallel `OllamaJudge` (see previous repo); the selector / pipeline APIs are decoupled.
- For paper alignment with GRATH, see considerations in comments and `README_GRATH_NOTES.md`.


### Evaluate a trained champion
After `bash scripts/run_train.sh` finishes, it saves `champion_header.txt` and `champion_genome.pkl` under `runs/<timestamp>/`.
Run a full TruthfulQA eval (MC1/MC2) using them:

```bash
python -m neatjudge.eval.evaluate_truthfulqa \
  --config config/default.yaml \
  --header_path runs/<timestamp>/champion_header.txt \
  --selector_genome runs/<timestamp>/champion_genome.pkl
```
