
# NEATJudge — TruthfulQA (HF) with MC1 — Selector + Prompt Evolution

**What you get**
- Step 1: Clean, modular repo with unit tests.
- Step 2: Hugging Face TruthfulQA (file-based) + **MC1** metric implementation.
- Step 3: **Two training paths**
  - `scripts/train_selector.sh`: evolve a **NEAT selector** that chooses the best few-shot exemplars per question.
  - `scripts/train_prompt.sh`: evolve the **prompt genome** (header + formatting toggles) to maximize MC1.
- Eval + Ablations:
  - `scripts/eval_truthfulqa.sh`: evaluates on the **test split**, auto-loading `runs/best_selector.pkl` and `runs/best_prompt.json` if present.
  - `scripts/run_ablations.sh`: runs a small suite: baseline (no retrieval), MMR only, selector, selector+prompt.

## Install
```bash
pip install -r requirements.txt
```

## Baseline (no training)
For a pure no-retrieval baseline, edit `config/default.yaml`:
```yaml
data:
  bank_frac: 0.0
  val_frac: 0.0
  test_frac: 1.0
selector:
  K: 0
```
Then:
```bash
bash scripts/eval_truthfulqa.sh
```

## Train the NEAT selector
```bash
bash scripts/train_selector.sh
```
This writes `runs/neat_config.ini` (dynamic input size) and `runs/best_selector.pkl`.

## Evolve the prompt
```bash
bash scripts/train_prompt.sh
```
This writes `runs/best_prompt.json`. The evolution is simple but effective: boolean toggles + small header mutations.

## Evaluate on TEST
```bash
bash scripts/eval_truthfulqa.sh
```
If `runs/best_selector.pkl` and/or `runs/best_prompt.json` exist, they are auto-loaded.

## Notes on MC1 and GRATH
- The dataset provides choices + labels; **MC1** is computed by taking the **top-scored choice per question**.
- The GRATH paper reports MC1 gains into the mid‑50s on TruthfulQA (e.g., ~54.7% with Llama2‑Chat‑7B). Our pipeline reports MC1
  from your LLM‑as‑judge; scores will vary by model and settings. (See GRATH for details.)

## Folder layout
- `neatjudge/data`: HF loader, splits, retrieval bank
- `neatjudge/selector`: feature map, NEAT wrapper, MMR
- `neatjudge/prompts`: prompt template + mutation ops
- `neatjudge/evolution`: selector training, prompt training, test-time eval
- `neatjudge/metrics`: MC1
- `neatjudge/ablate`: ablation runner
- `scripts/`: train/eval/ablations
- `tests/`: tiny smoke tests
