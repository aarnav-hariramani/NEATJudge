
#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from neatjudge.utils.io import load_yaml
from neatjudge.evolution.eval_mc1 import eval_truthfulqa_mc1
cfg = load_yaml('config/default.yaml')
# Auto-load trained selector/prompt if present
eval_truthfulqa_mc1(cfg, genome_path="runs/best_selector.pkl", prompt_path="runs/best_prompt.json", neat_config_path="runs/neat_config.ini")
PY
