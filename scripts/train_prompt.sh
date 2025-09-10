
#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from neatjudge.utils.io import load_yaml
from neatjudge.evolution.train_prompt import train_prompt
cfg = load_yaml('config/default.yaml')
path = train_prompt(cfg, selector_path="runs/best_selector.pkl")
print(f"[OK] Saved best prompt to: {path}")
PY
