
#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from neatjudge.utils.io import load_yaml
from neatjudge.evolution.train_selector import train_selector
cfg = load_yaml('config/default.yaml')
path = train_selector(cfg)
print(f"[OK] Saved best selector to: {path}")
PY
