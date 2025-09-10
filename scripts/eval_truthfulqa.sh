#!/usr/bin/env bash
set -euo pipefail
python -c "from neatjudge.utils.io import load_yaml; from neatjudge.evolution.eval_mc1 import eval_truthfulqa_mc1; cfg=load_yaml('config/default.yaml'); eval_truthfulqa_mc1(cfg)"
