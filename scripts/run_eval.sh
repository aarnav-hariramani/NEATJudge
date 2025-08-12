#!/usr/bin/env bash
set -e
python evaluate.py --config config/default.yaml --ckpt runs/ckpts/best.pkl
