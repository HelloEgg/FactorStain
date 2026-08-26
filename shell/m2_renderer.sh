#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m2_renderer
start_log m2_renderer
prior_allows_progress m1_factorial

"$PYTHON_BIN" scripts/prepare_factorial_split.py --config configs/m2_renderer.yaml
for model in factorstain reverse parallel joint; do
  run_ddp scripts/train_renderer.py --config configs/m2_renderer.yaml --model "$model"
done
"$PYTHON_BIN" scripts/evaluate_renderer.py --config configs/m2_renderer.yaml

