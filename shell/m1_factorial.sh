#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m1_factorial
start_log m1_factorial
prior_allows_progress m0_probe

"$PYTHON_BIN" scripts/prepare_factorial_split.py --config configs/m1_factorial.yaml
for model in joint parallel factorstain; do
  run_ddp scripts/train_renderer.py --config configs/m1_factorial.yaml --model "$model"
done
"$PYTHON_BIN" scripts/evaluate_renderer.py --config configs/m1_factorial.yaml

