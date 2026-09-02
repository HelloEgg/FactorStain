#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m1_factorial
start_log m1_factorial

"$PYTHON_BIN" scripts/prepare_factorial_split.py --config configs/m1_factorial.yaml
models=(joint parallel factorstain)
if [[ "${RUN_REVERSE_ABLATION:-0}" == "1" ]]; then
  models+=(reverse)
fi
for model in "${models[@]}"; do
  run_ddp scripts/train_renderer.py --config configs/m1_factorial.yaml --model "$model"
done
"$PYTHON_BIN" scripts/evaluate_renderer.py --config configs/m1_factorial.yaml
