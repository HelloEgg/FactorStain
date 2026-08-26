#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m4_adapter
start_log m4_adapter
prior_allows_progress m3_attribution

for model in uni virchow2; do
  run_ddp scripts/extract_counterfactual_features.py --config configs/m4_adapter.yaml --model "$model"
  if [[ -f "$OUTPUTS_ROOT/m4_adapter/counterfactual_features/$model/features.h5" ]]; then
    run_ddp scripts/train_adapter.py --config configs/m4_adapter.yaml --model "$model"
  else
    echo "Skipping adapter training for $model because model access/features are unavailable."
  fi
done
"$PYTHON_BIN" scripts/evaluate_adapter.py --config configs/m4_adapter.yaml

