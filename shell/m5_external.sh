#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m5_external
start_log m5_external
prior_allows_progress m4_adapter

"$PYTHON_BIN" scripts/preprocess_camelyon.py --config configs/m5_external.yaml
for model in uni virchow2; do
  run_ddp scripts/extract_external_features.py --config configs/m5_external.yaml --model "$model" --dataset camelyon
  run_ddp scripts/extract_external_features.py --config configs/m5_external.yaml --model "$model" --dataset midog
  if compgen -G "$OUTPUTS_ROOT/m5_external/external_features/camelyon/$model/*.h5" >/dev/null; then
    for representation in raw adapter; do
      if [[ "$representation" == "adapter" && ! -f "$OUTPUTS_ROOT/m4_adapter/checkpoints/$model/best.pt" ]]; then
        continue
      fi
      for center in 0 1 2 3 4; do
        run_ddp scripts/train_external.py --config configs/m5_external.yaml --model "$model" --representation "$representation" --heldout-center "$center"
      done
    done
  fi
done
"$PYTHON_BIN" scripts/run_external_evaluation.py --config configs/m5_external.yaml

