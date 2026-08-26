#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m0_probe
start_log m0_probe

"$PYTHON_BIN" scripts/build_plism_index.py --config configs/m0_probe.yaml
"$PYTHON_BIN" scripts/inspect_plism.py --config configs/m0_probe.yaml
for model in uni virchow2; do
  run_ddp scripts/extract_fm_features.py --config configs/m0_probe.yaml --model "$model"
done
"$PYTHON_BIN" scripts/run_m0.py --config configs/m0_probe.yaml

