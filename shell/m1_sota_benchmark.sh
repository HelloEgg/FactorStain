#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m1_sota_benchmark
start_log m1_sota_benchmark

CONFIG="${SOTA_CONFIG:-configs/m1_sota_benchmark.yaml}"
TIER="${TIER:-1}"
METHOD="${METHOD:-}"
METHODS="${METHODS:-}"
SOTA_SEEDS="${SOTA_SEEDS:-42,43,44}"
export SOTA_SEEDS

"$PYTHON_BIN" scripts/prepare_sota_benchmark.py --config "$CONFIG"

selection="$($PYTHON_BIN -c '
import sys
from factorstain.baselines.registry import resolve_methods
print(",".join(resolve_methods(sys.argv[1], sys.argv[2], sys.argv[3])))
' "$METHOD" "$METHODS" "$TIER")"

echo "Selected benchmark methods: $selection"
if [[ "${FETCH_THIRD_PARTY:-0}" == "1" ]]; then
  "$PYTHON_BIN" third_party/fetch_baselines.py --methods "$selection"
fi

IFS=',' read -r -a selected_methods <<< "$selection"
seeded_method() {
  case "$1" in
    stainnet|staingan|cyclegan|pix2pix|histaugan|cagan|sastaindiff|joint|parallel|factorstain) return 0 ;;
    *) return 1 ;;
  esac
}

run_one_method() {
  local method="$1"
  local gpu="${2:-}"
  if seeded_method "$method"; then
    IFS=',' read -r -a seeds <<< "$SOTA_SEEDS"
    if [[ "${FAST_DEV_RUN:-0}" == "1" ]]; then
      seeds=("${seeds[0]}")
    fi
    for seed in "${seeds[@]}"; do
      if [[ -n "$gpu" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/run_sota_method.py \
          --config "$CONFIG" --method "$method" --seed "$seed"
      else
        "$PYTHON_BIN" scripts/run_sota_method.py --config "$CONFIG" \
          --method "$method" --seed "$seed"
      fi
    done
  elif [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/run_sota_method.py \
      --config "$CONFIG" --method "$method"
  else
    "$PYTHON_BIN" scripts/run_sota_method.py --config "$CONFIG" --method "$method"
  fi
}

if [[ "${PARALLEL_GPU_METHODS:-0}" == "1" ]]; then
  IFS=',' read -r -a gpu_slots <<< "${GPU_SLOTS:-0,1,2,3}"
  slot=0
  active_pids=()
  for method in "${selected_methods[@]}"; do
    gpu="${gpu_slots[$((slot % ${#gpu_slots[@]}))]}"
    run_one_method "$method" "$gpu" &
    active_pids+=("$!")
    slot=$((slot + 1))
    if [[ "${#active_pids[@]}" -ge "${#gpu_slots[@]}" ]]; then
      for pid in "${active_pids[@]}"; do wait "$pid"; done
      active_pids=()
    fi
  done
  for pid in "${active_pids[@]}"; do wait "$pid"; done
else
  for method in "${selected_methods[@]}"; do
    run_one_method "$method"
  done
fi

"$PYTHON_BIN" scripts/aggregate_sota_benchmark.py \
  --config "$CONFIG" --method "$METHOD" --methods "$METHODS" --tier "$TIER"
