#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT}"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT/outputs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
  TORCHRUN_BIN="$ROOT/.venv/bin/torchrun"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
  TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
fi

acquire_lock() {
  local milestone="$1"
  mkdir -p "$OUTPUTS_ROOT/.locks"
  exec 9>"$OUTPUTS_ROOT/.locks/${milestone}.lock"
  if command -v flock >/dev/null 2>&1; then
    if ! flock -n 9; then
      echo "${milestone} is already running; leaving the active resumable run untouched."
      exit 0
    fi
  fi
}

start_log() {
  local milestone="$1"
  mkdir -p "$OUTPUTS_ROOT/$milestone/logs"
  exec > >(tee -a "$OUTPUTS_ROOT/$milestone/logs/launcher.log") 2>&1
  echo "[$(date --iso-8601=seconds)] Starting $milestone"
}

prior_allows_progress() {
  local milestone="$1"
  local decision="$OUTPUTS_ROOT/$milestone/GO_NOGO.json"
  if [[ ! -f "$decision" ]]; then
    echo "Required prior decision is missing: $decision"
    exit 2
  fi
  local status
  status="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$decision")"
  if [[ "$status" == "NO_GO" && "${FORCE_CONTINUE:-0}" != "1" ]]; then
    echo "$milestone is NO_GO. Set FORCE_CONTINUE=1 only for an explicitly scoped diagnostic continuation."
    exit 3
  fi
}

run_ddp() {
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" "$@"
}

