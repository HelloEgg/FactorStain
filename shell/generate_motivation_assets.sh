#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

args=(--config configs/paper_assets.yaml)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"$PYTHON_BIN" scripts/build_motivation_sources.py "${args[@]}"

