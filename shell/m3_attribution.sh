#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
acquire_lock m3_attribution
start_log m3_attribution
prior_allows_progress m2_renderer

"$PYTHON_BIN" scripts/run_attribution.py --config configs/m3_attribution.yaml

