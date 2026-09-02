#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"

MILESTONE="m_minus1_mmd"
CONFIG="configs/m_minus1_mmd.yaml"
acquire_lock "$MILESTONE"
start_log "$MILESTONE"

echo "Reusing cached PLISM frozen-DINOv3 features; this stage never extracts features."
"$PYTHON_BIN" scripts/run_mmd_analysis.py --config "$CONFIG"
