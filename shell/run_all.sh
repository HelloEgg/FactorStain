#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export OUTPUTS_ROOT="${OUTPUTS_ROOT:-$ROOT/outputs}"

milestones=(m0_probe m1_factorial m2_renderer m3_attribution m4_adapter m5_external)
scripts=(m0_probe.sh m1_factorial.sh m2_renderer.sh m3_attribution.sh m4_adapter.sh m5_external.sh)

for i in "${!scripts[@]}"; do
  bash "shell/${scripts[$i]}"
  decision="$OUTPUTS_ROOT/${milestones[$i]}/GO_NOGO.json"
  if [[ ! -f "$decision" ]]; then
    echo "Milestone ${milestones[$i]} did not create $decision; stopping."
    exit 2
  fi
  python_bin="python"
  [[ -x .venv/bin/python ]] && python_bin=.venv/bin/python
  status="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$decision")"
  echo "${milestones[$i]}: $status"
  if [[ "$status" == "NO_GO" && "${FORCE_CONTINUE:-0}" != "1" ]]; then
    echo "Stopping after measured NO_GO. To run later diagnostic stages anyway: FORCE_CONTINUE=1 bash shell/run_all.sh"
    exit 0
  fi
done

