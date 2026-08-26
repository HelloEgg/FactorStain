#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BOOTSTRAP="${PYTHON_BIN:-python3}"

if [[ ! -d .venv ]]; then
  "$PYTHON_BOOTSTRAP" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .

.venv/bin/python - <<'PY'
import torch
print(f"FactorStain environment ready: torch={torch.__version__}, CUDA={torch.version.cuda}")
if not torch.cuda.is_available():
    print("WARNING: CUDA is not visible in this shell. Neural milestones will stop with an actionable preflight error.")
PY

echo "Setup complete. Milestone scripts automatically use $ROOT/.venv."

