#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

METHODS="${METHODS:-stainnet,staingan,pix2pix,cyclegan,sastaindiff,cagan,histaugan}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "External neural baseline setup (dry run)"
  echo "Methods: $METHODS"
  echo "Would fetch/verify pinned source repositories and inspect the active FactorStain environment."
  echo "No datasets, checkpoints, packages, or model weights are read or created."
  exit 0
fi

if [[ "${SKIP_PYTHON_SETUP:-0}" != "1" ]]; then
  bash shell/setup.sh
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "SKIP_PYTHON_SETUP=1 requires an existing $ROOT/.venv" >&2
  exit 2
fi

"$PYTHON_BIN" third_party/fetch_baselines.py --methods "$METHODS"
"$PYTHON_BIN" scripts/check_external_baselines.py --methods "$METHODS"

if [[ ",$METHODS," == *",cagan,"* && "${SKIP_CAGAN_VGG_CACHE:-0}" != "1" ]]; then
  "$PYTHON_BIN" -c 'from torchvision.models import VGG16_Weights, vgg16; vgg16(weights=VGG16_Weights.IMAGENET1K_V1); print("CAGAN ImageNet VGG16 perceptual weights cached")'
fi

echo "External neural baseline sources and adapter runtime are ready."
echo "Run a one-seed smoke benchmark with:"
echo "FAST_DEV_RUN=1 SOTA_SEEDS=42 METHODS=$METHODS bash shell/m1_sota_benchmark.sh"
