#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "$0")/common.sh"

MILESTONE="m_minus1_domain_audit"
CONFIG="configs/m_minus1_domain_audit.yaml"
acquire_lock "$MILESTONE"
start_log "$MILESTONE"

echo "Preparing deterministic PLISM and MIDOG21 sample manifests and raw-domain figures"
"$PYTHON_BIN" scripts/prepare_domain_audit.py --config "$CONFIG"

echo "Extracting/reusing frozen official DINOv3 features for PLISM on ${NPROC_PER_NODE} GPU process(es)"
run_ddp scripts/extract_dinov3_audit_features.py --config "$CONFIG" --dataset plism

echo "Extracting/reusing frozen official DINOv3 features for MIDOG21 on ${NPROC_PER_NODE} GPU process(es)"
run_ddp scripts/extract_dinov3_audit_features.py --config "$CONFIG" --dataset midog21

echo "Running group-held-out probes, controlled distances, PCA/UMAP, and dashboards"
"$PYTHON_BIN" scripts/run_domain_audit.py --config "$CONFIG"
