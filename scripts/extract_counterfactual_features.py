#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from factorstain.models.fm_registry import BlockedModelAccess, encode_batch, load_foundation_model, preprocess_tensor_batch
from factorstain.training.renderer import _load_image, build_renderer
from factorstain.utils.config import load_config
from factorstain.utils.runtime import barrier, init_distributed, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m4_adapter.yaml")
    parser.add_argument("--model", required=True, choices=["uni", "virchow2"])
    args = parser.parse_args()
    config = load_config(args.config)
    device, rank, _, world_size = init_distributed(require_cuda=True)
    seed_everything(config["seed"] + rank)
    outputs = Path(config["paths"]["outputs_root"])
    renderer_out = outputs / "m2_renderer"
    if not (renderer_out / "checkpoints" / "factorstain" / "best.pt").exists():
        renderer_out = outputs / "m1_factorial"
    index = pd.read_parquet(renderer_out / "plism_index_with_splits.parquet")
    cache_dir = outputs / "m4_adapter" / "counterfactual_features" / args.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / "features.h5"
    if final.exists():
        with h5py.File(final, "r") as handle:
            cache_is_dev = bool(handle.attrs.get("fast_dev_run", False))
        if config["fast_dev_run"] or not cache_is_dev:
            if rank == 0:
                print(f"Reusing counterfactual cache {final}")
            return
        if rank == 0:
            print(f"Refreshing development-only counterfactual cache for a full run: {final}")
    try:
        foundation = load_foundation_model(args.model, device)
    except BlockedModelAccess as exc:
        if rank == 0:
            (cache_dir / "status.json").write_text(json.dumps({"status": "BLOCKED_MODEL_ACCESS", "message": str(exc)}, indent=2), encoding="utf-8")
        return
    stains, scanners = sorted(index.stain_id.unique()), sorted(index.scanner_id.unique())
    renderer = build_renderer("factorstain", len(stains), len(scanners)).to(device).eval()
    renderer_state = torch.load(renderer_out / "checkpoints" / "factorstain" / "best.pt", map_location=device, weights_only=False)
    renderer.load_state_dict(renderer_state["model"])
    groups = list(index[index.image_exists].groupby("aligned_group_id"))
    max_groups = 12 if config["fast_dev_run"] else config.get("counterfactual_groups", 5000)
    if config["fast_dev_run"]:
        selected_groups = []
        for split_name in ("train", "val", "test"):
            selected_groups.extend([(group_id, group) for group_id, group in groups if str(group.morphology_split.iloc[0]) == split_name][:max(2, max_groups // 3)])
        groups = selected_groups
    else:
        groups = groups[:max_groups]
    groups = groups[rank::world_size]
    rng = np.random.default_rng(config["seed"] + rank)
    combinations_per_group = 4 if config["fast_dev_run"] else config.get("counterfactuals_per_group", 12)
    records, feature_chunks = [], []
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    all_combinations = [(i, j) for i in range(len(stains)) for j in range(len(scanners))]
    for group_id, group in groups:
        source_row = group.iloc[0]
        selected = rng.choice(len(all_combinations), min(combinations_per_group, len(all_combinations)), replace=False)
        targets = [all_combinations[int(position)] for position in selected]
        source = _load_image(source_row.image_path, 512).unsqueeze(0).to(device).expand(len(targets), -1, -1, -1)
        stain_ids = torch.tensor([value[0] for value in targets], device=device)
        scanner_ids = torch.tensor([value[1] for value in targets], device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
            generated = renderer(source, stain_ids, scanner_ids)
            features = encode_batch(foundation, preprocess_tensor_batch(foundation, generated)).float().cpu().numpy()
        feature_chunks.append(features)
        for stain_idx, scanner_idx in targets:
            records.append(
                {
                    "aligned_group_id": str(group_id),
                    "tissue_type": str(source_row.tissue_type),
                    "stain_id": str(stains[stain_idx]),
                    "scanner_id": str(scanners[scanner_idx]),
                    "morphology_split": str(source_row.morphology_split),
                    "source_image_id": str(source_row.image_id),
                }
            )
    np.savez_compressed(cache_dir / f"rank{rank}.npz", metadata=np.asarray(records, dtype=object), features=np.concatenate(feature_chunks) if feature_chunks else np.empty((0, foundation.feature_dim), np.float32))
    barrier()
    if rank == 0:
        all_records, all_features = [], []
        for part in range(world_size):
            payload = np.load(cache_dir / f"rank{part}.npz", allow_pickle=True)
            all_records.extend(payload["metadata"].tolist())
            all_features.append(payload["features"])
        metadata = pd.DataFrame(all_records)
        features = np.concatenate(all_features)
        temporary = final.with_suffix(".h5.tmp")
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("features", data=features, compression="gzip", chunks=True)
            for column in metadata:
                handle.create_dataset(column, data=metadata[column].astype(str).to_numpy(dtype=object), dtype=h5py.string_dtype("utf-8"))
            handle.attrs["foundation_model"] = args.model
            handle.attrs["renderer_checkpoint"] = str(renderer_out / "checkpoints" / "factorstain" / "best.pt")
            handle.attrs["fast_dev_run"] = bool(config["fast_dev_run"])
        os.replace(temporary, final)
        (cache_dir / "status.json").write_text(json.dumps({"status": "AVAILABLE", "n": len(metadata), "feature_dim": features.shape[1]}, indent=2), encoding="utf-8")
    barrier()


if __name__ == "__main__":
    main()
