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
from torch.utils.data import DataLoader, DistributedSampler

from factorstain.data.plism import PLISMDataset
from factorstain.models.fm_registry import BlockedModelAccess, encode_batch, load_foundation_model
from factorstain.utils.config import load_config
from factorstain.utils.runtime import barrier, distributed_context, init_distributed


def _cache_complete(path: Path, expected_ids: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return set(value.decode() if isinstance(value, bytes) else str(value) for value in handle["image_id"][:]) >= expected_ids
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m0_probe.yaml")
    parser.add_argument("--model", required=True, choices=["uni", "virchow2"])
    args = parser.parse_args()
    config = load_config(args.config)
    device, rank, _, world_size = init_distributed(require_cuda=True)
    out = Path(config["paths"]["outputs_root"]) / config["milestone"]
    index = pd.read_parquet(out / "plism_index.parquet")
    existing = index[index.image_exists].copy()
    limit = config["fast_dev_max_images"] if config["fast_dev_run"] else config["max_probe_images"]
    if len(existing) > limit:
        # Group-aware stable subset retains multiple acquisitions of each selected morphology.
        groups = existing.aligned_group_id.drop_duplicates().sample(frac=1, random_state=config["seed"])
        chosen = []
        total = 0
        for group in groups:
            positions = existing.index[existing.aligned_group_id == group].tolist()
            chosen.extend(positions)
            total += len(positions)
            if total >= limit:
                break
        existing = existing.loc[chosen[:limit]].reset_index(drop=True)
    cache_dir = out / "features" / args.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    final_cache = cache_dir / "features.h5"
    if _cache_complete(final_cache, set(existing.image_id)):
        if rank == 0:
            print(f"Reusing complete cache {final_cache}")
        return
    try:
        foundation = load_foundation_model(args.model, device)
    except BlockedModelAccess as exc:
        if rank == 0:
            (cache_dir / "status.json").write_text(json.dumps({"status": "BLOCKED_MODEL_ACCESS", "message": str(exc)}, indent=2), encoding="utf-8")
            print(str(exc))
        return
    dataset = PLISMDataset(existing, transform=foundation.transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"], pin_memory=True)
    all_ids, all_features = [], []
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=autocast_dtype):
            features = encode_batch(foundation, images)
        all_ids.extend(batch["image_id"])
        all_features.append(features.cpu().numpy())
    np.savez_compressed(cache_dir / f"rank{rank}.npz", image_id=np.asarray(all_ids), features=np.concatenate(all_features))
    barrier()
    if rank == 0:
        ids, features = [], []
        for part in range(world_size):
            payload = np.load(cache_dir / f"rank{part}.npz")
            ids.extend(payload["image_id"].tolist())
            features.append(payload["features"])
        combined = pd.DataFrame({"image_id": ids, "position": np.arange(len(ids))}).drop_duplicates("image_id")
        stacked = np.concatenate(features)
        temporary = final_cache.with_suffix(".h5.tmp")
        with h5py.File(temporary, "w") as handle:
            string_type = h5py.string_dtype("utf-8")
            handle.create_dataset("image_id", data=combined.image_id.to_numpy(dtype=object), dtype=string_type)
            handle.create_dataset("features", data=stacked[combined.position.to_numpy()], compression="gzip", chunks=True)
            handle.attrs["model"] = args.model
            handle.attrs["feature_dim"] = foundation.feature_dim
        os.replace(temporary, final_cache)
        (cache_dir / "status.json").write_text(json.dumps({"status": "AVAILABLE", "n": len(combined), "feature_dim": foundation.feature_dim, "parameter_count": sum(parameter.numel() for parameter in foundation.model.parameters())}, indent=2), encoding="utf-8")
    barrier()


if __name__ == "__main__":
    main()
