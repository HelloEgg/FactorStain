#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler

from factorstain.data.dinov3_audit import DINOAuditDataset, ProcessorCollator
from factorstain.models.dinov3 import (
    load_dinov3,
    load_feature_cache,
    manifest_signature,
    processor_description,
    save_feature_cache,
    select_image_embedding,
)
from factorstain.utils.domain_audit import (
    load_domain_audit_config,
    prepare_domain_audit_output,
)
from factorstain.utils.runtime import (
    barrier,
    init_distributed,
    is_main_process,
    seed_everything,
)


def _metadata_path(out: Path, dataset_name: str) -> Path:
    return (
        out
        / "metadata"
        / (
            "plism_samples.parquet"
            if dataset_name == "plism"
            else "midog_samples.parquet"
        )
    )


def _feature_path(out: Path, dataset_name: str) -> Path:
    return (
        out
        / "features"
        / ("plism_dinov3.npz" if dataset_name == "plism" else "midog21_dinov3.npz")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m_minus1_domain_audit.yaml")
    parser.add_argument("--dataset", required=True, choices=["plism", "midog21"])
    args = parser.parse_args()
    config = load_domain_audit_config(args.config)
    out = prepare_domain_audit_output(config)
    device, rank, _, world_size = init_distributed(require_cuda=True)
    seed_everything(config["seed"] + rank)
    metadata = pd.read_parquet(_metadata_path(out, args.dataset))
    if args.dataset == "plism" and "feature_selected" in metadata:
        metadata = metadata[metadata.feature_selected].copy()
    metadata = metadata.sort_values("sample_id").reset_index(drop=True)
    model_name = config["dinov3"]["model_name"]
    signature = manifest_signature(metadata, model_name)
    final_cache = _feature_path(out, args.dataset)
    if final_cache.exists() and not config["force_reextract"]:
        _, cached_ids, cached_metadata = load_feature_cache(final_cache)
        if cached_metadata.get("manifest_signature") == signature and set(
            cached_ids
        ) == set(metadata.sample_id.astype(str)):
            if is_main_process():
                print(f"Reusing validated DINOv3 cache: {final_cache}")
            barrier()
            return
        if is_main_process():
            print(
                f"Existing cache signature differs; re-extracting {args.dataset} DINOv3 features"
            )

    processor, model, model_dtype = load_dinov3(
        model_name, device, config["dinov3"]["dtype"]
    )
    dataset = DINOAuditDataset(metadata, args.dataset)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    loader = DataLoader(
        dataset,
        batch_size=config["dinov3"]["batch_size"],
        sampler=sampler,
        num_workers=config["dinov3"]["num_workers"],
        pin_memory=True,
        collate_fn=ProcessorCollator(processor),
        persistent_workers=config["dinov3"]["num_workers"] > 0,
    )
    ids, chunks, pooling_strategy = [], [], None
    autocast_enabled = device.type == "cuda" and model_dtype != torch.float32
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=model_dtype, enabled=autocast_enabled
            ):
                outputs = model(pixel_values=pixel_values)
                embeddings, pooling = select_image_embedding(outputs, model.config)
            if pooling_strategy is None:
                pooling_strategy = pooling
            elif pooling != pooling_strategy:
                raise RuntimeError(
                    f"DINOv3 pooling strategy changed within one extraction: {pooling_strategy} -> {pooling}"
                )
            ids.extend(batch["sample_id"])
            chunks.append(embeddings.float().cpu().numpy())
    if not chunks:
        raise RuntimeError(f"No DINOv3 batches were produced for {args.dataset}")
    part_dir = out / "features" / ".parts" / args.dataset
    part_dir.mkdir(parents=True, exist_ok=True)
    save_feature_cache(
        part_dir / f"rank{rank}.npz",
        np.concatenate(chunks),
        np.asarray(ids),
        {
            "rank": rank,
            "pooling_strategy": pooling_strategy,
            "manifest_signature": signature,
        },
    )
    barrier()
    if is_main_process():
        all_features, all_ids = [], []
        for part in range(world_size):
            features, sample_ids, part_metadata = load_feature_cache(
                part_dir / f"rank{part}.npz"
            )
            if part_metadata["pooling_strategy"] != pooling_strategy:
                raise RuntimeError("DINOv3 ranks disagreed on pooling strategy")
            all_features.append(features)
            all_ids.extend(sample_ids.tolist())
        stacked = np.concatenate(all_features)
        positions = pd.DataFrame(
            {"sample_id": all_ids, "position": np.arange(len(all_ids))}
        ).drop_duplicates("sample_id")
        ordered = pd.DataFrame({"sample_id": metadata.sample_id.astype(str)}).merge(
            positions, on="sample_id", how="left", validate="one_to_one"
        )
        if ordered.position.isna().any():
            missing = ordered[ordered.position.isna()].sample_id.head().tolist()
            raise RuntimeError(
                f"Distributed DINOv3 extraction missed manifest samples: {missing}"
            )
        ordered_features = stacked[ordered.position.astype(int).to_numpy()]
        model_details = processor_description(
            processor, model, pooling_strategy, ordered_features.shape[1]
        )
        cache_metadata = {
            **model_details,
            "manifest_signature": signature,
            "dataset": args.dataset,
            "n_samples": len(metadata),
            "requested_dtype": config["dinov3"]["dtype"],
            "resolved_dtype": str(model_dtype).replace("torch.", ""),
            "frozen": True,
            "inference_mode": True,
        }
        save_feature_cache(
            final_cache,
            ordered_features,
            metadata.sample_id.astype(str).to_numpy(),
            cache_metadata,
        )
        (out / "features" / f"{args.dataset}_dinov3_metadata.json").write_text(
            json.dumps(cache_metadata, indent=2), encoding="utf-8"
        )
        print(
            f"Saved {len(metadata):,} {args.dataset} DINOv3 embeddings ({ordered_features.shape[1]}D) to {final_cache}"
        )
    barrier()


if __name__ == "__main__":
    main()
