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
from PIL import Image, ImageOps

from factorstain.data.midog import parse_midog_annotations
from factorstain.models.fm_registry import BlockedModelAccess, encode_batch, load_foundation_model
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output
from factorstain.utils.runtime import barrier, init_distributed


def _write_bag(path: Path, features: np.ndarray, coordinates: np.ndarray, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("features", data=features, compression="gzip", chunks=True)
        handle.create_dataset("coordinates", data=coordinates)
        for key, value in attrs.items():
            handle.attrs[key] = value
    os.replace(temporary, path)


def _camelyon(config: dict, model_name: str, foundation, out: Path, device: torch.device, rank: int, world_size: int) -> None:
    import tiffslide

    index = pd.read_parquet(out / "camelyon_index.parquet")
    if config["fast_dev_run"]:
        index = index.groupby("center", group_keys=False).head(max(1, config["fast_dev_slides"] // 5))
    bag_dir = out / "external_features" / "camelyon" / model_name
    for row_number, (_, slide) in enumerate(index.iterrows()):
        if row_number % world_size != rank:
            continue
        destination = bag_dir / f"{slide.slide_id}.h5"
        if destination.exists():
            continue
        coords_path = out / "camelyon_coordinates" / f"{slide.slide_id}.parquet"
        coordinates = pd.read_parquet(coords_path)
        feature_chunks = []
        with tiffslide.TiffSlide(slide.slide_path) as wsi:
            for start in range(0, len(coordinates), config["batch_size"]):
                subset = coordinates.iloc[start : start + config["batch_size"]]
                images = []
                for _, coordinate in subset.iterrows():
                    read_size = int(coordinate.get("read_size", config["tile_size"]))
                    image = wsi.read_region((int(coordinate.x), int(coordinate.y)), 0, (read_size, read_size)).convert("RGB")
                    images.append(foundation.transform(image))
                batch = torch.stack(images).to(device)
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    feature_chunks.append(encode_batch(foundation, batch).cpu().numpy())
        _write_bag(destination, np.concatenate(feature_chunks), coordinates[["x", "y"]].to_numpy(), {"slide_id": slide.slide_id, "patient_id": int(slide.patient_id), "center": int(slide.center), "label": int(slide.label), "model": model_name})


def _crop_patch(image: Image.Image, x: float, y: float, size: int) -> Image.Image:
    half = size // 2
    return ImageOps.pad(image.crop((round(x) - half, round(y) - half, round(x) + half, round(y) + half)).convert("RGB"), (size, size), color="white")


def _midog(config: dict, model_name: str, foundation, out: Path, device: torch.device, rank: int, world_size: int) -> None:
    annotations = parse_midog_annotations(config["paths"]["midog_root"])
    if annotations.empty:
        raise RuntimeError("MIDOG.json contains no annotations")
    cases = list(annotations.groupby("case_id"))
    if config["fast_dev_run"]:
        cases = cases[:8]
    bag_dir = out / "external_features" / "midog" / model_name
    rng = np.random.default_rng(config["seed"] + rank)
    for case_number, (case_id, group) in enumerate(cases):
        if case_number % world_size != rank:
            continue
        destination = bag_dir / f"{case_id}.h5"
        if destination.exists():
            continue
        with Image.open(group.image_path.iloc[0]) as opened:
            image = opened.convert("RGB")
            records = group[["x", "y", "label"]].copy()
            records["negative_type"] = np.where(records.label.eq(1), "mitosis", "annotated_hard_negative")
            positive = records[records.label.eq(1)]
            # Mine cell-like hard negatives by edge variance, plus low-texture background negatives.
            candidates = []
            attempts = 0
            desired = max(1, len(positive))
            while len(candidates) < desired * 8 and attempts < max(200, len(positive) * 100):
                x = rng.integers(config["midog_patch_size"] // 2, max(config["midog_patch_size"] // 2 + 1, image.width - config["midog_patch_size"] // 2))
                y = rng.integers(config["midog_patch_size"] // 2, max(config["midog_patch_size"] // 2 + 1, image.height - config["midog_patch_size"] // 2))
                distance = np.sqrt((positive.x.to_numpy() - x) ** 2 + (positive.y.to_numpy() - y) ** 2)
                if not len(distance) or distance.min() > config["hard_negative_radius"]:
                    patch = np.asarray(_crop_patch(image, x, y, config["midog_patch_size"]).convert("L"), dtype=np.float32)
                    edge_score = float(np.var(np.diff(patch, axis=0)) + np.var(np.diff(patch, axis=1)))
                    candidates.append({"x": x, "y": y, "label": 0, "edge_score": edge_score})
                attempts += 1
            candidate_frame = pd.DataFrame(candidates).sort_values("edge_score")
            background = candidate_frame.head(desired).assign(negative_type="background_negative")
            hard = candidate_frame.tail(desired).assign(negative_type="mined_hard_negative")
            records = pd.concat([records, background, hard], ignore_index=True)
            chunks = []
            for start in range(0, len(records), config["batch_size"]):
                subset = records.iloc[start : start + config["batch_size"]]
                batch = torch.stack([foundation.transform(_crop_patch(image, row.x, row.y, config["midog_patch_size"])) for _, row in subset.iterrows()]).to(device)
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                    chunks.append(encode_batch(foundation, batch).cpu().numpy())
        _write_bag(destination, np.concatenate(chunks), records[["x", "y"]].to_numpy(), {"case_id": str(case_id), "scanner_id": str(group.scanner_id.iloc[0]), "labels_json": json.dumps(records.label.astype(int).tolist()), "model": model_name})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m5_external.yaml")
    parser.add_argument("--model", required=True, choices=["uni", "virchow2"])
    parser.add_argument("--dataset", required=True, choices=["camelyon", "midog"])
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    device, rank, _, world_size = init_distributed(require_cuda=True)
    try:
        foundation = load_foundation_model(args.model, device)
    except BlockedModelAccess as exc:
        if rank == 0:
            status = out / "external_features" / args.dataset / args.model / "status.json"; status.parent.mkdir(parents=True, exist_ok=True); status.write_text(json.dumps({"status": "BLOCKED_MODEL_ACCESS", "message": str(exc)}, indent=2), encoding="utf-8")
        return
    if args.dataset == "camelyon":
        _camelyon(config, args.model, foundation, out, device, rank, world_size)
    else:
        _midog(config, args.model, foundation, out, device, rank, world_size)
    barrier()


if __name__ == "__main__":
    main()
