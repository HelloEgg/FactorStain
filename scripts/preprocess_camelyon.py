#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from factorstain.data.camelyon17 import build_camelyon_index, tissue_coordinates
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output
from factorstain.utils.preflight import preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m5_external.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    preflight(config, dataset_keys=("camelyon_root",))
    out = prepare_output(config)
    index = build_camelyon_index(config["paths"]["camelyon_root"])
    if index.empty:
        raise RuntimeError(f"No labeled CAMELYON17 training slides (patients 000–099) found under {config['paths']['camelyon_root']}")
    if (index.label < 0).any():
        raise RuntimeError("CAMELYON slide labels could not be verified from lesion masks or metadata; refusing to assume labels")
    index.to_parquet(out / "camelyon_index.parquet", index=False)
    verification = index.groupby("center").agg(patients=("patient_id", "nunique"), slides=("slide_id", "size"), positives=("label", "sum")).reset_index()
    verification["mapping_method"] = "official patient blocks: 000-019 ... 080-099"
    verification["verified_patient_range"] = verification.center.map(lambda c: f"{c*20:03d}-{c*20+19:03d}")
    verification.to_csv(out / "camelyon_center_mapping_verification.csv", index=False)
    coordinate_dir = out / "camelyon_coordinates"; coordinate_dir.mkdir(parents=True, exist_ok=True)
    slides = index.groupby("center", group_keys=False).head(max(1, config["fast_dev_slides"] // 5)) if config["fast_dev_run"] else index
    for _, slide in slides.iterrows():
        destination = coordinate_dir / f"{slide.slide_id}.parquet"
        if destination.exists():
            continue
        coordinates = tissue_coordinates(slide.slide_path, config["tile_size"], config["target_mpp"], config["tissue_threshold"], config["tiles_per_slide"], config["seed"])
        temporary = destination.with_suffix(".parquet.tmp")
        coordinates.to_parquet(temporary, index=False); temporary.replace(destination)
        print(f"{slide.slide_id}: {len(coordinates)} tissue tiles")


if __name__ == "__main__":
    main()
