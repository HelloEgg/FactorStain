#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from factorstain.data.domain_audit import (
    build_midog_case_index,
    sample_midog_for_audit,
    sample_plism_for_audit,
)
from factorstain.data.plism import build_plism_index, discover_plism_metadata
from factorstain.utils.domain_audit import (
    load_domain_audit_config,
    prepare_domain_audit_output,
)
from factorstain.utils.preflight import preflight
from factorstain.visualization.domain_audit import (
    pixel_statistics,
    plot_color_statistics,
    plot_domain_grid,
    plot_plism_controlled,
    plot_sharpness,
    select_domain_examples,
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _expand_midog_fov(selection: pd.DataFrame) -> pd.DataFrame:
    expanded = selection.copy()
    for index, row in expanded.iterrows():
        original = int(row.read_size)
        size = min(original * 2, int(row.slide_width), int(row.slide_height))
        expanded.loc[index, "x"] = max(
            0, min(int(row.x) - original // 2, int(row.slide_width) - size)
        )
        expanded.loc[index, "y"] = max(
            0, min(int(row.y) - original // 2, int(row.slide_height) - size)
        )
        expanded.loc[index, "read_size"] = size
        expanded.loc[index, "large_visual_crop"] = True
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m_minus1_domain_audit.yaml")
    args = parser.parse_args()
    config = load_domain_audit_config(args.config)
    preflight(config, dataset_keys=("plism_root", "midog_root"), require_cuda=False)
    out = prepare_domain_audit_output(config)
    metadata_dir, figures = out / "metadata", out / "figures"
    plism_path, midog_path = (
        metadata_dir / "plism_samples.parquet",
        metadata_dir / "midog_samples.parquet",
    )
    controlled_path = metadata_dir / "plism_controlled_comparisons.parquet"
    quality_path = metadata_dir / "sampling_quality.csv"
    if config["resample"] or not (
        plism_path.exists() and midog_path.exists() and controlled_path.exists()
    ):
        metadata_source = discover_plism_metadata(config["paths"]["plism_root"])
        plism_index = build_plism_index(config["paths"]["plism_root"], metadata_source)
        plism_samples, controlled, plism_quality = sample_plism_for_audit(
            plism_index, config, config["seed"]
        )
        midog_cases, verification = build_midog_case_index(
            config["paths"]["midog_root"]
        )
        midog_samples, midog_quality = sample_midog_for_audit(
            midog_cases, config, config["seed"]
        )
        _atomic_parquet(plism_samples, plism_path)
        _atomic_parquet(controlled, controlled_path)
        _atomic_parquet(midog_samples, midog_path)
        _atomic_parquet(plism_index, metadata_dir / "plism_full_index.parquet")
        _atomic_parquet(midog_cases, metadata_dir / "midog_case_index.parquet")
        pd.concat([plism_quality, midog_quality], ignore_index=True, sort=False).to_csv(
            quality_path, index=False
        )
        (metadata_dir / "dataset_structure.json").write_text(
            json.dumps(
                {
                    "plism_metadata": str(metadata_source),
                    "plism_counts": {
                        "images": len(plism_index),
                        "groups": plism_index.aligned_group_id.nunique(),
                        "stains": plism_index.stain_id.nunique(),
                        "scanners": plism_index.scanner_id.nunique(),
                        "tissues": plism_index.tissue_type.nunique(),
                    },
                    "midog_verification": verification,
                    "midog_cases": len(midog_cases),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        print(
            "Reusing deterministic M-1 sample manifests; set RESAMPLE=1 to select new examples"
        )
    plism_samples, controlled, midog_samples = (
        pd.read_parquet(plism_path),
        pd.read_parquet(controlled_path),
        pd.read_parquet(midog_path),
    )
    example_count = config["plism"]["random_examples_per_domain"]
    plism_stain_examples = select_domain_examples(
        plism_samples, "stain_id", example_count, config["seed"]
    )
    plism_scanner_examples = select_domain_examples(
        plism_samples, "scanner_id", example_count, config["seed"] + 100
    )
    midog_examples = select_domain_examples(
        midog_samples, "scanner_id", 8, config["seed"]
    )
    midog_large = _expand_midog_fov(
        select_domain_examples(midog_samples, "scanner_id", 4, config["seed"] + 100)
    )
    visual_samples = pd.concat(
        [
            plism_stain_examples.assign(visual_name="plism_random_by_stain"),
            plism_scanner_examples.assign(visual_name="plism_random_by_scanner"),
            midog_examples.assign(visual_name="midog_random_by_scanner"),
            midog_large.assign(visual_name="midog_large_examples_by_scanner"),
        ],
        ignore_index=True,
        sort=False,
    )
    _atomic_parquet(visual_samples, metadata_dir / "raw_visual_samples.parquet")
    plot_domain_grid(
        plism_stain_examples,
        "stain_id",
        "plism",
        figures / "plism" / "plism_random_by_stain.png",
        "PLISM random tissue patches grouped by stain",
    )
    plot_domain_grid(
        plism_scanner_examples,
        "scanner_id",
        "plism",
        figures / "plism" / "plism_random_by_scanner.png",
        "PLISM random tissue patches grouped by scanner",
    )
    plot_domain_grid(
        midog_examples,
        "scanner_id",
        "midog21",
        figures / "midog21" / "midog_random_by_scanner.png",
        "MIDOG21 tissue patches grouped by scanner domain",
    )
    plot_domain_grid(
        midog_large,
        "scanner_id",
        "midog21",
        figures / "midog21" / "midog_large_examples_by_scanner.png",
        "MIDOG21 larger representative scanner-domain crops",
        display_size=320,
    )
    plot_plism_controlled(controlled, figures / "plism")
    plism_stats_path, midog_stats_path = (
        metadata_dir / "plism_pixel_statistics.parquet",
        metadata_dir / "midog_pixel_statistics.parquet",
    )
    if config["resample"] or not plism_stats_path.exists():
        _atomic_parquet(pixel_statistics(plism_samples, "plism"), plism_stats_path)
    if config["resample"] or not midog_stats_path.exists():
        _atomic_parquet(pixel_statistics(midog_samples, "midog21"), midog_stats_path)
    plism_stats = pd.read_parquet(plism_stats_path)
    plot_color_statistics(
        plism_stats, "stain_id", figures / "plism" / "plism_color_stats_by_stain.png"
    )
    plot_color_statistics(
        plism_stats,
        "scanner_id",
        figures / "plism" / "plism_color_stats_by_scanner.png",
    )
    plot_sharpness(
        plism_stats, "scanner_id", figures / "plism" / "plism_sharpness_by_scanner.png"
    )
    print(
        f"Prepared {len(plism_samples):,} PLISM images and {len(midog_samples):,} MIDOG21 tissue patches for M-1"
    )


if __name__ == "__main__":
    main()
