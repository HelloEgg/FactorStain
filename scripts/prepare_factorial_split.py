#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from factorstain.data.plism import build_plism_index
from factorstain.data.samplers import enumerate_rendering_episodes
from factorstain.data.splits import (
    build_combination_split,
    group_train_val_test_split,
    morphology_split_manifest,
    save_combination_split,
)
from factorstain.models.factorstain import FactorStain
from factorstain.models.joint_baseline import (
    JointConditionalGenerator,
    ParallelFactorGenerator,
)
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output
from factorstain.utils.runtime import atomic_json_dump


def _locate_index(config: dict) -> tuple[pd.DataFrame, Path]:
    explicit = os.getenv("PLISM_INDEX")
    outputs = Path(config["paths"]["outputs_root"])
    candidates = [
        Path(explicit) if explicit else None,
        outputs / "m_minus1_domain_audit" / "metadata" / "plism_samples.parquet",
        outputs / "m_minus1_domain_audit" / "metadata" / "plism_full_index.parquet",
        outputs / "m0_probe" / "plism_index.parquet",
    ]
    source = next((path for path in candidates if path and path.exists()), None)
    if source is not None:
        return pd.read_parquet(source), source
    root = Path(config["paths"]["plism_root"])
    if not root.exists():
        raise FileNotFoundError(
            "No reusable PLISM index was found and the configured PLISM root does not "
            f"exist: {root}. Expected the M-1 manifest under {candidates[1]}."
        )
    return build_plism_index(root), root


def _validate_index(index: pd.DataFrame) -> pd.DataFrame:
    required = {
        "image_path",
        "aligned_group_id",
        "tissue_type",
        "stain_id",
        "scanner_id",
    }
    missing = required - set(index.columns)
    if missing:
        raise ValueError(
            f"PLISM index is missing required M1 columns: {sorted(missing)}"
        )
    frame = index.copy().reset_index(drop=True)
    for column in ("aligned_group_id", "tissue_type", "stain_id", "scanner_id"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"PLISM index contains blank {column} values")
    if "sample_id" not in frame:
        frame["sample_id"] = (
            frame["image_id"].astype(str)
            if "image_id" in frame
            else frame.index.map(lambda value: f"plism-{value:08d}")
        )
    if "image_id" not in frame:
        frame["image_id"] = frame.sample_id.astype(str)
    if "image_exists" not in frame:
        frame["image_exists"] = frame.image_path.map(
            lambda value: Path(value).is_file()
        )
    if not frame.image_exists.any():
        raise RuntimeError("The PLISM index contains no readable image paths")
    duplicates = frame.duplicated(
        ["aligned_group_id", "stain_id", "scanner_id"], keep=False
    )
    if duplicates.any():
        examples = frame.loc[
            duplicates, ["aligned_group_id", "stain_id", "scanner_id"]
        ].head()
        raise ValueError(
            "M1 requires one image per aligned acquisition cell; duplicate examples: "
            f"{examples.to_dict('records')}"
        )
    return frame


def _cell_set(items: list[dict]) -> set[tuple[str, str]]:
    return {(str(item["stain_id"]), str(item["scanner_id"])) for item in items}


def _model_counts(stains: int, scanners: int, width: int) -> dict[str, int]:
    models = {
        "joint": JointConditionalGenerator(stains, scanners, width=width),
        "parallel": ParallelFactorGenerator(stains, scanners, width=width),
        "factorstain": FactorStain(stains, scanners, width=width),
    }
    return {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }


def _plot_split(split: dict, destination: Path) -> None:
    stains, scanners = split["stains"], split["scanners"]
    train = _cell_set(split["train_cells"])
    validation = _cell_set(split["validation_cells"])
    test = _cell_set(split["test_cells"])
    matrix = np.full((len(stains), len(scanners)), -1, dtype=int)
    for row, stain in enumerate(stains):
        for column, scanner in enumerate(scanners):
            cell = (stain, scanner)
            if cell in train:
                matrix[row, column] = 0
            elif cell in validation:
                matrix[row, column] = 1
            elif cell in test:
                matrix[row, column] = 2
    colors = ListedColormap(["#adb5bd", "#2b8a3e", "#f08c00", "#c92a2a"])
    fig, axis = plt.subplots(
        figsize=(max(9, len(scanners) * 1.25), max(8, len(stains) * 0.72))
    )
    axis.imshow(matrix + 1, cmap=colors, vmin=0, vmax=3, aspect="auto")
    axis.set_xticks(range(len(scanners)), scanners, rotation=35, ha="right")
    axis.set_yticks(range(len(stains)), stains)
    axis.set_xlabel("Scanner ID")
    axis.set_ylabel("Stain ID")
    axis.set_title(
        f"Global stain × scanner split — seed {split['seed']}\n"
        "green=train, orange=validation, red=held-out test, gray=unobserved"
    )
    labels = {-1: "N/A", 0: "TRAIN", 1: "VAL", 2: "TEST"}
    for row in range(len(stains)):
        for column in range(len(scanners)):
            axis.text(
                column,
                row,
                labels[matrix[row, column]],
                color="white" if matrix[row, column] >= 0 else "#343a40",
                ha="center",
                va="center",
                fontsize=7,
                weight="bold",
            )
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_factorial.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    index, source = _locate_index(config)
    index = _validate_index(index)
    ratios = tuple(config.get("morphology_split", [0.70, 0.15, 0.15]))
    morphology_labels = group_train_val_test_split(
        index, ratios=ratios, seed=config["seed"]
    )
    index["morphology_split"] = morphology_labels
    split = build_combination_split(
        index,
        config.get("holdout_fraction", 0.18),
        config["seed"],
        config.get("validation_fraction", 0.10),
    )
    split_path = out / "splits" / f"combination_split_seed{config['seed']}.json"
    save_combination_split(split, split_path)
    morphology_manifest = morphology_split_manifest(
        index, morphology_labels, ratios, config["seed"]
    )
    atomic_json_dump(
        morphology_manifest,
        out / "splits" / f"morphology_split_seed{config['seed']}.json",
    )
    index.to_parquet(out / "metadata" / "plism_index_with_splits.parquet", index=False)
    # Backward compatibility for existing M2 scripts.
    index.to_parquet(out / "plism_index_with_splits.parquet", index=False)

    train_cells = _cell_set(split["train_cells"])
    validation_cells = _cell_set(split["validation_cells"])
    episodes = enumerate_rendering_episodes(
        index,
        train_cells,
        morphology_split="train",
        max_per_group=config.get("episodes_per_group", 64),
        seed=config["seed"],
    )
    validation_episodes = enumerate_rendering_episodes(
        index,
        train_cells,
        validation_cells,
        morphology_split="val",
        max_per_group=config.get("validation_episodes_per_group", 16),
        seed=config["seed"] + 1,
    )
    episode_types = pd.Series(
        [episode.pair_type for episode in episodes]
    ).value_counts()
    train_rows = index[
        index.morphology_split.eq("train")
        & index.apply(
            lambda row: (str(row.stain_id), str(row.scanner_id)) in train_cells,
            axis=1,
        )
    ]
    heldout = _cell_set(split["test_cells"])
    heldout_rows = index[
        index.apply(
            lambda row: (str(row.stain_id), str(row.scanner_id)) in heldout, axis=1
        )
    ]
    statistics = {
        "metadata_source": str(source),
        "images": len(index),
        "readable_images": int(index.image_exists.sum()),
        "usable_groups": len(
            {index.loc[item.target, "aligned_group_id"] for item in episodes}
        ),
        "source_target_episodes": len(episodes),
        "validation_episodes": len(validation_episodes),
        "episode_types": episode_types.to_dict(),
        "exact_scanner_paired_sample_count": int(episode_types.get("scanner_pair", 0)),
        "cross_stain_sample_count": int(
            episode_types.get("cross_stain", 0) + episode_types.get("factorial", 0)
        ),
        "stain_frequencies": train_rows.stain_id.value_counts().sort_index().to_dict(),
        "scanner_frequencies": train_rows.scanner_id.value_counts()
        .sort_index()
        .to_dict(),
        "heldout_combination_coverage": (
            heldout_rows.groupby(["stain_id", "scanner_id"])
            .size()
            .rename("samples")
            .reset_index()
            .to_dict("records")
        ),
        "combination_counts": {
            "train": len(split["train_cells"]),
            "validation": len(split["validation_cells"]),
            "test": len(split["test_cells"]),
        },
        "morphology_group_counts": morphology_manifest["group_counts"],
        "model_parameter_counts": _model_counts(
            len(split["stains"]),
            len(split["scanners"]),
            config.get("model_width", 64),
        ),
        "serial_section_policy": (
            "Only same-stain scanner pairs receive strong pixel supervision; "
            "all stain-changing episodes use distributional/structural losses."
        ),
    }
    atomic_json_dump(statistics, out / "metadata" / "factorial_episode_statistics.json")
    _plot_split(split, out / "figures" / "heldout_combination_matrix.png")
    print(f"Validated PLISM index: {source}")
    print(
        f"Cells: train={len(split['train_cells'])}, validation={len(split['validation_cells'])}, "
        f"held-out test={len(split['test_cells'])}; training episodes={len(episodes):,}"
    )
    print("Held-out cells:")
    for cell in split["test_cells"]:
        print(f"  {cell['stain_id']} × {cell['scanner_id']}")


if __name__ == "__main__":
    main()
