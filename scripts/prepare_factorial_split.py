#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factorstain.data.splits import build_combination_split, group_train_val_test_split, save_combination_split
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output
from factorstain.utils.preflight import preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_factorial.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    preflight(config, dataset_keys=("plism_root",), required_milestone="m0_probe")
    out = prepare_output(config)
    source_index = Path(config["paths"]["outputs_root"]) / "m0_probe" / "plism_index.parquet"
    index = pd.read_parquet(source_index)
    index["morphology_split"] = group_train_val_test_split(index, ratios=tuple(config.get("morphology_split", [0.70, 0.15, 0.15])), seed=config["seed"])
    index.to_parquet(out / "plism_index_with_splits.parquet", index=False)
    split = build_combination_split(index, config.get("holdout_fraction", 0.18), config["seed"])
    split_path = out / "splits" / f"combination_split_seed{config['seed']}.json"
    save_combination_split(split, split_path)
    # Also expose the canonical experiment split at repository level for review/versioning.
    canonical = Path(config["paths"]["project_root"]) / "splits" / split_path.name
    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(split_path, canonical)

    stains, scanners = split["stains"], split["scanners"]
    heldout = {(item["stain_id"], item["scanner_id"]) for item in split["heldout_cells"]}
    matrix = np.zeros((len(stains), len(scanners)))
    for i, stain in enumerate(stains):
        for j, scanner in enumerate(scanners):
            matrix[i, j] = 1 if (stain, scanner) in heldout else 0
    fig, axis = plt.subplots(figsize=(10, 9))
    axis.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#2b8a3e", "#c92a2a"]), vmin=0, vmax=1)
    axis.set_xticks(range(len(scanners)), scanners, rotation=30, ha="right")
    axis.set_yticks(range(len(stains)), stains)
    axis.set_xlabel("Scanner")
    axis.set_ylabel("Stain")
    axis.set_title(f"Deterministic held-out combinations, seed {config['seed']}\nred = never jointly observed during training")
    for i in range(len(stains)):
        for j in range(len(scanners)):
            axis.text(j, i, "TEST" if matrix[i, j] else "TRAIN", color="white", ha="center", va="center", fontsize=7, weight="bold")
    fig.tight_layout(); fig.savefig(out / "figures" / "heldout_combination_matrix.png", dpi=180, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()

