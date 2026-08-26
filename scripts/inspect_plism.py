#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd

from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output
from factorstain.visualization.plism import (
    appearance_statistics,
    plot_appearance_by_factor,
    plot_plism_summary,
    plot_real_grid,
    plot_sharpness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m0_probe.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    index = pd.read_parquet(out / "plism_index.parquet")
    max_images = config["fast_dev_max_images"] if config["fast_dev_run"] else min(5000, config["max_probe_images"])
    stats = appearance_statistics(index, max_images=max_images, seed=config["seed"])
    stats.to_parquet(out / "appearance_statistics.parquet", index=False)
    figures = out / "figures"
    plot_real_grid(index, figures / "plism_13x7_real_grid.png", groups=1)
    plot_plism_summary(index, figures / "plism_dataset_summary.png")
    plot_appearance_by_factor(stats, "stain_id", figures / "color_statistics_by_stain.png")
    plot_appearance_by_factor(stats, "scanner_id", figures / "color_statistics_by_scanner.png")
    plot_sharpness(stats, figures / "sharpness_by_scanner.png")


if __name__ == "__main__":
    main()

