#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import pandas as pd

from factorstain.training.renderer import train_renderer
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["factorstain", "joint", "parallel", "reverse"],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    index_path = out / "metadata" / "plism_index_with_splits.parquet"
    if not index_path.exists():
        index_path = out / "plism_index_with_splits.parquet"
    index = pd.read_parquet(index_path)
    split = json.loads(
        (out / "splits" / f"combination_split_seed{config['seed']}.json").read_text(
            encoding="utf-8"
        )
    )
    train_renderer(config, args.model, index, split, out)


if __name__ == "__main__":
    main()
