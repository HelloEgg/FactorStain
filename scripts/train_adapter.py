#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from factorstain.training.adapter import train_adapter
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m4_adapter.yaml")
    parser.add_argument("--model", required=True, choices=["uni", "virchow2"])
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    cache = out / "counterfactual_features" / args.model / "features.h5"
    if not cache.exists():
        print(f"Skipping {args.model}: counterfactual feature cache unavailable ({cache})")
        return
    train_adapter(config, args.model, cache, out)


if __name__ == "__main__":
    main()
