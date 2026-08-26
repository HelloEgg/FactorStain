#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from factorstain.data.plism import build_plism_index, discover_plism_metadata
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output
from factorstain.utils.preflight import preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m0_probe.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    preflight(config, dataset_keys=("plism_root",))
    out = prepare_output(config)
    metadata = discover_plism_metadata(config["paths"]["plism_root"])
    index = build_plism_index(config["paths"]["plism_root"], metadata)
    if not index.image_exists.any():
        raise RuntimeError(
            "PLISM metadata was parsed, but none of its image paths resolve. "
            "Check whether paths are relative to a different PLISM_ROOT."
        )
    destination = out / "plism_index.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    index.to_parquet(temporary, index=False)
    temporary.replace(destination)
    print(f"PLISM metadata: {metadata}")
    print(f"Indexed {len(index):,} images ({index.image_exists.sum():,} existing) to {destination}")


if __name__ == "__main__":
    main()
