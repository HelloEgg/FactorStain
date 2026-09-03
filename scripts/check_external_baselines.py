#!/usr/bin/env python
"""Preflight pinned sources and bundled external neural adapter dependencies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factorstain.baselines.official_neural import (
    SUPPORTED_METHODS,
    _source_commit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(sorted(SUPPORTED_METHODS)))
    args = parser.parse_args()
    methods = [
        value.strip().lower() for value in args.methods.split(",") if value.strip()
    ]
    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported external methods: {sorted(unknown)}")
    for method in methods:
        root, commit = _source_commit(method)
        print(f"{method}: source ready {commit} ({root})")

    import numpy
    import pandas
    import torch
    import torchvision
    import yaml

    print(
        "adapter runtime ready: "
        f"python={sys.version.split()[0]}, torch={torch.__version__}, "
        f"torchvision={torchvision.__version__}, numpy={numpy.__version__}, "
        f"pandas={pandas.__version__}, pyyaml={yaml.__version__}, "
        f"cuda={torch.cuda.is_available()}"
    )


if __name__ == "__main__":
    main()
