#!/usr/bin/env python
from __future__ import annotations

import argparse

from factorstain.training.external import train_camelyon
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m5_external.yaml")
    parser.add_argument("--model", required=True, choices=["uni", "virchow2"])
    parser.add_argument("--representation", required=True, choices=["raw", "adapter"])
    parser.add_argument("--heldout-center", required=True, type=int, choices=range(5))
    args = parser.parse_args()
    config = load_config(args.config); out = prepare_output(config)
    train_camelyon(config, args.model, args.representation, args.heldout_center, out)


if __name__ == "__main__":
    main()
