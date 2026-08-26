#!/usr/bin/env python
from __future__ import annotations

import argparse

from factorstain.utils.config import load_config
from factorstain.utils.outputs import update_master


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/m0_probe.yaml"); args = parser.parse_args()
    config = load_config(args.config); update_master(config["paths"]["outputs_root"])


if __name__ == "__main__": main()
