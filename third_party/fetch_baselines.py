#!/usr/bin/env python
"""Fetch reviewed official repositories at immutable commits without installing them."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DESTINATIONS = {
    "stainnet": "StainNet",
    "staingan": "StainGAN",
    "cyclegan": "pytorch-CycleGAN-and-pix2pix",
    "pix2pix": "pytorch-CycleGAN-and-pix2pix",
    "histaugan": "HistAuGAN",
    "cagan": "CAGAN",
    "sastaindiff": "SAStainDiff",
    "histofs": "HistoFS",
}


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def fetch(method: str, root: Path) -> None:
    from factorstain.baselines.registry import BASELINES

    spec = BASELINES[method]
    destination = root / DESTINATIONS[method]
    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(
                f"Refusing to overwrite non-git third-party directory: {destination}"
            )
        observed = _git("rev-parse", "HEAD", cwd=destination)
        if observed != spec.source_commit:
            raise RuntimeError(
                f"{destination} is at {observed}, expected reviewed commit "
                f"{spec.source_commit}; preserve local work and resolve manually"
            )
        print(f"{method}: already pinned at {observed}")
        return
    _git(
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        spec.official_repository,
        str(destination),
    )
    _git("checkout", "--detach", spec.source_commit, cwd=destination)
    observed = _git("rev-parse", "HEAD", cwd=destination)
    if observed != spec.source_commit:
        raise RuntimeError(f"Commit verification failed for {method}: {observed}")
    print(f"{method}: fetched {spec.official_repository}@{observed}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from factorstain.baselines.registry import resolve_methods

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="")
    parser.add_argument("--tier", default="all")
    args = parser.parse_args()
    methods = resolve_methods(methods=args.methods, tier=args.tier)
    root = Path(__file__).resolve().parent
    fetched = set()
    for method in methods:
        if method not in DESTINATIONS:
            continue
        destination = DESTINATIONS[method]
        if destination in fetched:
            continue
        fetch(method, root)
        fetched.add(destination)


if __name__ == "__main__":
    main()
