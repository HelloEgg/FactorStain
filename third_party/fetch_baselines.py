#!/usr/bin/env python
"""Fetch reviewed official repositories at immutable commits without installing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
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
SOURCE_MARKER = ".factorstain-source.json"


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _tracked_files(checkout: Path) -> list[str]:
    return [
        value
        for value in _git("ls-files", cwd=checkout).splitlines()
        if value and value != SOURCE_MARKER
    ]


def _snapshot_digest(root: Path, files: list[str]) -> str | None:
    digest = hashlib.sha256()
    for relative in files:
        path = root / relative
        if not path.exists() or not path.is_file():
            return None
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _write_marker(destination: Path, spec, files: list[str], digest: str) -> None:
    marker = destination / SOURCE_MARKER
    payload = {
        "schema_version": 1,
        "official_repository": spec.official_repository,
        "source_commit": spec.source_commit,
        "verification": "all upstream tracked files matched the pinned commit",
        "tracked_file_count": len(files),
        "tracked_files": files,
        "snapshot_sha256": digest,
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)


def _valid_marker(destination: Path, spec) -> bool:
    marker = destination / SOURCE_MARKER
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        files = [str(value) for value in payload["tracked_files"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("official_repository") == spec.official_repository
        and payload.get("source_commit") == spec.source_commit
        and _snapshot_digest(destination, files) == payload.get("snapshot_sha256")
    )


def _clone_pinned(spec, destination: Path) -> None:
    _git(
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        spec.official_repository,
        str(destination),
    )
    _git("checkout", "--detach", spec.source_commit, cwd=destination)


def _verify_vendored_snapshot(destination: Path, spec, root: Path) -> None:
    if _valid_marker(destination, spec):
        print(f"{destination.name}: verified vendored snapshot at {spec.source_commit}")
        return
    if not destination.is_dir():
        raise RuntimeError(f"Third-party destination is not a directory: {destination}")
    with tempfile.TemporaryDirectory(prefix=".factorstain-verify-", dir=root) as temp:
        checkout = Path(temp) / "checkout"
        _clone_pinned(spec, checkout)
        observed = _git("rev-parse", "HEAD", cwd=checkout)
        if observed != spec.source_commit:
            raise RuntimeError(
                f"Temporary verification checkout is at {observed}, expected "
                f"{spec.source_commit}"
            )
        files = _tracked_files(checkout)
        expected = _snapshot_digest(checkout, files)
        actual = _snapshot_digest(destination, files)
        if expected is None or actual != expected:
            raise RuntimeError(
                f"Existing non-git directory does not exactly match the tracked files "
                f"at {spec.official_repository}@{spec.source_commit}: {destination}. "
                "It was not overwritten; move it aside or restore the pinned snapshot."
            )
        _write_marker(destination, spec, files, expected)
    print(f"{destination.name}: verified vendored snapshot at {spec.source_commit}")


def fetch(method: str, root: Path) -> None:
    from factorstain.baselines.registry import BASELINES

    spec = BASELINES[method]
    destination = root / DESTINATIONS[method]
    if destination.exists():
        if destination.is_dir() and not any(destination.iterdir()):
            # A fresh checkout can materialize a gitlink as an empty directory.
            # ``git clone`` accepts an existing empty destination, so initialize it
            # directly instead of requiring the user to remove it first.
            _clone_pinned(spec, destination)
            observed = _git("rev-parse", "HEAD", cwd=destination)
            if observed != spec.source_commit:
                raise RuntimeError(
                    f"Commit verification failed for {method}: {observed}"
                )
            files = _tracked_files(destination)
            digest = _snapshot_digest(destination, files)
            if digest is None:
                raise RuntimeError(
                    f"Pinned checkout is missing tracked files: {destination}"
                )
            _write_marker(destination, spec, files, digest)
            print(
                f"{method}: initialized empty gitlink from "
                f"{spec.official_repository}@{observed}"
            )
            return
        if not (destination / ".git").exists():
            _verify_vendored_snapshot(destination, spec, root)
            return
        observed = _git("rev-parse", "HEAD", cwd=destination)
        if observed != spec.source_commit:
            raise RuntimeError(
                f"{destination} is at {observed}, expected reviewed commit "
                f"{spec.source_commit}; preserve local work and resolve manually"
            )
        modified = _git(
            "status", "--porcelain", "--untracked-files=no", cwd=destination
        )
        if modified:
            raise RuntimeError(
                f"Tracked files under {destination} differ from pinned commit "
                f"{spec.source_commit}; preserve or restore the local changes first"
            )
        print(f"{method}: already pinned at {observed}")
        return
    _clone_pinned(spec, destination)
    observed = _git("rev-parse", "HEAD", cwd=destination)
    if observed != spec.source_commit:
        raise RuntimeError(f"Commit verification failed for {method}: {observed}")
    files = _tracked_files(destination)
    digest = _snapshot_digest(destination, files)
    if digest is None:
        raise RuntimeError(f"Pinned checkout is missing tracked files: {destination}")
    _write_marker(destination, spec, files, digest)
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
