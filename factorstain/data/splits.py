from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def build_combination_split(
    index: pd.DataFrame,
    holdout_fraction: float = 0.18,
    seed: int = 42,
) -> dict:
    """Deterministically hold out cells while retaining every factor in training."""
    cells = sorted({(str(s), str(q)) for s, q in zip(index.stain_id, index.scanner_id)})
    stains = sorted({s for s, _ in cells})
    scanners = sorted({q for _, q in cells})
    desired = max(1, round(len(cells) * holdout_fraction))
    rng = np.random.default_rng(seed)
    candidates = list(cells)
    rng.shuffle(candidates)
    heldout: list[tuple[str, str]] = []
    for candidate in candidates:
        proposal = heldout + [candidate]
        remaining = set(cells) - set(proposal)
        if all(any(s == stain for s, _ in remaining) for stain in stains) and all(
            any(q == scanner for _, q in remaining) for scanner in scanners
        ):
            heldout.append(candidate)
        if len(heldout) == desired:
            break
    if len(heldout) < desired:
        raise RuntimeError("Unable to construct a valid combination holdout with the observed sparse grid")
    train = sorted(set(cells) - set(heldout))
    return {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "stains": stains,
        "scanners": scanners,
        "train_cells": [{"stain_id": s, "scanner_id": q} for s, q in train],
        "heldout_cells": [{"stain_id": s, "scanner_id": q} for s, q in sorted(heldout)],
    }


def save_combination_split(split: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")


def group_train_val_test_split(
    index: pd.DataFrame,
    group_column: str = "aligned_group_id",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.Series:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to one")
    groups = index[group_column].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, train_size=ratios[0], random_state=seed)
    train_idx, remainder_idx = next(splitter.split(index, groups=groups))
    labels = np.full(len(index), "", dtype=object)
    labels[train_idx] = "train"
    remainder = index.iloc[remainder_idx]
    relative_val = ratios[1] / (ratios[1] + ratios[2])
    inner = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=seed + 1)
    val_local, test_local = next(inner.split(remainder, groups=remainder[group_column].astype(str)))
    labels[remainder_idx[val_local]] = "val"
    labels[remainder_idx[test_local]] = "test"
    result = pd.Series(labels, index=index.index, name="morphology_split")
    for group, values in pd.DataFrame({"group": groups, "split": result}).groupby("group"):
        if values.split.nunique() != 1:
            raise AssertionError(f"Group leakage detected for {group}")
    return result

