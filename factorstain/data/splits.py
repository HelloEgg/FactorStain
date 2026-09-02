from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _select_cells(
    cells: set[tuple[str, str]],
    count: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Select cells while leaving every stain and scanner in the remainder."""
    stains = {stain for stain, _ in cells}
    scanners = {scanner for _, scanner in cells}
    candidates = sorted(cells)
    np.random.default_rng(seed).shuffle(candidates)
    selected: list[tuple[str, str]] = []
    for candidate in candidates:
        remainder = cells - {*selected, candidate}
        if all(
            any(stain == value for stain, _ in remainder) for value in stains
        ) and all(
            any(scanner == value for _, scanner in remainder) for value in scanners
        ):
            selected.append(candidate)
        if len(selected) == count:
            return sorted(selected)
    raise RuntimeError(
        f"Unable to select {count} acquisition cells while retaining every factor"
    )


def build_combination_split(
    index: pd.DataFrame,
    holdout_fraction: float = 0.18,
    seed: int = 42,
    validation_fraction: float = 0.10,
) -> dict:
    """Create deterministic train/validation/test acquisition-cell partitions.

    Test cells are the never-jointly-observed combinations used by both M1
    protocols. Validation cells are also excluded from generator training, but
    never contribute to the final test result.
    """
    required = {"stain_id", "scanner_id"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Combination split metadata is missing {sorted(missing)}")
    cells = {
        (str(stain), str(scanner))
        for stain, scanner in zip(index.stain_id, index.scanner_id)
    }
    if len(cells) < 4:
        raise ValueError("At least four observed stain/scanner cells are required")
    stains = sorted({stain for stain, _ in cells})
    scanners = sorted({scanner for _, scanner in cells})
    test_count = max(1, round(len(cells) * holdout_fraction))
    validation_count = max(1, round(len(cells) * validation_fraction))
    test = _select_cells(cells, test_count, seed)
    after_test = cells - set(test)
    validation = _select_cells(after_test, validation_count, seed + 1)
    train = sorted(after_test - set(validation))
    train_stains = {stain for stain, _ in train}
    train_scanners = {scanner for _, scanner in train}
    if train_stains != set(stains) or train_scanners != set(scanners):
        raise AssertionError(
            "Every stain and scanner must remain individually observed in training"
        )
    return {
        "seed": int(seed),
        "holdout_fraction": float(holdout_fraction),
        "validation_fraction": float(validation_fraction),
        "observed_cell_count": len(cells),
        "full_grid_cell_count": len(stains) * len(scanners),
        "stains": stains,
        "scanners": scanners,
        "train_cells": [
            {"stain_id": stain, "scanner_id": scanner} for stain, scanner in train
        ],
        "validation_cells": [
            {"stain_id": stain, "scanner_id": scanner} for stain, scanner in validation
        ],
        "test_cells": [
            {"stain_id": stain, "scanner_id": scanner} for stain, scanner in test
        ],
        # Backward-compatible name used by existing M1/M2 consumers.
        "heldout_cells": [
            {"stain_id": stain, "scanner_id": scanner} for stain, scanner in test
        ],
    }


def save_combination_split(split: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")


def _group_tissue(index: pd.DataFrame, group_column: str) -> pd.Series:
    def stable_mode(values: pd.Series) -> str:
        counts = values.astype(str).value_counts()
        return min(counts[counts.eq(counts.max())].index)

    return index.groupby(group_column).tissue_type.apply(stable_mode)


def group_train_val_test_split(
    index: pd.DataFrame,
    group_column: str = "aligned_group_id",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.Series:
    """Tissue-aware deterministic group split with no morphology leakage."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to one")
    required = {group_column}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Morphology split metadata is missing {sorted(missing)}")
    if "tissue_type" not in index:
        index = index.assign(tissue_type="unknown")
    group_tissues = _group_tissue(index, group_column)
    assignments: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    labels = ("train", "val", "test")
    for _, tissue_groups in group_tissues.groupby(group_tissues, sort=True):
        groups = tissue_groups.index.astype(str).to_numpy()
        rng.shuffle(groups)
        raw = np.asarray(ratios) * len(groups)
        counts = np.floor(raw).astype(int)
        for position in np.argsort(-(raw - counts))[: len(groups) - counts.sum()]:
            counts[position] += 1
        # When the stratum permits it, represent it in all three partitions.
        if len(groups) >= 3:
            for position in range(3):
                if counts[position] == 0:
                    donor = int(np.argmax(counts))
                    counts[donor] -= 1
                    counts[position] += 1
        start = 0
        for label, count in zip(labels, counts):
            for group in groups[start : start + count]:
                assignments[str(group)] = label
            start += count
    result = index[group_column].astype(str).map(assignments).rename("morphology_split")
    if result.isna().any():
        raise AssertionError("At least one morphology group was not assigned")
    leakage = index.assign(_split=result).groupby(group_column)._split.nunique()
    if leakage.max() != 1:
        raise AssertionError(
            "aligned_group_id leakage detected across morphology splits"
        )
    return result


def morphology_split_manifest(
    index: pd.DataFrame,
    labels: pd.Series,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict:
    frame = index.assign(morphology_split=labels)
    split_ids = {
        name: sorted(
            frame.loc[frame.morphology_split.eq(name), "aligned_group_id"]
            .astype(str)
            .unique()
        )
        for name in ("train", "val", "test")
    }
    tissue_counts = (
        frame.drop_duplicates("aligned_group_id")
        .groupby(["morphology_split", "tissue_type"])
        .size()
        .rename("groups")
        .reset_index()
        .to_dict("records")
    )
    return {
        "seed": int(seed),
        "ratios": {
            name: float(value) for name, value in zip(("train", "val", "test"), ratios)
        },
        "aligned_group_ids": split_ids,
        "group_counts": {name: len(values) for name, values in split_ids.items()},
        "tissue_group_counts": tissue_counts,
        "leakage_free": not any(
            set(split_ids[left]) & set(split_ids[right])
            for left, right in (
                ("train", "val"),
                ("train", "test"),
                ("val", "test"),
            )
        ),
    }
