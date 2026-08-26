from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


ALIASES = {
    "image_path": ("image_path", "path", "filepath", "file_path", "filename", "file", "patch_path"),
    "tissue_type": ("tissue_type", "tissue", "class", "label", "organ", "tissue_class"),
    "stain_id": ("stain_id", "stain", "staining", "staining_id", "stain_condition", "stain_type", "stainer"),
    "scanner_id": ("scanner_id", "scanner", "scan", "device", "scanner_name", "device_type"),
    "x": ("x", "coord_x", "x_coord", "position_x", "patch_x"),
    "y": ("y", "coord_y", "y_coord", "position_y", "patch_y"),
    "coordinate": ("coordinate", "coordinates", "coord", "position"),
    "group": ("aligned_group_id", "registration_id", "aligned_id", "group_id", "patch_id", "location_id", "tile_id"),
    "slide": ("slide_id", "wsi_id", "case_id", "sample_id", "subject_id", "slide"),
}


def _normalise_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _find_column(frame: pd.DataFrame, semantic: str, required: bool = False) -> str | None:
    columns = {_normalise_column(c): c for c in frame.columns}
    for alias in ALIASES[semantic]:
        if alias in columns:
            return columns[alias]
    if required:
        raise ValueError(
            f"Could not infer {semantic!r} from metadata columns: {list(frame.columns)}. "
            f"Recognized aliases: {ALIASES[semantic]}"
        )
    return None


def discover_plism_metadata(root: str | Path) -> Path:
    """Find the most plausible PLISM patch metadata table without filename assumptions."""
    root = Path(root)
    candidates = [*root.rglob("*.csv"), *root.rglob("*.parquet")]
    if not candidates:
        raise FileNotFoundError(f"No CSV or parquet metadata file found recursively under {root}")
    scored: list[tuple[int, int, Path]] = []
    for candidate in candidates:
        try:
            frame = pd.read_csv(candidate, nrows=20) if candidate.suffix.lower() == ".csv" else pd.read_parquet(candidate).head(20)
            normalized = {_normalise_column(c) for c in frame.columns}
            score = sum(any(alias in normalized for alias in aliases) for aliases in ALIASES.values())
            scored.append((score, candidate.stat().st_size, candidate))
        except Exception:
            continue
    if not scored:
        raise RuntimeError(f"Metadata candidates were found under {root}, but none could be read")
    scored.sort(reverse=True)
    best_score, _, best = scored[0]
    if best_score < 3:
        raise RuntimeError(f"No metadata file under {root} has recognizable path/stain/scanner columns")
    return best


def _resolve_image_paths(values: pd.Series, root: Path) -> pd.Series:
    resolved: dict[object, str] = {}
    unresolved_names: set[str] = set()
    for value in values.drop_duplicates():
        path = Path(str(value))
        direct = path if path.is_absolute() else root / path
        if direct.exists():
            resolved[value] = str(direct.resolve())
        else:
            unresolved_names.add(path.name)
    if unresolved_names:
        by_name: dict[str, list[Path]] = {name: [] for name in unresolved_names}
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.name in by_name:
                by_name[candidate.name].append(candidate)
        for value in values.drop_duplicates():
            if value in resolved:
                continue
            path = Path(str(value))
            matches = by_name.get(path.name, [])
            direct = path if path.is_absolute() else root / path
            resolved[value] = str(matches[0].resolve()) if len(matches) == 1 else str(direct.resolve())
    return values.map(resolved)


def _stable_id(parts: Sequence[object]) -> str:
    normalized = "|".join(str(value) for value in parts)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]


def build_plism_index(root: str | Path, metadata_path: str | Path | None = None) -> pd.DataFrame:
    """Normalize release-specific PLISM metadata into the FactorStain index schema."""
    root = Path(root)
    metadata = Path(metadata_path) if metadata_path else discover_plism_metadata(root)
    raw = pd.read_csv(metadata) if metadata.suffix.lower() == ".csv" else pd.read_parquet(metadata)
    path_col = _find_column(raw, "image_path", required=True)
    tissue_col = _find_column(raw, "tissue_type", required=True)
    stain_col = _find_column(raw, "stain_id", required=True)
    scanner_col = _find_column(raw, "scanner_id", required=True)
    x_col, y_col = _find_column(raw, "x"), _find_column(raw, "y")
    coordinate_col = _find_column(raw, "coordinate")
    group_col, slide_col = _find_column(raw, "group"), _find_column(raw, "slide")

    if x_col and y_col:
        coordinates = raw[x_col].astype(str) + "," + raw[y_col].astype(str)
    elif coordinate_col:
        coordinates = raw[coordinate_col].astype(str)
    else:
        # Remove acquisition tokens from filenames as a documented last-resort alignment key.
        coordinates = raw[path_col].map(
            lambda value: re.sub(r"(?i)(stain|scanner|scan|device)[-_]?[a-z0-9]+", "", Path(str(value)).stem)
        )
    slide_values = raw[slide_col].astype(str) if slide_col else pd.Series("unknown_slide", index=raw.index)
    if group_col:
        groups = raw[group_col].astype(str)
    else:
        groups = pd.Series(
            [_stable_id((slide, tissue, coordinate)) for slide, tissue, coordinate in zip(slide_values, raw[tissue_col], coordinates)],
            index=raw.index,
        )

    index = pd.DataFrame(
        {
            "image_path": _resolve_image_paths(raw[path_col], root),
            "tissue_type": raw[tissue_col].astype(str),
            "stain_id": raw[stain_col].astype(str),
            "scanner_id": raw[scanner_col].astype(str),
            "coordinate": coordinates,
            "aligned_group_id": groups,
            "slide_id": slide_values,
        }
    )
    index["image_id"] = index.apply(
        lambda row: _stable_id((row.image_path, row.stain_id, row.scanner_id)), axis=1
    )
    index["image_exists"] = index.image_path.map(lambda value: Path(value).is_file())
    duplicated = index.duplicated(["aligned_group_id", "stain_id", "scanner_id"], keep=False)
    if duplicated.any():
        examples = index.loc[duplicated, ["aligned_group_id", "stain_id", "scanner_id"]].head().to_dict("records")
        raise ValueError(f"Duplicate acquisition cells found in aligned groups; examples: {examples}")
    return index


class PLISMDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame | str | Path,
        transform: Callable | None = None,
        return_pil: bool = False,
    ) -> None:
        self.index = pd.read_parquet(index) if isinstance(index, (str, Path)) else index.reset_index(drop=True)
        self.transform = transform
        self.return_pil = return_pil
        self.stain_to_idx = {value: i for i, value in enumerate(sorted(self.index.stain_id.unique()))}
        self.scanner_to_idx = {value: i for i, value in enumerate(sorted(self.index.scanner_id.unique()))}
        self.tissue_to_idx = {value: i for i, value in enumerate(sorted(self.index.tissue_type.unique()))}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        row = self.index.iloc[item]
        with Image.open(row.image_path) as opened:
            image = opened.convert("RGB")
        output_image = self.transform(image) if self.transform else image
        if not self.return_pil and self.transform is None:
            array = np.asarray(image, dtype=np.float32) / 255.0
            output_image = torch.from_numpy(array).permute(2, 0, 1)
        return {
            "image": output_image,
            "image_id": row.image_id,
            "stain": self.stain_to_idx[row.stain_id],
            "scanner": self.scanner_to_idx[row.scanner_id],
            "tissue": self.tissue_to_idx[row.tissue_type],
            "aligned_group_id": row.aligned_group_id,
            "path": row.image_path,
        }
