from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .midog import midog_scanner

MIDOG_VERBOSE_SCANNERS = {
    "Hamamatsu XR": "Hamamatsu NanoZoomer XR",
    "Hamamatsu S360": "Hamamatsu NanoZoomer S360",
    "Aperio CS2": "Aperio ScanScope CS2",
    "Leica GT450": "Leica GT450",
}


def stable_sample_id(parts: Iterable[object]) -> str:
    return hashlib.sha1(
        "|".join(str(value) for value in parts).encode("utf-8")
    ).hexdigest()[:24]


def image_quality(
    image: Image.Image, minimum_tissue_fraction: float, maximum_white_fraction: float
) -> dict[str, float | bool]:
    thumbnail = ImageOps.contain(
        image.convert("RGB"), (256, 256), method=Image.Resampling.BILINEAR
    )
    hsv = np.asarray(thumbnail.convert("HSV"), dtype=np.uint8)
    saturation, value = hsv[..., 1], hsv[..., 2]
    white = (value >= 245) & (saturation <= 20)
    tissue = (saturation > 20) & (value < 245)
    white_fraction = float(white.mean())
    tissue_fraction = float(tissue.mean())
    return {
        "white_fraction": white_fraction,
        "tissue_fraction": tissue_fraction,
        "quality_pass": tissue_fraction >= minimum_tissue_fraction
        and white_fraction <= maximum_white_fraction,
    }


def _balanced_order(frame: pd.DataFrame, columns: list[str], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    buckets = []
    for _, group in frame.groupby(columns, sort=True, dropna=False):
        values = group.index.to_numpy().copy()
        rng.shuffle(values)
        buckets.append(values.tolist())
    rng.shuffle(buckets)
    order: list[int] = []
    while buckets:
        remaining = []
        for bucket in buckets:
            if bucket:
                order.append(bucket.pop())
            if bucket:
                remaining.append(bucket)
        buckets = remaining
    return order


def sample_plism_for_audit(
    index: pd.DataFrame, config: dict, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = config["plism"]
    target = (
        settings["fast_dev_images"]
        if config["fast_dev_run"]
        else settings["max_feature_images"]
    )
    controlled_groups = (
        settings["fast_dev_controlled_groups"]
        if config["fast_dev_run"]
        else settings["controlled_feature_groups"]
    )
    valid = index[index.image_exists].copy()
    if valid.empty:
        raise RuntimeError("PLISM index contains no readable image files")
    group_summary = valid.groupby("aligned_group_id").agg(
        n=("image_id", "size"),
        stains=("stain_id", "nunique"),
        scanners=("scanner_id", "nunique"),
    )
    eligible_groups = group_summary[
        (group_summary.stains >= 2) & (group_summary.scanners >= 2)
    ].sort_values("n", ascending=False)
    rng = np.random.default_rng(seed)
    leading = (
        eligible_groups[eligible_groups.n == eligible_groups.n.max()].index.to_numpy()
        if len(eligible_groups)
        else np.asarray([])
    )
    rng.shuffle(leading)
    chosen_groups = leading[:controlled_groups].tolist()
    if len(chosen_groups) < controlled_groups:
        remainder = eligible_groups.index.difference(chosen_groups).to_numpy()
        rng.shuffle(remainder)
        chosen_groups.extend(
            remainder[: controlled_groups - len(chosen_groups)].tolist()
        )

    priority = valid[valid.aligned_group_id.isin(chosen_groups)].index.tolist()
    rest = valid[~valid.index.isin(priority)]
    order = priority + _balanced_order(
        rest, ["stain_id", "scanner_id", "tissue_type"], seed + 1
    )
    accepted, quality_rows = [], []
    stain_counts: dict[str, int] = {}
    scanner_counts: dict[str, int] = {}
    for position in order:
        row = valid.loc[position]
        stain_key, scanner_key = str(row.stain_id), str(row.scanner_id)
        stain_cap = int(settings["max_images_per_stain"])
        scanner_cap = int(settings["max_images_per_scanner"])
        if not config["fast_dev_run"] and (
            stain_counts.get(stain_key, 0) >= stain_cap
            or scanner_counts.get(scanner_key, 0) >= scanner_cap
        ):
            quality_rows.append(
                {
                    "dataset": "plism",
                    "sample_id": row.image_id,
                    "reason": "balanced_domain_quota",
                    "white_fraction": np.nan,
                    "tissue_fraction": np.nan,
                    "quality_pass": False,
                }
            )
            continue
        try:
            with Image.open(row.image_path) as opened:
                quality = image_quality(
                    opened,
                    settings["minimum_tissue_fraction"],
                    settings["maximum_white_fraction"],
                )
            reason = "accepted" if quality["quality_pass"] else "low_tissue_or_white"
        except Exception as exc:  # noqa: BLE001 - corrupted image backends raise heterogeneous errors
            quality = {
                "white_fraction": np.nan,
                "tissue_fraction": np.nan,
                "quality_pass": False,
            }
            reason = f"read_error:{type(exc).__name__}"
        quality_rows.append(
            {"dataset": "plism", "sample_id": row.image_id, "reason": reason, **quality}
        )
        if quality["quality_pass"]:
            accepted.append(position)
            stain_counts[stain_key] = stain_counts.get(stain_key, 0) + 1
            scanner_counts[scanner_key] = scanner_counts.get(scanner_key, 0) + 1
        if len(accepted) >= target:
            break
    if len(accepted) < min(target, 100 if not config["fast_dev_run"] else 20):
        raise RuntimeError(
            f"Only {len(accepted)} usable PLISM patches passed conservative tissue/background checks"
        )
    samples = valid.loc[accepted].copy().reset_index(drop=True)
    samples["sample_id"] = samples.image_id.astype(str)
    samples["dataset"] = "plism"
    samples["feature_selected"] = True
    samples = samples.merge(
        pd.DataFrame(quality_rows)[["sample_id", "tissue_fraction", "white_fraction"]],
        on="sample_id",
        how="left",
    )

    comparisons = []
    controlled_source = samples.copy()
    scanner_candidates = []
    for (group_id, stain), group in controlled_source.groupby(
        ["aligned_group_id", "stain_id"]
    ):
        if group.scanner_id.nunique() >= 2:
            scanner_candidates.append(
                (group.scanner_id.nunique(), str(group_id), str(stain), group)
            )
    scanner_candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    rng.shuffle(scanner_candidates)
    scanner_candidates.sort(key=lambda value: -value[0])
    for comparison_number, (_, group_id, stain, group) in enumerate(
        scanner_candidates[: settings["controlled_examples"]], start=1
    ):
        for order_number, (_, row) in enumerate(
            group.sort_values("scanner_id").iterrows()
        ):
            comparisons.append(
                {
                    "comparison_type": "same_morphology_same_stain_across_scanners",
                    "comparison_id": comparison_number,
                    "display_order": order_number,
                    **row.to_dict(),
                }
            )
    stain_candidates = []
    for (group_id, scanner), group in controlled_source.groupby(
        ["aligned_group_id", "scanner_id"]
    ):
        if group.stain_id.nunique() >= 2:
            stain_candidates.append(
                (group.stain_id.nunique(), str(group_id), str(scanner), group)
            )
    stain_candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    rng.shuffle(stain_candidates)
    stain_candidates.sort(key=lambda value: -value[0])
    for comparison_number, (_, group_id, scanner, group) in enumerate(
        stain_candidates[: settings["controlled_examples"]], start=1
    ):
        for order_number, (_, row) in enumerate(
            group.sort_values("stain_id").iterrows()
        ):
            comparisons.append(
                {
                    "comparison_type": "aligned_tissue_same_scanner_across_stains",
                    "comparison_id": comparison_number,
                    "display_order": order_number,
                    **row.to_dict(),
                }
            )
    return samples, pd.DataFrame(comparisons), pd.DataFrame(quality_rows)


def _find_midog_json(root: Path) -> Path:
    direct = root / "MIDOG.json"
    if direct.exists():
        return direct
    matches = list(root.rglob("MIDOG.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one MIDOG.json under {root}; found {len(matches)}"
        )
    return matches[0]


def build_midog_case_index(root: str | Path) -> tuple[pd.DataFrame, dict]:
    root = Path(root)
    annotation_path = _find_midog_json(root)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    if not images:
        raise RuntimeError(
            f"MIDOG metadata has no COCO image records: {annotation_path}"
        )
    unresolved = {Path(str(item.get("file_name", item["id"]))).name for item in images}
    file_map: dict[str, list[Path]] = {name: [] for name in unresolved}
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.name in file_map:
            file_map[candidate.name].append(candidate)
    rows, metadata_scanners, mismatches, unresolved_records = [], {}, [], []
    for item in images:
        declared = Path(str(item.get("file_name", item["id"])))
        direct = declared if declared.is_absolute() else root / declared
        matches = file_map.get(declared.name, [])
        image_path = (
            direct if direct.exists() else matches[0] if len(matches) == 1 else None
        )
        if image_path is None:
            unresolved_records.append(
                {
                    "file_name": str(declared),
                    "basename_matches": len(matches),
                }
            )
            continue
        case_id = declared.stem
        short_scanner = midog_scanner(case_id)
        resolved_scanner = MIDOG_VERBOSE_SCANNERS[short_scanner]
        metadata_value = next(
            (item[key] for key in ("scanner", "scanner_id", "device") if key in item),
            None,
        )
        if metadata_value is not None:
            metadata_scanners[case_id] = str(metadata_value)
            normalized_expected = "".join(
                character
                for character in resolved_scanner.lower()
                if character.isalnum()
            )
            normalized_short = "".join(
                character for character in short_scanner.lower() if character.isalnum()
            )
            normalized_observed = "".join(
                character
                for character in str(metadata_value).lower()
                if character.isalnum()
            )
            recognized = any(
                observed in expected or expected in observed
                for observed in (normalized_observed,)
                for expected in (normalized_expected, normalized_short)
            )
            if not recognized:
                mismatches.append(
                    {
                        "case_id": case_id,
                        "range_mapping": resolved_scanner,
                        "metadata_value": str(metadata_value),
                    }
                )
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(image_path.resolve()),
                "scanner_id": resolved_scanner,
                "range_scanner_id": short_scanner,
            }
        )
    verification = {
        "annotation_path": str(annotation_path.resolve()),
        "mapping_source": "MIDOG21 documented case-number ranges, cross-checked against image-record scanner fields when present",
        "metadata_scanner_fields_found": len(metadata_scanners),
        "metadata_mismatches": mismatches,
        "unresolved_image_count": len(unresolved_records),
        "unresolved_image_examples": unresolved_records[:20],
        "resolved_scanners": sorted({row["scanner_id"] for row in rows}),
    }
    if mismatches:
        raise RuntimeError(
            f"MIDOG scanner metadata disagrees with documented case-number mapping; examples: {mismatches[:5]}"
        )
    resolved_domains = {row["scanner_id"] for row in rows}
    expected_domains = set(MIDOG_VERBOSE_SCANNERS.values())
    if not expected_domains.issubset(resolved_domains):
        raise RuntimeError(
            "MIDOG indexing did not resolve cases from every documented scanner domain. "
            f"Resolved={sorted(resolved_domains)}; missing={sorted(expected_domains - resolved_domains)}; "
            f"unresolved image examples={unresolved_records[:5]}"
        )
    return pd.DataFrame(rows).drop_duplicates("case_id").sort_values(
        "case_id"
    ).reset_index(drop=True), verification


def _thumbnail_metadata(
    path: str | Path, maximum_size: int, assumed_mpp: float
) -> tuple[Image.Image, int, int, float, str, str]:
    try:
        import tiffslide

        with tiffslide.TiffSlide(str(path)) as slide:
            width, height = slide.dimensions
            thumbnail = slide.get_thumbnail((maximum_size, maximum_size)).convert("RGB")
            properties = slide.properties
            recorded_mpp = (
                properties.get("tiffslide.mpp-x")
                or properties.get("openslide.mpp-x")
                or properties.get("aperio.MPP")
            )
            mpp = (
                float(recorded_mpp) if recorded_mpp is not None else float(assumed_mpp)
            )
            mpp_source = (
                "slide_metadata"
                if recorded_mpp is not None
                else "configured_assumption"
            )
        return thumbnail, int(width), int(height), mpp, "tiffslide", mpp_source
    except Exception:  # noqa: BLE001 - unsupported TIFFs intentionally fall back to Pillow
        with Image.open(path) as opened:
            width, height = opened.size
            thumbnail = ImageOps.contain(
                opened.convert("RGB"),
                (maximum_size, maximum_size),
                method=Image.Resampling.BILINEAR,
            )
        return (
            thumbnail,
            int(width),
            int(height),
            float(assumed_mpp),
            "pillow_fallback",
            "configured_assumption",
        )


def sample_midog_case_patches(
    case: pd.Series | dict, settings: dict, seed: int = 42
) -> pd.DataFrame:
    case = pd.Series(case)
    thumbnail, width, height, source_mpp, backend, mpp_source = _thumbnail_metadata(
        case.image_path, settings["thumbnail_max_size"], settings["assumed_mpp"]
    )
    read_size = max(
        32, round(settings["patch_size"] * settings["target_mpp"] / source_mpp)
    )
    hsv = np.asarray(thumbnail.convert("HSV"), dtype=np.uint8)
    saturation, value = hsv[..., 1], hsv[..., 2]
    tissue_mask = (saturation > 20) & (value < 245)
    white_mask = (saturation <= 20) & (value >= 245)
    scale_x, scale_y = thumbnail.width / width, thumbnail.height / height
    stride = read_size
    candidates = []
    grid_candidates = 0
    for y in range(0, max(1, height - read_size + 1), stride):
        for x in range(0, max(1, width - read_size + 1), stride):
            grid_candidates += 1
            x0, x1 = (
                int(x * scale_x),
                max(int((x + read_size) * scale_x), int(x * scale_x) + 1),
            )
            y0, y1 = (
                int(y * scale_y),
                max(int((y + read_size) * scale_y), int(y * scale_y) + 1),
            )
            tissue_fraction = float(tissue_mask[y0:y1, x0:x1].mean())
            white_fraction = float(white_mask[y0:y1, x0:x1].mean())
            if (
                tissue_fraction >= settings["minimum_tissue_fraction"]
                and white_fraction <= settings["maximum_white_fraction"]
            ):
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "tissue_fraction": tissue_fraction,
                        "white_fraction": white_fraction,
                    }
                )
    if not candidates:
        empty = pd.DataFrame()
        empty.attrs.update({"grid_candidates": grid_candidates, "tissue_candidates": 0})
        return empty
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))[: settings["max_patches_per_case"]]
    selected = pd.DataFrame([candidates[int(position)] for position in order])
    selected["case_id"] = str(case.case_id)
    selected["image_path"] = str(case.image_path)
    selected["scanner_id"] = str(case.scanner_id)
    selected["level"] = 0
    selected["read_size"] = int(read_size)
    selected["patch_size"] = int(settings["patch_size"])
    selected["source_mpp"] = float(source_mpp)
    selected["target_mpp"] = float(settings["target_mpp"])
    selected["mpp_source"] = mpp_source
    selected["reader_backend"] = backend
    selected["slide_width"] = int(width)
    selected["slide_height"] = int(height)
    selected["sample_id"] = [
        stable_sample_id(
            (case.case_id, int(row.x), int(row.y), read_size, settings["patch_size"])
        )
        for _, row in selected.iterrows()
    ]
    selected["dataset"] = "midog21"
    selected.attrs.update(
        {"grid_candidates": grid_candidates, "tissue_candidates": len(candidates)}
    )
    return selected


def sample_midog_for_audit(
    cases: pd.DataFrame, config: dict, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = dict(config["midog"])
    target = (
        settings["fast_dev_patches_per_scanner"]
        if config["fast_dev_run"]
        else settings["patches_per_scanner"]
    )
    settings["max_patches_per_case"] = (
        settings["fast_dev_max_patches_per_case"]
        if config["fast_dev_run"]
        else settings["max_patches_per_case"]
    )
    sampled, quality = [], []
    for scanner_number, (scanner, group) in enumerate(
        cases.groupby("scanner_id", sort=True)
    ):
        case_order = group.sample(
            frac=1, random_state=seed + scanner_number
        ).reset_index(drop=True)
        scanner_samples = []
        for case_number, (_, case) in enumerate(case_order.iterrows()):
            patches = sample_midog_case_patches(
                case, settings, seed + scanner_number * 10000 + case_number
            )
            quality.append(
                {
                    "dataset": "midog21",
                    "case_id": case.case_id,
                    "scanner_id": scanner,
                    "grid_candidates": patches.attrs.get("grid_candidates", 0),
                    "tissue_candidates": patches.attrs.get("tissue_candidates", 0),
                    "candidate_accepted": len(patches),
                    "reason": "accepted" if len(patches) else "no_tissue_candidates",
                }
            )
            if len(patches):
                scanner_samples.append(patches)
            if sum(len(frame) for frame in scanner_samples) >= target:
                break
        if not scanner_samples:
            raise RuntimeError(
                f"No usable MIDOG tissue patches found for scanner {scanner}"
            )
        sampled.append(pd.concat(scanner_samples, ignore_index=True).head(target))
    samples = pd.concat(sampled, ignore_index=True)
    counts = samples.groupby("scanner_id").size()
    if (counts < target).any():
        raise RuntimeError(
            f"MIDOG tissue sampling did not reach the per-scanner target {target}: {counts.to_dict()}"
        )
    return samples, pd.DataFrame(quality)


def read_midog_patch(row: pd.Series | dict) -> Image.Image:
    row = pd.Series(row)
    size = int(row.read_size)
    try:
        import tiffslide

        with tiffslide.TiffSlide(str(row.image_path)) as slide:
            patch = slide.read_region(
                (int(row.x), int(row.y)), int(row.get("level", 0)), (size, size)
            ).convert("RGB")
    except Exception:  # noqa: BLE001 - unsupported TIFFs intentionally fall back to Pillow
        with Image.open(row.image_path) as opened:
            patch = opened.convert("RGB").crop(
                (int(row.x), int(row.y), int(row.x) + size, int(row.y) + size)
            )
    return ImageOps.fit(
        patch,
        (int(row.patch_size), int(row.patch_size)),
        method=Image.Resampling.BILINEAR,
    )
