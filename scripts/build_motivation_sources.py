#!/usr/bin/env python
"""Render CVPR motivation source panels from completed FactorStain experiments.

This script is intentionally strict: it never substitutes approximate metrics,
recomputes UMAP, or silently uses a FAST_DEV_RUN.  It only performs inexpensive
plotting and nearest-neighbor retrieval from the saved frozen-DINOv3 cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorstain.visualization.paper_assets import (
    load_paper_asset_config,
    print_dry_run,
)

DPI = 300
SCANNER_ORDER = ("AT2", "GT450", "P", "S210", "S360", "S60", "SQ")
CONTROLLED_CATEGORIES = (
    "same morphology / same stain / different scanner",
    "different morphology / same stain / same scanner",
    "different morphology / same stain / different scanner",
)
CONTROLLED_SHORT_LABELS = (
    "Same morph.\nΔ scanner",
    "Δ morph.\nSame scanner",
    "Δ morph.\nΔ scanner",
)
MOTIVATION_OUTPUT_FILES = (
    "01_same_tissue_across_scanners.png",
    "02_aligned_tissue_across_stains.png",
    "03_raw_domain_examples.png",
    "04_dinov3_umap_scanner.png",
    "05_dinov3_umap_stain.png",
    "06_dinov3_umap_tissue.png",
    "07_probe_accuracy.png",
    "08_controlled_feature_distance.png",
    "09_scanner_mmd_controlled.png",
    "10_stain_mmd_balanced.png",
    "11_nearest_neighbor_examples.png",
    "12_motivation_key_numbers.png",
    "manifest.json",
    "README.md",
)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read the repository's NPZ cache without requiring an editable install."""
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        sample_ids = payload["sample_id"].astype(str)
        metadata = json.loads(str(payload["metadata_json"].item()))
    if len(features) != len(sample_ids):
        raise RuntimeError(
            f"Feature cache row mismatch in {path}: "
            f"{len(features)} features vs {len(sample_ids)} IDs"
        )
    return features, sample_ids, metadata


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        lines = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Required completed-experiment artifacts are missing:\n"
            f"  - {lines}\n"
            "Copy/mount the existing M-1 domain-audit and MMD outputs; do not "
            "replace them with approximate values."
        )


def _assert_full_run(payload: dict[str, Any], source: Path) -> None:
    if payload.get("fast_dev_run") or payload.get("audit_valid") is False:
        raise RuntimeError(f"Refusing non-scientific FAST_DEV_RUN metrics: {source}")


def _save_plot(fig: plt.Figure, destination: Path, vectors: bool = True) -> list[str]:
    fig.savefig(destination, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    files = [destination.name]
    if vectors:
        for suffix in (".svg", ".pdf"):
            companion = destination.with_suffix(suffix)
            fig.savefig(companion, bbox_inches="tight", pad_inches=0.03)
            files.append(companion.name)
    plt.close(fig)
    return files


def _scanner_key(value: object) -> tuple[int, str]:
    text = str(value)
    normalized = text.upper().replace(" ", "").replace("-", "")
    for position, known in enumerate(SCANNER_ORDER):
        if normalized == known or normalized.endswith(known):
            return position, text
    return len(SCANNER_ORDER), text


def _natural_key(value: object) -> tuple[Any, ...]:
    import re

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def _resolve_image_paths(
    frame: pd.DataFrame, path_maps: list[tuple[str, str]]
) -> pd.DataFrame:
    result = frame.copy()

    def resolve(value: object) -> str:
        text = str(value)
        candidate = Path(text)
        if candidate.exists():
            return str(candidate.resolve())
        normalized = text.replace("\\", "/")
        for old, new in path_maps:
            old_normalized = old.replace("\\", "/").rstrip("/")
            if normalized == old_normalized or normalized.startswith(
                old_normalized + "/"
            ):
                relative = normalized[len(old_normalized) :].lstrip("/")
                mapped = Path(new) / Path(relative)
                if mapped.exists():
                    return str(mapped.resolve())
        return text

    result["image_path"] = result.image_path.map(resolve)
    return result


def _validate_image_rows(frame: pd.DataFrame, context: str) -> None:
    missing = [path for path in frame.image_path.astype(str) if not Path(path).exists()]
    if missing:
        preview = "\n  - ".join(missing[:8])
        raise FileNotFoundError(
            f"{context} references unavailable PLISM image files:\n  - {preview}\n"
            "Use --path-map OLD=NEW if the dataset root moved."
        )


def _load_rgb(path: object) -> np.ndarray:
    with Image.open(str(path)) as opened:
        return np.asarray(opened.convert("RGB"))


def _plot_strip(
    rows: pd.DataFrame,
    label_column: str,
    destination: Path,
    title: str,
) -> list[str]:
    count = len(rows)
    fig, axes = plt.subplots(1, count, figsize=(1.65 * count, 1.82), squeeze=False)
    for axis, (_, row) in zip(axes[0], rows.iterrows()):
        axis.imshow(_load_rgb(row.image_path), interpolation="none")
        axis.set_title(str(row[label_column]), fontsize=8.5, pad=3)
        axis.set_axis_off()
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle(title, fontsize=9.5, y=0.995)
    fig.tight_layout(pad=0.15, rect=(0, 0, 1, 0.92))
    return _save_plot(fig, destination, vectors=False)


def _choose_scanner_strip(samples: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[int, float, str, str, pd.DataFrame]] = []
    for (group_id, stain), group in samples.groupby(
        ["aligned_group_id", "stain_id"], sort=True
    ):
        group = group.drop_duplicates("scanner_id")
        if group.scanner_id.nunique() < 2:
            continue
        tissue = float(group.get("tissue_fraction", pd.Series([0.0])).median())
        candidates.append(
            (group.scanner_id.nunique(), tissue, str(group_id), str(stain), group)
        )
    if not candidates:
        raise RuntimeError(
            "No same-group/same-stain multi-scanner PLISM strip is available"
        )
    chosen = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[-1]
    return chosen.sort_values("scanner_id", key=lambda values: values.map(_scanner_key))


def _even_subset(rows: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(rows) <= count:
        return rows
    positions = np.round(np.linspace(0, len(rows) - 1, count)).astype(int)
    return rows.iloc[positions]


def _choose_stain_strip(samples: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[int, float, str, str, pd.DataFrame]] = []
    for (group_id, scanner), group in samples.groupby(
        ["aligned_group_id", "scanner_id"], sort=True
    ):
        group = group.drop_duplicates("stain_id")
        if group.stain_id.nunique() < 5:
            continue
        tissue = float(group.get("tissue_fraction", pd.Series([0.0])).median())
        candidates.append(
            (group.stain_id.nunique(), tissue, str(group_id), str(scanner), group)
        )
    if not candidates:
        raise RuntimeError(
            "No aligned-group/same-scanner PLISM strip has at least five stains"
        )
    chosen = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[-1]
    chosen = chosen.sort_values("stain_id", key=lambda values: values.map(_natural_key))
    return _even_subset(chosen, 7)


def _choose_domain_grid(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    best: (
        tuple[tuple[int, int, float, str], pd.DataFrame, list[str], list[str]] | None
    ) = None
    for group_id, source in samples.groupby("aligned_group_id", sort=True):
        source = source.drop_duplicates(["stain_id", "scanner_id"])
        stains = sorted(source.stain_id.astype(str).unique(), key=_natural_key)
        scanners = sorted(source.scanner_id.astype(str).unique(), key=_scanner_key)
        observed = set(zip(source.stain_id.astype(str), source.scanner_id.astype(str)))
        tissue = float(source.get("tissue_fraction", pd.Series([0.0])).median())
        group_choice: tuple[list[str], list[str]] | None = None
        for n_stains in range(min(4, len(stains)), 1, -1):
            for n_scanners in range(min(4, len(scanners)), 1, -1):
                for stain_set in combinations(stains, n_stains):
                    for scanner_set in combinations(scanners, n_scanners):
                        if all(
                            (stain, scanner) in observed
                            for stain in stain_set
                            for scanner in scanner_set
                        ):
                            group_choice = (list(stain_set), list(scanner_set))
                            break
                    if group_choice is not None:
                        break
                if group_choice is not None:
                    break
            if group_choice is not None:
                break
        if group_choice is None:
            continue
        stain_set, scanner_set = group_choice
        selected = source[
            source.stain_id.astype(str).isin(stain_set)
            & source.scanner_id.astype(str).isin(scanner_set)
        ]
        score = (
            len(stain_set) * len(scanner_set),
            len(stain_set) + len(scanner_set),
            tissue,
            str(group_id),
        )
        if best is None or score > best[0]:
            best = (score, selected, stain_set, scanner_set)
    if best is None:
        raise RuntimeError(
            "No complete 2×2 stain/scanner grid exists within an aligned PLISM group"
        )
    return best[1], best[2], best[3]


def _plot_domain_grid(
    selected: pd.DataFrame,
    stains: list[str],
    scanners: list[str],
    destination: Path,
) -> list[str]:
    fig, axes = plt.subplots(
        len(stains),
        len(scanners),
        figsize=(1.5 * len(scanners), 1.5 * len(stains)),
        squeeze=False,
    )
    lookup = {
        (str(row.stain_id), str(row.scanner_id)): row for _, row in selected.iterrows()
    }
    for row_number, stain in enumerate(stains):
        for column_number, scanner in enumerate(scanners):
            axis = axes[row_number, column_number]
            row = lookup[(stain, scanner)]
            axis.imshow(_load_rgb(row.image_path), interpolation="none")
            axis.set_axis_off()
            axis.set_aspect("equal", adjustable="box")
            if row_number == 0:
                axis.set_title(scanner, fontsize=8.5, pad=3)
            if column_number == 0:
                axis.text(
                    -0.04,
                    0.5,
                    stain,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=8.5,
                )
    fig.tight_layout(pad=0.12)
    return _save_plot(fig, destination, vectors=False)


def _categorical_colors(count: int, palette: str = "tab20") -> list[Any]:
    cmap = plt.get_cmap(palette)
    if count <= getattr(cmap, "N", count):
        return [cmap(position) for position in np.linspace(0, 1, count, endpoint=False)]
    return [
        plt.get_cmap("turbo")(position) for position in np.linspace(0.04, 0.96, count)
    ]


def _plot_umap(
    frame: pd.DataFrame,
    factor: str,
    destination: Path,
    limits: tuple[tuple[float, float], tuple[float, float]],
    order: list[str],
    show_legend: bool,
) -> list[str]:
    fig, axis = plt.subplots(figsize=(5.2, 4.3))
    colors = _categorical_colors(len(order))
    for color, label in zip(colors, order):
        selected = frame[frame[factor].astype(str).eq(label)]
        axis.scatter(
            selected.umap_1,
            selected.umap_2,
            s=5,
            alpha=0.72,
            linewidths=0,
            color=color,
            rasterized=True,
            label=label,
        )
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.65)
    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=color,
                markersize=4,
                label=label,
            )
            for color, label in zip(colors, order)
        ]
        columns = 2 if len(order) > 8 else 1
        axis.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            ncol=columns,
            columnspacing=0.8,
            handletextpad=0.35,
        )
    else:
        axis.text(
            0.02,
            0.02,
            f"{len(order)} tissue classes",
            transform=axis.transAxes,
            fontsize=7.5,
            color="#374151",
            ha="left",
            va="bottom",
        )
    fig.tight_layout(pad=0.2)
    return _save_plot(fig, destination)


def _probe(metrics: dict[str, Any], target: str) -> dict[str, Any]:
    matches = [
        row
        for row in metrics.get("probes", [])
        if row.get("dataset") == "plism" and row.get("target") == target
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one saved PLISM/{target} probe, found {len(matches)}"
        )
    return matches[0]


def _plot_probes(probes: list[dict[str, Any]], destination: Path) -> list[str]:
    labels = ["Scanner", "Stain", "Tissue"]
    values = [float(row["balanced_accuracy"]) for row in probes]
    chances = [float(row["chance"]) for row in probes]
    if not np.isfinite(values + chances).all():
        raise RuntimeError(
            "Saved probe balanced accuracy/chance contains a non-finite value"
        )
    positions = np.arange(3)
    fig, axis = plt.subplots(figsize=(4.8, 3.6))
    bars = axis.bar(
        positions, values, width=0.58, color=("#3b82f6", "#f59e0b", "#10b981")
    )
    for position, chance in zip(positions, chances):
        axis.plot(
            [position - 0.34, position + 0.34],
            [chance, chance],
            color="#111827",
            lw=1.6,
        )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, min(1.05, max(values) + 0.14))
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Domain information remains decodable", pad=6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#d1d5db", linewidth=0.55, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[Line2D([0], [0], color="#111827", lw=1.6, label="Chance")],
        loc="upper right",
        frameon=False,
    )
    fig.tight_layout(pad=0.3)
    return _save_plot(fig, destination)


def _plot_feature_distance(values: pd.DataFrame, destination: Path) -> list[str]:
    groups = []
    for category in CONTROLLED_CATEGORIES:
        group = (
            values.loc[values.category.eq(category), "distance"]
            .dropna()
            .to_numpy(float)
        )
        if not len(group):
            raise RuntimeError(
                f"Controlled feature-distance category missing: {category}"
            )
        groups.append(group)
    fig, axis = plt.subplots(figsize=(5.4, 3.9))
    violin = axis.violinplot(groups, showextrema=False, showmedians=True, widths=0.78)
    colors = ("#3b82f6", "#10b981", "#8b5cf6")
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    boxes = axis.boxplot(groups, widths=0.16, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axis.set_xticks(range(1, 4), CONTROLLED_SHORT_LABELS)
    axis.set_ylabel("Cosine distance")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#d1d5db", linewidth=0.55, alpha=0.7)
    axis.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    return _save_plot(fig, destination)


def _read_matrix(path: Path, expected_order: list[str] | None = None) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if expected_order:
        missing = set(expected_order) - set(matrix.index)
        if missing:
            raise RuntimeError(f"MMD matrix {path} lacks domains: {sorted(missing)}")
        matrix = matrix.loc[expected_order, expected_order]
    values = matrix.to_numpy(float)
    if matrix.shape[0] != matrix.shape[1] or not np.allclose(
        values, values.T, equal_nan=False
    ):
        raise RuntimeError(f"MMD matrix is not finite symmetric square data: {path}")
    return matrix.astype(float)


def _plot_heatmap(matrix: pd.DataFrame, destination: Path, title: str) -> list[str]:
    values = matrix.to_numpy(float)
    upper = max(float(np.nanmax(values)), 1e-10)
    size = max(4.7, len(matrix) * 0.52)
    fig, axis = plt.subplots(figsize=(size + 0.55, size))
    image = axis.imshow(
        values, cmap="magma", vmin=0, vmax=upper, interpolation="nearest"
    )
    axis.set_xticks(range(len(matrix)), matrix.columns, rotation=42, ha="right")
    axis.set_yticks(range(len(matrix)), matrix.index)
    font_size = max(4.6, 8.2 - 0.22 * len(matrix))
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            value = values[row, column]
            color = "white" if value < upper * 0.47 else "black"
            axis.text(
                column,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=font_size,
                color=color,
            )
    axis.set_title(title, pad=7)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    colorbar.set_label("MMD²", rotation=90)
    fig.tight_layout(pad=0.25)
    return _save_plot(fig, destination)


def _align_features(
    samples: pd.DataFrame, cache_path: Path
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    features, sample_ids, model = _load_feature_cache(cache_path)
    positions = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "feature_position": np.arange(len(sample_ids)),
        }
    )
    aligned = positions.merge(
        samples.assign(sample_id=samples.sample_id.astype(str)),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if aligned.image_path.isna().any():
        raise RuntimeError(
            "DINOv3 cache IDs do not align one-to-one with PLISM metadata"
        )
    return aligned, features, model


def _nearest_examples(
    metadata: pd.DataFrame, features: np.ndarray, seed: int = 42
) -> list[tuple[int, list[int]]]:
    values = np.asarray(features, dtype=np.float32)
    values /= np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(len(values))[: min(len(values), 512)]
    ranked: list[tuple[tuple[int, int, float], int, list[int]]] = []
    for query in candidates:
        similarities = values @ values[query]
        similarities[query] = -np.inf
        nearest = np.argsort(-similarities)[:3].astype(int).tolist()
        query_row = metadata.iloc[int(query)]
        morphology_matches = sum(
            str(metadata.iloc[position].tissue_type) == str(query_row.tissue_type)
            for position in nearest
        )
        acquisition_changes = sum(
            str(metadata.iloc[position].stain_id) != str(query_row.stain_id)
            or str(metadata.iloc[position].scanner_id) != str(query_row.scanner_id)
            for position in nearest
        )
        ranked.append(
            (
                (
                    morphology_matches,
                    acquisition_changes,
                    float(np.mean(similarities[nearest])),
                ),
                int(query),
                nearest,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, list[int]]] = []
    used_queries: set[int] = set()
    used_tissues: set[str] = set()
    for _, query, nearest in ranked:
        tissue = str(metadata.iloc[query].tissue_type)
        if query in used_queries or (tissue in used_tissues and len(used_tissues) < 3):
            continue
        selected.append((query, nearest))
        used_queries.add(query)
        used_tissues.add(tissue)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        raise RuntimeError("Could not select three distinct nearest-neighbor examples")
    return selected


def _plot_nearest_neighbors(
    metadata: pd.DataFrame,
    examples: list[tuple[int, list[int]]],
    destination: Path,
) -> list[str]:
    fig, axes = plt.subplots(3, 4, figsize=(6.8, 5.35), squeeze=False)
    for row_number, (query, neighbors) in enumerate(examples):
        for column_number, position in enumerate([query, *neighbors]):
            row = metadata.iloc[position]
            axis = axes[row_number, column_number]
            axis.imshow(_load_rgb(row.image_path), interpolation="none")
            prefix = "QUERY" if column_number == 0 else f"NN{column_number}"
            axis.set_title(
                f"{prefix}\nT:{row.tissue_type}  S:{row.stain_id}  Q:{row.scanner_id}",
                fontsize=6.3,
                color="#991b1b" if column_number == 0 else "#111827",
                pad=2,
            )
            axis.set_axis_off()
            axis.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=0.18, h_pad=0.5)
    return _save_plot(fig, destination, vectors=False)


def _plot_number_card(numbers: dict[str, float], destination: Path) -> list[str]:
    lines = [
        f"Scanner probe BA: {numbers['scanner_probe_ba']:.3f}",
        f"Stain probe BA: {numbers['stain_probe_ba']:.3f}",
        f"Tissue probe BA: {numbers['tissue_probe_ba']:.3f}",
        "",
        f"Largest controlled scanner MMD²: {numbers['largest_scanner_mmd2']:.6f}",
        f"Largest balanced stain MMD²: {numbers['largest_stain_mmd2']:.6f}",
        "",
        "Scanner centroid vs controlled MMD:",
        f"Spearman rho = {numbers['centroid_mmd_rho']:.3f}",
    ]
    fig = plt.figure(figsize=(5.4, 3.0))
    fig.text(
        0.035,
        0.95,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.35,
    )
    return _save_plot(fig, destination)


def _sample_record(row: pd.Series) -> dict[str, Any]:
    return {
        "sample_id": str(row.sample_id),
        "aligned_group_id": str(row.aligned_group_id),
        "stain_id": str(row.stain_id),
        "scanner_id": str(row.scanner_id),
        "tissue_type": str(row.tissue_type),
        "source_file": str(Path(str(row.image_path)).resolve()),
    }


def _asset(
    filename: str,
    scientific_message: str,
    data_source: Any,
    experiment_source: Any,
    sample_ids: Any,
    metrics: Any,
    important_caveats: list[str],
    companion_files: list[str] | None = None,
) -> dict[str, Any]:
    selection_rules = {
        "01_": "Maximize scanner coverage, then median saved tissue fraction, within one aligned_group_id and stain_id.",
        "02_": "Maximize stain coverage and tissue fraction within one aligned_group_id and scanner_id; choose seven evenly spaced protocols when more are available.",
        "03_": "Largest complete stain×scanner rectangle (up to 4×4) inside one aligned_group_id, then highest tissue fraction.",
        "04_": "Direct recoloring of the saved shared PLISM UMAP coordinates by scanner.",
        "05_": "Direct recoloring of the saved shared PLISM UMAP coordinates by stain.",
        "06_": "Direct recoloring of the saved shared PLISM UMAP coordinates by tissue.",
        "07_": "Direct use of saved aligned-group-held-out PLISM probe balanced accuracy and chance values.",
        "08_": "Direct use of all saved controlled feature-distance records in the three predefined categories.",
        "09_": "Direct rendering of the saved aligned-group/stain-controlled scanner MMD² matrix.",
        "10_": "Direct rendering of the saved tissue/scanner-balanced stain MMD² matrix.",
        "11_": "Unconstrained cosine retrieval from the saved feature cache; three queries selected deterministically for morphology and acquisition diversity.",
        "12_": "Direct extraction of exact probe, MMD-extreme, and correlation values from saved metrics.json files.",
    }

    records: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            if "sample_id" in value:
                records.append(value)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(sample_ids)
    aligned_groups = sorted(
        {
            str(record["aligned_group_id"])
            for record in records
            if "aligned_group_id" in record
        }
    )
    stains = sorted(
        {str(record["stain_id"]) for record in records if "stain_id" in record},
        key=_natural_key,
    )
    scanners = sorted(
        {str(record["scanner_id"]) for record in records if "scanner_id" in record},
        key=_scanner_key,
    )
    if isinstance(sample_ids, dict):
        if not aligned_groups and sample_ids.get("aligned_group_id") is not None:
            aligned_groups = [str(sample_ids["aligned_group_id"])]
        stains = stains or [
            str(value)
            for value in sample_ids.get(
                "stain_ids",
                [sample_ids["stain_id"]] if "stain_id" in sample_ids else [],
            )
        ]
        scanners = scanners or [
            str(value)
            for value in sample_ids.get(
                "scanner_ids",
                [sample_ids["scanner_id"]] if "scanner_id" in sample_ids else [],
            )
        ]
    metric_suffixes = {".csv", ".json", ".parquet"}
    source_metric_file = None
    if (
        isinstance(data_source, str)
        and Path(data_source).suffix.lower() in metric_suffixes
    ) or (
        isinstance(data_source, list)
        and data_source
        and all(
            isinstance(value, str) and Path(value).suffix.lower() in metric_suffixes
            for value in data_source
        )
    ):
        source_metric_file = data_source
    prefix = filename[:3]
    return {
        "filename": filename,
        "companion_files": companion_files or [],
        "scientific_message": scientific_message,
        "dataset": "PLISM",
        "source_experiment": experiment_source,
        "source_metric_file": source_metric_file,
        "source_image_ids": (
            [str(record["sample_id"]) for record in records] if records else sample_ids
        ),
        "aligned_group_id": aligned_groups or None,
        "stain_ids": stains,
        "scanner_ids": scanners,
        "checkpoint": None,
        "selection_rule": selection_rules[prefix],
        "data_source": data_source,
        "experiment_source": experiment_source,
        "sample_ids": sample_ids,
        "metrics": metrics,
        "important_caveats": important_caveats,
    }


def _write_readme(
    destination: Path,
    assets: list[dict[str, Any]],
    audit_dir: Path,
    mmd_dir: Path,
    selected_groups: dict[str, str],
) -> None:
    descriptions = "\n".join(
        f"- `{item['filename']}` — {item['scientific_message']}" for item in assets
    )
    text = f"""# Motivation source assets

These are clean source panels for downstream figure composition. They are not a final motivation figure. Pathology pixels retain their original RGB colors; no stain normalization or artificial recoloring was applied.

## Assets

{descriptions}

## Provenance

- Frozen-DINOv3 domain audit: `{audit_dir.resolve()}`
- Controlled/balanced MMD analysis: `{mmd_dir.resolve()}`
- Same-tissue scanner group: `{selected_groups["scanner_strip"]}`
- Aligned/serial stain group: `{selected_groups["stain_strip"]}`
- Raw domain grid group: `{selected_groups["domain_grid"]}`
- The scanner, stain, and tissue UMAP panels use exactly the same saved coordinates and axis limits. Coordinates were not recomputed.
- Probe bars and all number-card values are loaded from the completed experiment `metrics.json` files.
- Nearest neighbors are retrieved from the existing frozen-DINOv3 cache; this is an inexpensive read-only analysis, not feature re-extraction.

## Interpretation limits

- PLISM scanner strips fix aligned morphology and stain, but residual registration error can contribute.
- PLISM stain conditions may be serial sections. Asset 02 therefore says “Aligned/serial tissue sections,” never “exact same cells.”
- Balanced stain MMD reduces tissue/scanner composition confounding where supported, but serial-section morphology and any recorded minimum-mismatch fallback pairs remain limitations.
- UMAP is descriptive. The group-held-out probes and controlled distance/MMD analyses are stronger evidence for decodable, systematic acquisition signatures.
- MMD² is a distribution statistic tied to the saved kernel and sampling protocol; do not compare it as though it were a per-image distance.
"""
    destination.write_text(text, encoding="utf-8")


def _validate_outputs(
    output_dir: Path, expected_pngs: list[str]
) -> dict[str, list[int]]:
    dimensions: dict[str, list[int]] = {}
    for name in expected_pngs:
        path = output_dir / name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            # Wide pathology strips are intentionally compact source panels.
            # At 300 dpi, 500 px still supplies more than 1.6 inches of height.
            if min(width, height) < 500:
                raise RuntimeError(
                    f"Rendered source asset is unexpectedly small/blurry: {path} ({width}×{height})"
                )
            dimensions[name] = [width, height]
    return dimensions


def main() -> None:
    global DPI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/paper_assets.yaml")
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        help="Override the completed M-1 domain-audit directory.",
    )
    parser.add_argument(
        "--mmd-dir",
        type=Path,
        help="Override the completed M-1 MMD directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the motivation source-asset output directory.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite moved PLISM image roots; may be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expected I/O without reading data or generating assets.",
    )
    args = parser.parse_args()
    path_maps: list[tuple[str, str]] = []
    for value in args.path_map:
        if "=" not in value:
            parser.error(f"--path-map must be OLD=NEW, received {value!r}")
        path_maps.append(tuple(value.split("=", 1)))

    config, configured_paths = load_paper_asset_config(
        args.config, project_root=args.project_root
    )
    DPI = int(config.get("style", {}).get("dpi", 300))
    if DPI < 300:
        raise ValueError("Paper asset DPI must be at least 300")
    audit_dir = (args.audit_dir or configured_paths.motivation_audit).resolve()
    mmd_dir = (args.mmd_dir or configured_paths.motivation_mmd).resolve()
    output_dir = (args.output_dir or configured_paths.motivation_output).resolve()
    metadata_dir = audit_dir / "metadata"
    feature_path = audit_dir / "features" / "plism_dinov3.npz"
    sample_path = metadata_dir / "plism_samples.parquet"
    projection_path = metadata_dir / "feature_projections.parquet"
    distance_path = metadata_dir / "plism_controlled_feature_distances.csv"
    audit_metrics_path = audit_dir / "metrics.json"
    mmd_metrics_path = mmd_dir / "metrics.json"
    scanner_matrix_path = mmd_dir / "tables" / "scanner_mmd_controlled.csv"
    stain_matrix_path = mmd_dir / "tables" / "stain_mmd_balanced.csv"
    required_inputs = (
        sample_path,
        projection_path,
        distance_path,
        feature_path,
        audit_metrics_path,
        mmd_metrics_path,
        scanner_matrix_path,
        stain_matrix_path,
    )
    if args.dry_run:
        print_dry_run(
            "motivation source assets",
            args.config,
            required_inputs,
            output_dir,
            MOTIVATION_OUTPUT_FILES,
        )
        return

    _configure_style()
    _require(required_inputs)
    audit_metrics, mmd_metrics = (
        _read_json(audit_metrics_path),
        _read_json(mmd_metrics_path),
    )
    _assert_full_run(audit_metrics, audit_metrics_path)
    _assert_full_run(mmd_metrics, mmd_metrics_path)

    samples = _resolve_image_paths(pd.read_parquet(sample_path), path_maps)
    required_columns = {
        "sample_id",
        "image_path",
        "aligned_group_id",
        "stain_id",
        "scanner_id",
        "tissue_type",
    }
    missing_columns = required_columns - set(samples.columns)
    if missing_columns:
        raise RuntimeError(
            f"PLISM sample manifest lacks columns: {sorted(missing_columns)}"
        )
    _validate_image_rows(samples, "PLISM sample manifest")
    output_dir.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    created: list[str] = []

    scanner_rows = _choose_scanner_strip(samples)
    scanner_group = str(scanner_rows.iloc[0].aligned_group_id)
    scanner_files = _plot_strip(
        scanner_rows,
        "scanner_id",
        output_dir / "01_same_tissue_across_scanners.png",
        "Same tissue, same stain — different scanners",
    )
    created.extend(scanner_files)
    assets.append(
        _asset(
            scanner_files[0],
            "The same aligned tissue field under one stain changes visibly across scanners.",
            [_sample_record(row) for _, row in scanner_rows.iterrows()],
            str(audit_dir.resolve()),
            {
                "aligned_group_id": scanner_group,
                "stain_id": str(scanner_rows.iloc[0].stain_id),
                "tissue_type": str(scanner_rows.iloc[0].tissue_type),
                "scanner_ids": scanner_rows.scanner_id.astype(str).tolist(),
                "samples": [_sample_record(row) for _, row in scanner_rows.iterrows()],
            },
            {},
            [
                "Aligned morphology is registration-based; residual registration error can remain."
            ],
        )
    )

    stain_rows = _choose_stain_strip(samples)
    stain_group = str(stain_rows.iloc[0].aligned_group_id)
    stain_files = _plot_strip(
        stain_rows,
        "stain_id",
        output_dir / "02_aligned_tissue_across_stains.png",
        "Aligned/serial tissue sections — different H&E protocols",
    )
    created.extend(stain_files)
    assets.append(
        _asset(
            stain_files[0],
            "Aligned or serial tissue sections show protocol-dependent H&E appearance at a fixed scanner.",
            [_sample_record(row) for _, row in stain_rows.iterrows()],
            str(audit_dir.resolve()),
            {
                "aligned_group_id": stain_group,
                "scanner_id": str(stain_rows.iloc[0].scanner_id),
                "tissue_type": str(stain_rows.iloc[0].tissue_type),
                "stain_ids": stain_rows.stain_id.astype(str).tolist(),
                "samples": [_sample_record(row) for _, row in stain_rows.iterrows()],
            },
            {},
            [
                "Stain conditions may be serial sections; this panel does not claim exact same cells."
            ],
        )
    )

    grid_rows, grid_stains, grid_scanners = _choose_domain_grid(samples)
    grid_group = str(grid_rows.iloc[0].aligned_group_id)
    grid_files = _plot_domain_grid(
        grid_rows, grid_stains, grid_scanners, output_dir / "03_raw_domain_examples.png"
    )
    created.extend(grid_files)
    assets.append(
        _asset(
            grid_files[0],
            "Real PLISM patches show joint stain/scanner appearance variability while aligned-group biology is held similar.",
            [_sample_record(row) for _, row in grid_rows.iterrows()],
            str(audit_dir.resolve()),
            {
                "aligned_group_id": grid_group,
                "stain_ids": grid_stains,
                "scanner_ids": grid_scanners,
                "samples": [_sample_record(row) for _, row in grid_rows.iterrows()],
            },
            {},
            ["Rows across stain protocols may represent serial sections."],
        )
    )

    projections = pd.read_parquet(projection_path)
    projections = projections[projections.dataset.astype(str).eq("plism")]
    umap = projections[["sample_id", "umap_1", "umap_2"]].merge(
        samples[["sample_id", "scanner_id", "stain_id", "tissue_type"]].assign(
            sample_id=samples.sample_id.astype(str)
        ),
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(umap) != len(samples):
        raise RuntimeError(
            "Saved PLISM UMAP coordinates do not align with the complete feature sample manifest"
        )
    x_pad = max(float(np.ptp(umap.umap_1)) * 0.035, 1e-6)
    y_pad = max(float(np.ptp(umap.umap_2)) * 0.035, 1e-6)
    limits = (
        (float(umap.umap_1.min() - x_pad), float(umap.umap_1.max() + x_pad)),
        (float(umap.umap_2.min() - y_pad), float(umap.umap_2.max() + y_pad)),
    )
    coordinate_hash = hashlib.sha256(
        umap.sort_values("sample_id")[["sample_id", "umap_1", "umap_2"]]
        .to_csv(index=False, float_format="%.9g")
        .encode("utf-8")
    ).hexdigest()
    factor_specs = (
        (
            "scanner_id",
            "04_dinov3_umap_scanner.png",
            sorted(umap.scanner_id.astype(str).unique(), key=_scanner_key),
            True,
            "scanner",
        ),
        (
            "stain_id",
            "05_dinov3_umap_stain.png",
            sorted(umap.stain_id.astype(str).unique(), key=_natural_key),
            True,
            "stain",
        ),
        (
            "tissue_type",
            "06_dinov3_umap_tissue.png",
            sorted(umap.tissue_type.astype(str).unique(), key=_natural_key),
            False,
            "tissue",
        ),
    )
    for factor, name, order, legend, short in factor_specs:
        files = _plot_umap(umap, factor, output_dir / name, limits, order, legend)
        created.extend(files)
        assets.append(
            _asset(
                files[0],
                f"Frozen-DINOv3 global geometry recolored by {short}; this shares coordinates and limits with the other two UMAP panels.",
                str(projection_path.resolve()),
                str(audit_dir.resolve()),
                umap.sample_id.astype(str).tolist(),
                {
                    "n_samples": len(umap),
                    "n_classes": len(order),
                    "umap_coordinate_sha256": coordinate_hash,
                    "x_limits": list(limits[0]),
                    "y_limits": list(limits[1]),
                },
                [
                    "UMAP is descriptive and does not establish absence or causality of domain information."
                ],
                files[1:],
            )
        )

    probes = [
        _probe(audit_metrics, target) for target in ("scanner", "stain", "tissue")
    ]
    probe_files = _plot_probes(probes, output_dir / "07_probe_accuracy.png")
    created.extend(probe_files)
    assets.append(
        _asset(
            probe_files[0],
            "Scanner, stain, and tissue labels remain decodable from frozen-DINOv3 features under aligned-group-held-out evaluation.",
            str(audit_metrics_path.resolve()),
            str(audit_dir.resolve()),
            [],
            {
                row["target"]: {
                    "balanced_accuracy": float(row["balanced_accuracy"]),
                    "chance": float(row["chance"]),
                    "group_overlap_count": int(row["group_overlap_count"]),
                }
                for row in probes
            },
            [
                "Balanced accuracy is group-held-out and should be interpreted with the saved class-support exclusions."
            ],
            probe_files[1:],
        )
    )

    distances = pd.read_csv(distance_path)
    distance_files = _plot_feature_distance(
        distances, output_dir / "08_controlled_feature_distance.png"
    )
    created.extend(distance_files)
    assets.append(
        _asset(
            distance_files[0],
            "Scanner-induced feature movement is smaller than morphology-induced movement but remains non-zero and systematic.",
            str(distance_path.resolve()),
            str(audit_dir.resolve()),
            sorted(
                set(distances.left_sample_id.astype(str))
                | set(distances.right_sample_id.astype(str))
            ),
            {
                category: {
                    "n": int((distances.category == category).sum()),
                    "mean_cosine_distance": float(
                        distances.loc[distances.category == category, "distance"].mean()
                    ),
                    "median_cosine_distance": float(
                        distances.loc[
                            distances.category == category, "distance"
                        ].median()
                    ),
                }
                for category in CONTROLLED_CATEGORIES
            },
            [
                "The scanner comparison fixes aligned morphology and stain; residual registration error can contribute."
            ],
            distance_files[1:],
        )
    )

    scanner_order = [str(value) for value in mmd_metrics["domains"]["scanners"]]
    scanner_order = sorted(scanner_order, key=_scanner_key)
    stain_order = sorted(
        [str(value) for value in mmd_metrics["domains"]["stains"]], key=_natural_key
    )
    scanner_matrix = _read_matrix(scanner_matrix_path, scanner_order)
    stain_matrix = _read_matrix(stain_matrix_path, stain_order)
    scanner_mmd_files = _plot_heatmap(
        scanner_matrix,
        output_dir / "09_scanner_mmd_controlled.png",
        "Controlled scanner distribution shift",
    )
    created.extend(scanner_mmd_files)
    assets.append(
        _asset(
            scanner_mmd_files[0],
            "Aligned-morphology/stain-controlled MMD² identifies systematic scanner-dependent representation shifts.",
            str(scanner_matrix_path.resolve()),
            str(mmd_dir.resolve()),
            [],
            {
                "matrix": scanner_matrix.to_dict(),
                "largest_pair": mmd_metrics["extrema"]["controlled_scanner_largest"],
            },
            [
                "MMD² follows the saved multi-kernel estimator and controlled sampling protocol."
            ],
            scanner_mmd_files[1:],
        )
    )
    stain_mmd_files = _plot_heatmap(
        stain_matrix,
        output_dir / "10_stain_mmd_balanced.png",
        "Balanced stain-condition distribution shift",
    )
    created.extend(stain_mmd_files)
    assets.append(
        _asset(
            stain_mmd_files[0],
            "Tissue/scanner-balanced MMD² identifies systematic stain-condition representation shifts.",
            str(stain_matrix_path.resolve()),
            str(mmd_dir.resolve()),
            [],
            {
                "matrix": stain_matrix.to_dict(),
                "largest_pair": mmd_metrics["extrema"]["balanced_stain_largest"],
                "fallback_pairs": mmd_metrics.get("stain_balance_fallback_pairs", []),
            },
            [
                "Serial-section morphology remains a limitation.",
                "Minimum-mismatch fallback pairs, if listed, retain residual composition confounding.",
            ],
            stain_mmd_files[1:],
        )
    )

    aligned_metadata, features, feature_model = _align_features(samples, feature_path)
    aligned_metadata = _resolve_image_paths(aligned_metadata, path_maps)
    nn_examples = _nearest_examples(aligned_metadata, features)
    nn_files = _plot_nearest_neighbors(
        aligned_metadata, nn_examples, output_dir / "11_nearest_neighbor_examples.png"
    )
    created.extend(nn_files)
    values = features / np.clip(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None
    )
    nn_records = []
    for query, neighbors in nn_examples:
        nn_records.append(
            {
                "query": _sample_record(aligned_metadata.iloc[query]),
                "neighbors": [
                    {
                        **_sample_record(aligned_metadata.iloc[position]),
                        "cosine_similarity": float(values[query] @ values[position]),
                    }
                    for position in neighbors
                ],
            }
        )
    assets.append(
        _asset(
            nn_files[0],
            "Frozen-DINOv3 nearest neighbors often retain tissue morphology across acquisition-label changes.",
            {"feature_cache": str(feature_path.resolve()), "model": feature_model},
            str(audit_dir.resolve()),
            nn_records,
            {
                "retrieval": "unconstrained cosine nearest neighbors; three display queries selected from deterministic candidates for morphology/acquisition diversity"
            },
            [
                "Displayed examples are illustrative retrievals, not an aggregate retrieval-accuracy estimate."
            ],
        )
    )

    numbers = {
        "scanner_probe_ba": float(probes[0]["balanced_accuracy"]),
        "stain_probe_ba": float(probes[1]["balanced_accuracy"]),
        "tissue_probe_ba": float(probes[2]["balanced_accuracy"]),
        "largest_scanner_mmd2": float(
            mmd_metrics["extrema"]["controlled_scanner_largest"]["mmd2"]
        ),
        "largest_stain_mmd2": float(
            mmd_metrics["extrema"]["balanced_stain_largest"]["mmd2"]
        ),
        "centroid_mmd_rho": float(
            mmd_metrics["correlations"]["scanner_centroid_vs_controlled_mmd_spearman"][
                "rho"
            ]
        ),
    }
    if not np.isfinite(list(numbers.values())).all():
        raise RuntimeError("Key-number source metrics contain a non-finite value")
    number_files = _plot_number_card(
        numbers, output_dir / "12_motivation_key_numbers.png"
    )
    created.extend(number_files)
    assets.append(
        _asset(
            number_files[0],
            "Compact exact callout values for downstream figure composition.",
            [str(audit_metrics_path.resolve()), str(mmd_metrics_path.resolve())],
            [str(audit_dir.resolve()), str(mmd_dir.resolve())],
            [],
            numbers,
            [
                "Values are copied from completed experiment outputs without approximation."
            ],
            number_files[1:],
        )
    )

    pngs = [item["filename"] for item in assets]
    dimensions = _validate_outputs(output_dir, pngs)
    for item in assets:
        item["rendered_pixels"] = dimensions[item["filename"]]
        item["png_dpi"] = DPI
    selected_groups = {
        "scanner_strip": scanner_group,
        "stain_strip": stain_group,
        "domain_grid": grid_group,
    }
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output_dir.resolve()),
        "source_experiments": {
            "domain_audit": str(audit_dir.resolve()),
            "mmd": str(mmd_dir.resolve()),
        },
        "selected_plism_group_ids": selected_groups,
        "shared_umap_coordinate_sha256": coordinate_hash,
        "assets": assets,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_readme(output_dir / "README.md", assets, audit_dir, mmd_dir, selected_groups)
    created.extend(["manifest.json", "README.md"])

    print("files created")
    for name in created:
        print(f"  {output_dir / name}")
    print("source experiment paths")
    print(f"  {audit_dir.resolve()}")
    print(f"  {mmd_dir.resolve()}")
    print("selected PLISM group IDs")
    for label, value in selected_groups.items():
        print(f"  {label}: {value}")
    print("exact output directory")
    print(f"  {output_dir.resolve()}")


if __name__ == "__main__":
    main()
