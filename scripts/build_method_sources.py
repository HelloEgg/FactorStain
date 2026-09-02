#!/usr/bin/env python
"""Generate independent FactorStain methodology source assets.

The generator reads the exact M1 split/metrics/checkpoints.  It performs only a
small deterministic inference pass for recorded evaluation pairs whose output
pixels were not persisted by M1; it never trains a model or extracts DINOv3
features.  ``--dry-run`` reads no experiment data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageOps

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from factorstain.training.renderer import build_renderer
from factorstain.visualization.paper_assets import (
    atomic_json,
    configure_paper_style,
    load_paper_asset_config,
    method_required_inputs,
    natural_key,
    print_dry_run,
    require_inputs,
    save_figure,
)

METHOD_OUTPUT_FILES = (
    "01_image_formation_pipeline.png",
    "02_factorial_2x2_real_grid.png",
    "03_heldout_combination_matrix.png",
    "04_compositional_task_example.png",
    "05_factorstain_operator_examples.png",
    "06_joint_parallel_ordered_comparison.svg",
    "06_joint_parallel_ordered_comparison.png",
    "07_factor_isolation_examples.png",
    "08_generated_vs_real_unseen.png",
    "09_seen_vs_unseen_results.png",
    "10_method_key_equations.svg",
    "10_method_key_equations.png",
    "manifest.json",
    "README.md",
)
METHODS = ("joint", "parallel", "factorstain")
METHOD_LABELS = {"joint": "Joint", "parallel": "Parallel", "factorstain": "Ordered"}
METHOD_COLORS = {"joint": "#6b7280", "parallel": "#2563eb", "factorstain": "#d97706"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cell_set(items: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(item["stain_id"]), str(item["scanner_id"])) for item in items}


def _image_id(row: pd.Series) -> str:
    for column in ("sample_id", "image_id"):
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return f"index:{row.name}"


def _image_record(row: pd.Series) -> dict[str, str]:
    return {
        "image_id": _image_id(row),
        "aligned_group_id": str(row.aligned_group_id),
        "stain_id": str(row.stain_id),
        "scanner_id": str(row.scanner_id),
        "tissue_type": str(row.tissue_type),
        "source_file": str(Path(str(row.image_path)).resolve()),
    }


def _read_rgb(path: object, size: int | None = None) -> np.ndarray:
    with Image.open(str(path)) as opened:
        image = opened.convert("RGB")
        if size is not None:
            image = ImageOps.fit(
                image,
                (size, size),
                method=Image.Resampling.BILINEAR,
                centering=(0.5, 0.5),
            )
        return np.asarray(image)


def _quality_score(path: object) -> float:
    image = _read_rgb(path, 128).astype(np.float32)
    gray = image.mean(axis=2)
    tissue = float((gray < 242).mean())
    nuclei = float((gray < 155).mean())
    contrast = float(np.clip(gray.std() / 64.0, 0, 1))
    return 0.55 * tissue + 0.30 * nuclei + 0.15 * contrast


def _validate_index(index: pd.DataFrame) -> pd.DataFrame:
    required = {
        "image_path",
        "aligned_group_id",
        "tissue_type",
        "stain_id",
        "scanner_id",
        "morphology_split",
    }
    missing = required - set(index.columns)
    if missing:
        raise RuntimeError(f"M1 index lacks required columns: {sorted(missing)}")
    result = index.copy().reset_index(drop=True)
    for column in ("aligned_group_id", "tissue_type", "stain_id", "scanner_id"):
        result[column] = result[column].astype(str)
    return result


def _find_factorial_example(
    index: pd.DataFrame,
    split: dict[str, Any],
    max_quality_candidates: int = 256,
) -> dict[str, Any]:
    train = _cell_set(split["train_cells"])
    test = _cell_set(split["test_cells"])
    candidates: list[dict[str, Any]] = []
    for group_id, group in index.groupby("aligned_group_id", sort=True):
        lookup = {
            (str(row.stain_id), str(row.scanner_id)): row
            for _, row in group.drop_duplicates(["stain_id", "scanner_id"]).iterrows()
            if Path(str(row.image_path)).is_file()
        }
        found = None
        for stain_b, scanner_y in sorted(
            test, key=lambda item: (natural_key(item[0]), natural_key(item[1]))
        ):
            if (stain_b, scanner_y) not in lookup:
                continue
            for stain_a in sorted(
                {cell[0] for cell in train if cell[0] != stain_b}, key=natural_key
            ):
                for scanner_x in sorted(
                    {cell[1] for cell in train if cell[1] != scanner_y}, key=natural_key
                ):
                    required = {
                        (stain_a, scanner_x),
                        (stain_a, scanner_y),
                        (stain_b, scanner_x),
                    }
                    if required.issubset(train) and required.issubset(lookup):
                        found = {
                            "aligned_group_id": str(group_id),
                            "stain_a": stain_a,
                            "stain_b": stain_b,
                            "scanner_x": scanner_x,
                            "scanner_y": scanner_y,
                            "ax": lookup[(stain_a, scanner_x)],
                            "ay": lookup[(stain_a, scanner_y)],
                            "bx": lookup[(stain_b, scanner_x)],
                            "by": lookup[(stain_b, scanner_y)],
                        }
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is not None:
            candidates.append(found)
        if len(candidates) >= max_quality_candidates:
            break
    if not candidates:
        raise RuntimeError(
            "No real PLISM 2×2 group matches the saved TRAIN/HELD-OUT split. "
            "Run: bash shell/m1_factorial.sh"
        )
    for candidate in candidates:
        candidate["quality_score"] = float(
            np.mean(
                [
                    _quality_score(candidate[key].image_path)
                    for key in ("ax", "ay", "bx", "by")
                ]
            )
        )
    return max(
        candidates,
        key=lambda item: (item["quality_score"], item["aligned_group_id"]),
    )


def _select_ranked_rows(metrics: pd.DataFrame, metric: str) -> dict[str, pd.Series]:
    required = {
        "method",
        "protocol",
        "source_index",
        "target_index",
        "aligned_group_id",
        metric,
    }
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(f"M1 per-sample metrics lacks columns: {sorted(missing)}")
    frame = metrics[
        metrics.method.astype(str).eq("factorstain")
        & metrics.protocol.astype(str).str.startswith("unseen_combo")
    ].dropna(subset=[metric])
    frame = frame.drop_duplicates(["source_index", "target_index", "protocol"])
    if len(frame) < 3:
        raise RuntimeError(
            "At least three saved FactorStain unseen-combination rows are required. "
            "Run: bash shell/m1_factorial.sh"
        )
    frame = frame.sort_values(
        [metric, "aligned_group_id", "source_index", "target_index"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    middle = min(
        range(len(frame)), key=lambda position: abs(position - (len(frame) - 1) / 2)
    )
    return {"best": frame.iloc[0], "median": frame.iloc[middle], "hard": frame.iloc[-1]}


class _Renderer:
    def __init__(
        self,
        index: pd.DataFrame,
        checkpoints: dict[str, Path],
        image_size: int,
        width: int,
        device_name: str,
    ) -> None:
        self.index = index
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.width = width
        self.stains = sorted(index.stain_id.astype(str).unique())
        self.scanners = sorted(index.scanner_id.astype(str).unique())
        self.stain_map = {value: position for position, value in enumerate(self.stains)}
        self.scanner_map = {
            value: position for position, value in enumerate(self.scanners)
        }
        if device_name == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.models: dict[str, torch.nn.Module] = {}

    def _model(self, method: str) -> torch.nn.Module:
        if method not in self.models:
            model = build_renderer(
                method, len(self.stains), len(self.scanners), width=self.width
            )
            payload = torch.load(
                self.checkpoints[method], map_location=self.device, weights_only=False
            )
            model.load_state_dict(payload["model"], strict=True)
            self.models[method] = model.to(self.device).eval().requires_grad_(False)
        return self.models[method]

    def _tensor(self, row: pd.Series) -> torch.Tensor:
        with Image.open(str(row.image_path)) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if max(width, height) / max(1, min(width, height)) > 1.05:
                raise RuntimeError(
                    f"M1 source patch is not square enough for stretch-free inference: {row.image_path}"
                )
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def render(
        self,
        method: str,
        source: pd.Series,
        target_stain: str,
        target_scanner: str,
    ) -> np.ndarray:
        stain = torch.tensor([self.stain_map[str(target_stain)]], device=self.device)
        scanner = torch.tensor(
            [self.scanner_map[str(target_scanner)]], device=self.device
        )
        generated = self._model(method)(self._tensor(source), stain, scanner)[0]
        return generated.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _axis_image(
    axis: plt.Axes, image: np.ndarray, title: str, border: str | None = None
) -> None:
    axis.imshow(image, interpolation="none")
    axis.set_title(title, fontsize=8.2, pad=3)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(border is not None)
        if border is not None:
            spine.set_color(border)
            spine.set_linewidth(2.2)
    axis.set_aspect("equal", adjustable="box")


def _arrow_between(fig: plt.Figure, left: plt.Axes, right: plt.Axes) -> None:
    left_box, right_box = left.get_position(), right.get_position()
    arrow = FancyArrowPatch(
        (left_box.x1 + 0.004, (left_box.y0 + left_box.y1) / 2),
        (right_box.x0 - 0.004, (right_box.y0 + right_box.y1) / 2),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color="#374151",
    )
    fig.add_artist(arrow)


def _plot_image_formation(
    example: dict[str, Any], destination: Path, dpi: int
) -> list[Path]:
    fig, axes = plt.subplots(
        1, 4, figsize=(7.3, 1.95), gridspec_kw={"width_ratios": [1, 1, 0.72, 1]}
    )
    _axis_image(
        axes[0],
        _read_rgb(example["ax"].image_path),
        "Tissue morphology\n(aligned field)",
    )
    _axis_image(
        axes[1],
        _read_rgb(example["bx"].image_path),
        f"H&E staining\n{example['stain_b']}",
    )
    axes[2].set_axis_off()
    axes[2].add_patch(
        Rectangle(
            (0.1, 0.31),
            0.8,
            0.38,
            transform=axes[2].transAxes,
            facecolor="#f3f4f6",
            edgecolor="#4b5563",
            linewidth=1.1,
        )
    )
    axes[2].text(
        0.5,
        0.54,
        "WSI scanner",
        ha="center",
        va="center",
        fontsize=8.5,
        transform=axes[2].transAxes,
    )
    axes[2].text(
        0.5,
        0.42,
        str(example["scanner_y"]),
        ha="center",
        va="center",
        fontsize=7.5,
        transform=axes[2].transAxes,
    )
    _axis_image(axes[3], _read_rgb(example["by"].image_path), "Digital H&E")
    fig.tight_layout(pad=0.2)
    for left, right in pairwise(axes):
        _arrow_between(fig, left, right)
    return save_figure(fig, destination, dpi=dpi)


def _plot_factorial_grid(
    example: dict[str, Any], destination: Path, dpi: int
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(3.65, 3.75), squeeze=False)
    cells = (
        ("ax", "A × X"),
        ("ay", "A × Y"),
        ("bx", "B × X"),
        ("by", "B × Y\nheld out during training"),
    )
    for axis, (key, label) in zip(axes.flat, cells):
        _axis_image(
            axis,
            _read_rgb(example[key].image_path),
            label,
            "#dc2626" if key == "by" else None,
        )
    fig.text(
        0.54,
        0.985,
        f"Scanner X: {example['scanner_x']}",
        ha="right",
        va="top",
        fontsize=7.5,
    )
    fig.text(
        0.98,
        0.985,
        f"Scanner Y: {example['scanner_y']}",
        ha="right",
        va="top",
        fontsize=7.5,
    )
    fig.text(
        0.01,
        0.71,
        f"Stain A: {example['stain_a']}",
        ha="left",
        va="center",
        rotation=90,
        fontsize=7.5,
    )
    fig.text(
        0.01,
        0.27,
        f"Stain B: {example['stain_b']}",
        ha="left",
        va="center",
        rotation=90,
        fontsize=7.5,
    )
    fig.tight_layout(pad=0.25, rect=(0.04, 0, 1, 0.95))
    return save_figure(fig, destination, dpi=dpi)


def _plot_split(split: dict[str, Any], destination: Path, dpi: int) -> list[Path]:
    stains = [str(value) for value in split["stains"]]
    scanners = [str(value) for value in split["scanners"]]
    train, validation, test = (
        _cell_set(split["train_cells"]),
        _cell_set(split["validation_cells"]),
        _cell_set(split["test_cells"]),
    )
    values = np.zeros((len(stains), len(scanners)), dtype=int)
    labels = np.full(values.shape, "UNAVAILABLE", dtype=object)
    for row, stain in enumerate(stains):
        for column, scanner in enumerate(scanners):
            cell = (stain, scanner)
            if cell in train:
                values[row, column], labels[row, column] = 1, "TRAIN"
            elif cell in validation:
                values[row, column], labels[row, column] = 2, "VALIDATION"
            elif cell in test:
                values[row, column], labels[row, column] = 3, "HELD-OUT\nTEST"
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#e5e7eb", "#3b82f6", "#f59e0b", "#dc2626"])
    fig, axis = plt.subplots(
        figsize=(max(6.2, len(scanners) * 0.85), max(5.4, len(stains) * 0.43))
    )
    axis.imshow(
        values, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest"
    )
    axis.set_xticks(range(len(scanners)), scanners, rotation=38, ha="right")
    axis.set_yticks(range(len(stains)), stains)
    axis.set_xlabel("Scanner")
    axis.set_ylabel("Stain")
    fontsize = max(4.4, 7.0 - 0.12 * len(stains))
    for row in range(len(stains)):
        for column in range(len(scanners)):
            axis.text(
                column,
                row,
                labels[row, column],
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if values[row, column] else "#374151",
            )
    legend = [
        Line2D(
            [0], [0], marker="s", linestyle="", color=color, markersize=7, label=label
        )
        for color, label in (
            ("#3b82f6", "TRAIN"),
            ("#f59e0b", "VALIDATION"),
            ("#dc2626", "HELD-OUT TEST"),
            ("#e5e7eb", "UNAVAILABLE"),
        )
    ]
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(pad=0.25)
    return save_figure(fig, destination, dpi=dpi, vector_suffixes=(".svg", ".pdf"))


def _plot_composition(
    example: dict[str, Any], generated: np.ndarray, destination: Path, dpi: int
) -> list[Path]:
    fig, axes = plt.subplots(
        1, 6, figsize=(9.4, 2.05), gridspec_kw={"width_ratios": [1, 1, 1, 0.35, 1, 1]}
    )
    for axis, key, label in zip(
        axes[:3],
        ("ax", "ay", "bx"),
        ("Observed\nA × X", "Observed\nA × Y", "Observed\nB × X"),
    ):
        _axis_image(axis, _read_rgb(example[key].image_path), label)
    axes[3].set_axis_off()
    axes[3].text(
        0.5,
        0.52,
        "→",
        transform=axes[3].transAxes,
        ha="center",
        va="center",
        fontsize=20,
        color="#374151",
    )
    _axis_image(axes[4], generated, "Generated\nB × Y", "#d97706")
    _axis_image(
        axes[5], _read_rgb(example["by"].image_path), "Real held-out\nB × Y", "#111827"
    )
    fig.tight_layout(pad=0.2)
    return save_figure(fig, destination, dpi=dpi)


def _row_pair(
    index: pd.DataFrame, metric_row: pd.Series
) -> tuple[pd.Series, pd.Series]:
    source = index.iloc[int(metric_row.source_index)]
    target = index.iloc[int(metric_row.target_index)]
    if str(source.aligned_group_id) != str(target.aligned_group_id):
        raise RuntimeError("Saved M1 source/target pair crosses aligned_group_id")
    return source, target


def _plot_operator_examples(
    index: pd.DataFrame,
    selected: dict[str, pd.Series],
    selection_metric: str,
    renderer: _Renderer,
    destination: Path,
    dpi: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    fig, axes = plt.subplots(2, 5, figsize=(8.3, 3.65), squeeze=False)
    records = []
    for row_number, rank in enumerate(("best", "median")):
        metric_row = selected[rank]
        source, target = _row_pair(index, metric_row)
        images = (
            _read_rgb(source.image_path),
            renderer.render(
                "factorstain", source, str(target.stain_id), str(source.scanner_id)
            ),
            renderer.render(
                "factorstain", source, str(source.stain_id), str(target.scanner_id)
            ),
            renderer.render(
                "factorstain", source, str(target.stain_id), str(target.scanner_id)
            ),
            _read_rgb(target.image_path),
        )
        titles = ("SOURCE", "STAIN SWAP", "SCANNER SWAP", "BOTH", "REAL TARGET")
        for axis, image, title in zip(axes[row_number], images, titles):
            _axis_image(axis, image, title)
        axes[row_number, 0].text(
            -0.08,
            0.5,
            "GOOD" if rank == "best" else "MEDIAN",
            transform=axes[row_number, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=7.5,
        )
        records.append(
            {
                "rank": rank,
                "metric": float(metric_row[selection_metric]),
                "source": _image_record(source),
                "target": _image_record(target),
            }
        )
    fig.tight_layout(pad=0.2, h_pad=0.45)
    return save_figure(fig, destination, dpi=dpi), records


def _draw_box(
    axis: plt.Axes, x: float, y: float, width: float, text: str, color: str
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y), width, 0.12, facecolor=color, edgecolor="#374151", linewidth=0.8
        )
    )
    axis.text(x + width / 2, y + 0.06, text, ha="center", va="center", fontsize=7.5)


def _draw_arrow(
    axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#4b5563",
        )
    )


def _plot_architectures(destination: Path, dpi: int) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(7.3, 4.6), squeeze=False)
    pale, stain, scanner, output = "#eef2ff", "#fef3c7", "#dbeafe", "#ecfdf5"
    for axis in axes.flat:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_axis_off()
    axis = axes[0, 0]
    axis.text(0.01, 0.82, "JOINT", fontsize=8.5, weight="bold")
    for x, text, color, width in (
        (0.08, "Input /\nmorphology", pale, 0.17),
        (0.38, "Target stain +\ntarget scanner", "#f3e8ff", 0.20),
        (0.68, "Joint generator", output, 0.18),
    ):
        _draw_box(axis, x, 0.38, width, text, color)
    _draw_arrow(axis, (0.25, 0.44), (0.38, 0.44))
    _draw_arrow(axis, (0.58, 0.44), (0.68, 0.44))
    _draw_arrow(axis, (0.86, 0.44), (0.96, 0.44))
    axis = axes[1, 0]
    axis.text(0.01, 0.82, "PARALLEL", fontsize=8.5, weight="bold")
    _draw_box(axis, 0.08, 0.38, 0.18, "Morphology", pale)
    _draw_box(axis, 0.40, 0.57, 0.17, "Stain factor", stain)
    _draw_box(axis, 0.40, 0.19, 0.17, "Scanner factor", scanner)
    _draw_box(axis, 0.72, 0.38, 0.18, "Parallel fusion", output)
    _draw_arrow(axis, (0.26, 0.44), (0.40, 0.63))
    _draw_arrow(axis, (0.26, 0.44), (0.40, 0.25))
    _draw_arrow(axis, (0.57, 0.63), (0.72, 0.47))
    _draw_arrow(axis, (0.57, 0.25), (0.72, 0.41))
    _draw_arrow(axis, (0.90, 0.44), (0.97, 0.44))
    axis = axes[2, 0]
    axis.text(0.01, 0.82, "ORDERED FACTORSTAIN", fontsize=8.5, weight="bold")
    for x, text, color, width in (
        (0.08, "Morphology", pale, 0.17),
        (0.36, "Stain Operator", stain, 0.19),
        (0.66, "Scanner Operator", scanner, 0.20),
    ):
        _draw_box(axis, x, 0.38, width, text, color)
    _draw_arrow(axis, (0.25, 0.44), (0.36, 0.44))
    _draw_arrow(axis, (0.55, 0.44), (0.66, 0.44))
    _draw_arrow(axis, (0.86, 0.44), (0.97, 0.44))
    fig.tight_layout(pad=0.2)
    return save_figure(
        fig, destination.with_suffix(".png"), dpi=dpi, vector_suffixes=(".svg",)
    )


def _real_intervention_target(
    index: pd.DataFrame, source: pd.Series, stain: str, scanner: str
) -> pd.Series:
    match = index[
        index.aligned_group_id.astype(str).eq(str(source.aligned_group_id))
        & index.stain_id.astype(str).eq(str(stain))
        & index.scanner_id.astype(str).eq(str(scanner))
    ]
    if len(match) != 1:
        raise RuntimeError(
            "Factor-isolation source lacks an exact real aligned intervention target"
        )
    return match.iloc[0]


def _plot_factor_isolation(
    index: pd.DataFrame,
    metric_row: pd.Series,
    renderer: _Renderer,
    destination: Path,
    dpi: int,
) -> tuple[list[Path], dict[str, Any]]:
    source, target = _row_pair(index, metric_row)
    scanner_real = _real_intervention_target(
        index, source, str(source.stain_id), str(target.scanner_id)
    )
    stain_real = _real_intervention_target(
        index, source, str(target.stain_id), str(source.scanner_id)
    )
    fig, axes = plt.subplots(2, 3, figsize=(5.2, 3.55), squeeze=False)
    rows = (
        (
            "Scanner swap",
            source,
            renderer.render(
                "factorstain", source, str(source.stain_id), str(target.scanner_id)
            ),
            scanner_real,
        ),
        (
            "Stain swap",
            source,
            renderer.render(
                "factorstain", source, str(target.stain_id), str(source.scanner_id)
            ),
            stain_real,
        ),
    )
    for row_number, (label, origin, generated, real) in enumerate(rows):
        for axis, image, title in zip(
            axes[row_number],
            (_read_rgb(origin.image_path), generated, _read_rgb(real.image_path)),
            ("SOURCE", "GENERATED", "REAL ACQUISITION"),
        ):
            _axis_image(axis, image, title)
        axes[row_number, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[row_number, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=7.5,
        )
    fig.tight_layout(pad=0.2, h_pad=0.45)
    record = {
        "source": _image_record(source),
        "scanner_swap_real": _image_record(scanner_real),
        "stain_swap_real": _image_record(stain_real),
        "factor_isolation_score": float(metric_row.factor_isolation_score),
    }
    return save_figure(fig, destination, dpi=dpi), record


def _select_isolation_row(index: pd.DataFrame, metrics: pd.DataFrame) -> pd.Series:
    candidates = metrics[
        metrics.method.astype(str).eq("factorstain")
        & metrics.protocol.astype(str).str.startswith("unseen_combo")
    ].dropna(subset=["factor_isolation_score"])
    candidates = candidates.sort_values(
        ["factor_isolation_score", "aligned_group_id"], ascending=[False, True]
    )
    for _, row in candidates.iterrows():
        source, target = _row_pair(index, row)
        scanner_match = index[
            index.aligned_group_id.astype(str).eq(str(source.aligned_group_id))
            & index.stain_id.astype(str).eq(str(source.stain_id))
            & index.scanner_id.astype(str).eq(str(target.scanner_id))
        ]
        stain_match = index[
            index.aligned_group_id.astype(str).eq(str(source.aligned_group_id))
            & index.stain_id.astype(str).eq(str(target.stain_id))
            & index.scanner_id.astype(str).eq(str(source.scanner_id))
        ]
        if len(scanner_match) == 1 and len(stain_match) == 1:
            return row
    raise RuntimeError(
        "No saved unseen M1 row has both exact aligned real one-factor targets"
    )


def _plot_model_comparison(
    index: pd.DataFrame,
    selected: dict[str, pd.Series],
    selection_metric: str,
    renderer: _Renderer,
    destination: Path,
    dpi: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    fig, axes = plt.subplots(3, 5, figsize=(8.3, 5.25), squeeze=False)
    records = []
    for row_number, rank in enumerate(("best", "median", "hard")):
        metric_row = selected[rank]
        source, target = _row_pair(index, metric_row)
        images = [_read_rgb(source.image_path)]
        images.extend(
            renderer.render(
                method, source, str(target.stain_id), str(target.scanner_id)
            )
            for method in METHODS
        )
        images.append(_read_rgb(target.image_path))
        for axis, image, title in zip(
            axes[row_number],
            images,
            ("SOURCE", "JOINT", "PARALLEL", "ORDERED", "REAL TARGET"),
        ):
            _axis_image(axis, image, title)
        axes[row_number, 0].text(
            -0.08,
            0.5,
            rank.upper(),
            transform=axes[row_number, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=7.5,
        )
        records.append(
            {
                "rank": rank,
                "metric": float(metric_row[selection_metric]),
                "source": _image_record(source),
                "target": _image_record(target),
            }
        )
    fig.tight_layout(pad=0.2, h_pad=0.45)
    return save_figure(fig, destination, dpi=dpi), records


def _plot_seen_unseen(
    metrics: pd.DataFrame, destination: Path, dpi: int
) -> tuple[list[Path], dict[str, Any]]:
    metric_specs = (
        ("scanner_target_accuracy", "Scanner fidelity"),
        ("stain_target_accuracy", "Stain fidelity"),
        ("morphology_preservation", "Morphology"),
        ("factor_isolation_score", "Factor isolation"),
    )
    required = {"method", "protocol", *(name for name, _ in metric_specs)}
    missing = required - set(metrics.columns)
    if missing:
        raise RuntimeError(f"M1 metrics lacks seen/unseen columns: {sorted(missing)}")
    frame = metrics[metrics.method.astype(str).isin(METHODS)].copy()
    frame["combination"] = np.where(
        frame.protocol.astype(str).str.startswith("seen_combo"),
        "Seen",
        np.where(
            frame.protocol.astype(str).str.startswith("unseen_combo"), "Unseen", "Other"
        ),
    )
    frame = frame[frame.combination.isin(["Seen", "Unseen"])]
    if set(frame.combination) != {"Seen", "Unseen"}:
        raise RuntimeError(
            "Saved M1 metrics do not contain both seen and unseen protocols"
        )
    summary = frame.groupby(["method", "combination"])[
        [name for name, _ in metric_specs]
    ].mean()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.2), squeeze=False)
    positions = np.arange(len(METHODS))
    width = 0.34
    for axis, (metric, label) in zip(axes.flat, metric_specs):
        seen = [float(summary.loc[(method, "Seen"), metric]) for method in METHODS]
        unseen = [float(summary.loc[(method, "Unseen"), metric]) for method in METHODS]
        colors = [METHOD_COLORS[method] for method in METHODS]
        axis.bar(
            positions - width / 2, seen, width, color=colors, alpha=0.55, label="Seen"
        )
        axis.bar(
            positions + width / 2,
            unseen,
            width,
            color=colors,
            hatch="//",
            edgecolor=colors,
            linewidth=0.7,
            label="Unseen",
        )
        axis.set_xticks(
            positions, [METHOD_LABELS[method] for method in METHODS], rotation=18
        )
        axis.set_ylim(0, 1.02)
        axis.set_title(label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#d1d5db", linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#9ca3af", alpha=0.55, label="Seen"),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="#6b7280",
            hatch="//",
            label="Unseen",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1))
    payload = {
        method: {
            combination.lower(): {
                metric: float(summary.loc[(method, combination), metric])
                for metric, _ in metric_specs
            }
            for combination in ("Seen", "Unseen")
        }
        for method in METHODS
    }
    return save_figure(
        fig, destination, dpi=dpi, vector_suffixes=(".svg", ".pdf")
    ), payload


def _plot_equations(destination: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(6.8, 2.0))
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_axis_off()
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            facecolor="white",
            edgecolor="none",
        )
    )
    axis.text(
        0.04,
        0.76,
        "Ordered FactorStain",
        fontsize=9,
        color="#374151",
        transform=axis.transAxes,
    )
    axis.text(
        0.04,
        0.53,
        r"$\hat{x}_{(s,q)} = Q_q\!\left(S_s\!\left(E(x)\right)\right)$",
        fontsize=20,
        transform=axis.transAxes,
    )
    axis.text(
        0.55,
        0.76,
        "Compositional task",
        fontsize=9,
        color="#374151",
        transform=axis.transAxes,
    )
    axis.text(
        0.55,
        0.53,
        r"$(A,X),\ (A,Y),\ (B,X)\ \rightarrow\ (B,Y)$",
        fontsize=17,
        transform=axis.transAxes,
    )
    return save_figure(
        fig, destination.with_suffix(".png"), dpi=dpi, vector_suffixes=(".svg",)
    )


def _entry(
    filename: str,
    message: str,
    source_experiment: Path,
    source_metric_file: Path | None,
    image_records: Any,
    groups: Any,
    stains: Any,
    scanners: Any,
    checkpoint: Any,
    selection_rule: str,
    caveats: list[str],
    companion_files: list[str] | None = None,
    metrics: Any = None,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "companion_files": companion_files or [],
        "scientific_message": message,
        "dataset": "PLISM",
        "source_experiment": str(source_experiment.resolve()),
        "source_metric_file": str(source_metric_file.resolve())
        if source_metric_file
        else None,
        "source_image_ids": image_records,
        "aligned_group_id": groups,
        "stain_ids": stains,
        "scanner_ids": scanners,
        "checkpoint": checkpoint,
        "selection_rule": selection_rule,
        "metrics": metrics or {},
        "important_caveats": caveats,
    }


def _validate_outputs(output: Path) -> dict[str, list[int]]:
    dimensions = {}
    for name in METHOD_OUTPUT_FILES:
        path = output / name
        if not path.exists():
            raise RuntimeError(f"Method asset was not created: {path}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if min(image.size) < 500:
                    raise RuntimeError(
                        f"Method asset is unexpectedly small: {path} {image.size}"
                    )
                dimensions[name] = list(image.size)
        elif path.suffix == ".svg":
            ElementTree.parse(path)
    return dimensions


def _write_readme(output: Path, m1: Path, assets: list[dict[str, Any]]) -> None:
    listing = "\n".join(
        f"- `{item['filename']}` — {item['scientific_message']}" for item in assets
    )
    text = f"""# Methodology source assets

These files are independent source panels for downstream FigureLabs composition. They are not a final CVPR method figure.

## Assets

{listing}

## Provenance and selection

- M1 source experiment: `{m1.resolve()}`
- The 13×7 matrix is read from the existing saved combination split; this generator never creates a new split.
- Good/median/hard examples are selected deterministically from `tables/per_sample_metrics.csv` using the configured saved metric.
- Generated pixels come from a small inference-only pass through the exact saved M1 checkpoints for the selected recorded source/target indices. No training, feature extraction, or metric recomputation occurs.
- Pathology images retain RGB color. The renderer uses the same saved M1 image size and label ordering.

## Interpretation limits

- PLISM stain protocols may use serial sections, so cross-stain targets are not pixel-perfect identical tissue.
- Joint, Parallel, and Ordered are shown neutrally. Asset 09 reports the saved measurements and does not assert that Ordered is superior.
- Best/median/hard ranking uses one predefined morphology-preservation metric and must not be read as a complete performance ranking.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/paper_assets.yaml")
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--m1-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, paths = load_paper_asset_config(args.config, args.project_root)
    method_config = config.get("method", {})
    dpi = int(config.get("style", {}).get("dpi", 300))
    if dpi < 300:
        raise ValueError("Paper asset DPI must be at least 300")
    seed = int(method_config.get("seed", 42))
    m1 = (args.m1_dir or paths.method_m1).resolve()
    output = (args.output_dir or paths.method_output).resolve()
    required = method_required_inputs(paths, seed)
    if m1 != paths.method_m1:
        required = [m1 / path.relative_to(paths.method_m1) for path in required]
    if args.dry_run:
        print_dry_run(
            "methodology source assets",
            args.config,
            required,
            output,
            METHOD_OUTPUT_FILES,
        )
        return

    require_inputs(required, "bash shell/m1_factorial.sh")
    metrics_payload = _read_json(m1 / "metrics.json")
    if (
        metrics_payload.get("fast_dev_run")
        or metrics_payload.get("decision_valid") is False
    ):
        raise RuntimeError("Refusing FAST_DEV_RUN or invalid M1 scientific outputs")
    split_path = m1 / "splits" / f"combination_split_seed{seed}.json"
    split = _read_json(split_path)
    index_path = m1 / "metadata" / "plism_index_with_splits.parquet"
    index = _validate_index(pd.read_parquet(index_path))
    metric_path = m1 / "tables" / "per_sample_metrics.csv"
    metrics = pd.read_csv(metric_path)
    selection_metric = str(
        method_config.get("selection_metric", "morphology_preservation")
    )
    ranked = _select_ranked_rows(metrics, selection_metric)
    example = _find_factorial_example(index, split)

    m1_config_path = m1 / "config_resolved.yaml"
    with m1_config_path.open("r", encoding="utf-8") as handle:
        m1_config = yaml.safe_load(handle) or {}
    checkpoints = {
        method: m1 / "checkpoints" / method / "best.pt" for method in METHODS
    }
    renderer = _Renderer(
        index,
        checkpoints,
        int(m1_config.get("image_size", 256)),
        int(m1_config.get("model_width", 64)),
        str(method_config.get("inference_device", "auto")),
    )
    configure_paper_style(dpi)
    output.mkdir(parents=True, exist_ok=True)
    assets = []

    formation_files = _plot_image_formation(
        example, output / METHOD_OUTPUT_FILES[0], dpi
    )
    grid_files = _plot_factorial_grid(example, output / METHOD_OUTPUT_FILES[1], dpi)
    split_files = _plot_split(split, output / METHOD_OUTPUT_FILES[2], dpi)
    generated_by = renderer.render(
        "factorstain", example["ax"], example["stain_b"], example["scanner_y"]
    )
    composition_files = _plot_composition(
        example, generated_by, output / METHOD_OUTPUT_FILES[3], dpi
    )
    example_records = [_image_record(example[key]) for key in ("ax", "ay", "bx", "by")]
    shared = {
        "groups": example["aligned_group_id"],
        "stains": [example["stain_a"], example["stain_b"]],
        "scanners": [example["scanner_x"], example["scanner_y"]],
    }
    assets.extend(
        [
            _entry(
                formation_files[0].name,
                "Pathology acquisition follows morphology, staining, scanning, and digital image formation.",
                m1,
                None,
                example_records,
                **shared,
                checkpoint=None,
                selection_rule="Highest tissue/nuclei/contrast score among the first 256 complete real held-out 2×2 candidates.",
                caveats=[
                    "The strip is explanatory; displayed real images are aligned acquisition observations, not physically observed intermediate states."
                ],
            ),
            _entry(
                grid_files[0].name,
                "A real PLISM 2×2 grid makes the held-out acquisition cell concrete.",
                m1,
                split_path,
                example_records,
                **shared,
                checkpoint=None,
                selection_rule="Complete real 2×2 with three TRAIN cells and one HELD-OUT TEST cell; ranked by image informativeness.",
                caveats=["Cross-stain cells may be serial sections."],
            ),
            _entry(
                split_files[0].name,
                "The exact global M1 split holds out stain×scanner combinations while retaining individual factors.",
                m1,
                split_path,
                [],
                None,
                split["stains"],
                split["scanners"],
                None,
                "Direct rendering of the saved combination split; no split recomputation.",
                ["Unavailable means absent from the observed PLISM acquisition grid."],
                [path.name for path in split_files[1:]],
            ),
            _entry(
                composition_files[0].name,
                "Three observed acquisition cells define the unseen B×Y composition task and its generated/real comparison.",
                m1,
                metric_path,
                example_records,
                **shared,
                checkpoint=str(checkpoints["factorstain"].resolve()),
                selection_rule="Same complete real held-out 2×2 as asset 02; B×Y generated by inference from A×X.",
                caveats=[
                    "Generated output is checkpoint inference, not a stored ground-truth intermediate."
                ],
            ),
        ]
    )

    operator_files, operator_records = _plot_operator_examples(
        index,
        ranked,
        selection_metric,
        renderer,
        output / METHOD_OUTPUT_FILES[4],
        dpi,
    )
    assets.append(
        _entry(
            operator_files[0].name,
            "Ordered FactorStain exposes stain-only, scanner-only, and joint interventions for representative M1 pairs.",
            m1,
            metric_path,
            operator_records,
            [row["source"]["aligned_group_id"] for row in operator_records],
            sorted(
                {row["source"]["stain_id"] for row in operator_records}
                | {row["target"]["stain_id"] for row in operator_records},
                key=natural_key,
            ),
            sorted(
                {row["source"]["scanner_id"] for row in operator_records}
                | {row["target"]["scanner_id"] for row in operator_records},
                key=natural_key,
            ),
            str(checkpoints["factorstain"].resolve()),
            f"Good and median rows under saved {selection_metric}; deterministic tie-breaking.",
            ["Good/median labels refer only to the configured ranking metric."],
        )
    )

    architecture_files = _plot_architectures(output / METHOD_OUTPUT_FILES[5], dpi)
    assets.append(
        _entry(
            "06_joint_parallel_ordered_comparison.svg",
            "Neutral conceptual comparison of joint, parallel, and ordered internal architectures.",
            m1,
            None,
            [],
            None,
            [],
            [],
            [str(path.resolve()) for path in checkpoints.values()],
            "Architecture-only rendering from repository model definitions.",
            ["No architecture is marked as superior."],
            [
                path.name
                for path in architecture_files
                if path.name != "06_joint_parallel_ordered_comparison.svg"
            ],
        )
    )

    isolation_row = _select_isolation_row(index, metrics)
    isolation_files, isolation_record = _plot_factor_isolation(
        index, isolation_row, renderer, output / METHOD_OUTPUT_FILES[7], dpi
    )
    assets.append(
        _entry(
            isolation_files[0].name,
            "One-factor interventions change the requested factor while preserving the other acquisition factor and morphology.",
            m1,
            metric_path,
            isolation_record,
            isolation_record["source"]["aligned_group_id"],
            [
                isolation_record["source"]["stain_id"],
                isolation_record["stain_swap_real"]["stain_id"],
            ],
            [
                isolation_record["source"]["scanner_id"],
                isolation_record["scanner_swap_real"]["scanner_id"],
            ],
            str(checkpoints["factorstain"].resolve()),
            "Highest saved factor_isolation_score among unseen FactorStain evaluation rows with exact real intervention targets.",
            ["One illustrative example; aggregate scores are in asset 09."],
        )
    )

    comparison_files, comparison_records = _plot_model_comparison(
        index,
        ranked,
        selection_metric,
        renderer,
        output / METHOD_OUTPUT_FILES[8],
        dpi,
    )
    assets.append(
        _entry(
            comparison_files[0].name,
            "Best, median, and hard held-out examples compare all three M1 renderers against the real target.",
            m1,
            metric_path,
            comparison_records,
            [row["source"]["aligned_group_id"] for row in comparison_records],
            sorted(
                {row["source"]["stain_id"] for row in comparison_records}
                | {row["target"]["stain_id"] for row in comparison_records},
                key=natural_key,
            ),
            sorted(
                {row["source"]["scanner_id"] for row in comparison_records}
                | {row["target"]["scanner_id"] for row in comparison_records},
                key=natural_key,
            ),
            [str(path.resolve()) for path in checkpoints.values()],
            f"Best/median/hard quantiles of saved FactorStain {selection_metric}; same pairs rendered by each checkpoint.",
            [
                "Ranking is predefined and includes a hard example; it is not success-only cherry-picking."
            ],
        )
    )

    results_files, result_values = _plot_seen_unseen(
        metrics, output / METHOD_OUTPUT_FILES[9], dpi
    )
    assets.append(
        _entry(
            results_files[0].name,
            "Saved M1 metrics compare Joint, Parallel, and Ordered on seen and unseen acquisition combinations.",
            m1,
            metric_path,
            {"source_sample_manifest": str(index_path.resolve())},
            None,
            split["stains"],
            split["scanners"],
            [str(path.resolve()) for path in checkpoints.values()],
            "Mean of saved per-sample metrics by method and seen/unseen protocol.",
            ["Architectures are presented neutrally; no superiority claim is added."],
            [path.name for path in results_files[1:]],
            result_values,
        )
    )

    equation_files = _plot_equations(output / METHOD_OUTPUT_FILES[11], dpi)
    assets.append(
        _entry(
            "10_method_key_equations.svg",
            "Compact equations state the ordered renderer and compositional task.",
            m1,
            None,
            [],
            None,
            [],
            [],
            None,
            "Direct typesetting of the method definitions.",
            [],
            [
                path.name
                for path in equation_files
                if path.name != "10_method_key_equations.svg"
            ],
        )
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output.resolve()),
        "source_experiment": str(m1.resolve()),
        "combination_split": str(split_path.resolve()),
        "selection_metric": selection_metric,
        "inference_device": str(renderer.device),
        "assets": assets,
    }
    atomic_json(output / "manifest.json", manifest)
    _write_readme(output, m1, assets)
    dimensions = _validate_outputs(output)
    manifest["png_dimensions"] = dimensions
    atomic_json(output / "manifest.json", manifest)

    print("files created")
    for name in METHOD_OUTPUT_FILES:
        print(f"  {output / name}")
    print("source experiment paths")
    print(f"  {m1}")
    print("selected PLISM group IDs")
    print(f"  factorial_grid: {example['aligned_group_id']}")
    for rank, row in ranked.items():
        print(f"  {rank}: {row.aligned_group_id}")
    print("exact output directory")
    print(f"  {output}")


if __name__ == "__main__":
    main()
