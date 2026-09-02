from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def plot_projection(
    coordinates: np.ndarray, labels: pd.Series, destination: str | Path, title: str
) -> None:
    label_values = labels.astype(str).to_numpy()
    domains = sorted(np.unique(label_values))
    palette = plt.get_cmap("tab20", max(1, min(20, len(domains))))
    fig, axis = plt.subplots(figsize=(9, 7))
    for index, domain in enumerate(domains):
        selected = label_values == domain
        axis.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=9,
            alpha=0.62,
            linewidths=0,
            color=palette(index % 20),
            label=domain,
            rasterized=True,
        )
    axis.set_title(title, fontsize=14, weight="bold")
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    if len(domains) <= 24:
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=7,
            frameon=False,
            markerscale=1.5,
        )
    else:
        axis.text(
            1.01,
            0.5,
            f"{len(domains)} case labels\n(legend omitted)",
            transform=axis.transAxes,
            va="center",
            fontsize=9,
        )
    axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_heatmap(
    matrix: pd.DataFrame, destination: str | Path, title: str
) -> None:
    fig, axis = plt.subplots(
        figsize=(max(6.5, len(matrix) * 0.72), max(5.5, len(matrix) * 0.60))
    )
    image = axis.imshow(matrix.to_numpy(), cmap="magma", vmin=0, aspect="auto")
    axis.set_xticks(
        range(len(matrix)), matrix.columns, rotation=35, ha="right", fontsize=8
    )
    axis.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
    if len(matrix) <= 12:
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                axis.text(
                    column,
                    row,
                    f"{matrix.iloc[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="white"
                    if matrix.iloc[row, column] > matrix.to_numpy().max() * 0.45
                    else "black",
                    fontsize=7,
                )
    axis.set_title(title, fontsize=13, weight="bold")
    fig.colorbar(image, ax=axis, label="cosine distance")
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_distance_distribution(
    frame: pd.DataFrame, destination: str | Path, title: str
) -> None:
    categories = frame.category.drop_duplicates().tolist()
    values = [
        frame.loc[frame.category.eq(category), "distance"].dropna().to_numpy()
        for category in categories
    ]
    fig, axis = plt.subplots(figsize=(max(9, 2.6 * len(categories)), 6))
    violin = axis.violinplot(values, showmeans=True, showmedians=True, widths=0.75)
    for body in violin["bodies"]:
        body.set_facecolor("#3b82f6")
        body.set_alpha(0.45)
    axis.boxplot(
        values,
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "alpha": 0.8},
    )
    axis.set_xticks(
        range(1, len(categories) + 1),
        ["\n".join(textwrap.wrap(category, 24)) for category in categories],
        fontsize=8,
    )
    axis.set_ylabel("DINOv3 cosine distance")
    axis.set_title(title, fontsize=13, weight="bold")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_probe_scores(probes: list[dict], destination: str | Path, title: str) -> None:
    targets = [str(item["target"]) for item in probes]
    scores = [float(item["balanced_accuracy"]) for item in probes]
    chances = [float(item["chance"]) for item in probes]
    labels = [str(item["domain_signal"]) for item in probes]
    displayed_scores = [score if np.isfinite(score) else 0.0 for score in scores]
    positions = np.arange(len(targets))
    fig, axis = plt.subplots(figsize=(max(6.5, len(targets) * 2.2), 5.5))
    bars = axis.bar(
        positions,
        displayed_scores,
        color=[
            {
                "STRONG": "#15803d",
                "MODERATE": "#d97706",
                "WEAK": "#64748b",
                "NOT_ESTIMABLE": "#94a3b8",
            }[label]
            for label in labels
        ],
        alpha=0.85,
    )
    axis.scatter(
        positions,
        chances,
        marker="_",
        s=900,
        linewidth=3,
        color="#dc2626",
        label="chance",
    )
    for bar, score, label in zip(bars, scores, labels):
        annotation = (
            f"{score:.3f}\n{label}" if np.isfinite(score) else "N/A\nNOT ESTIMABLE"
        )
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.02, score + 0.035) if np.isfinite(score) else 0.035,
            annotation,
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    axis.set_xticks(positions, targets)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("group-held-out balanced accuracy")
    axis.set_title(title, fontsize=13, weight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _sample_label(row: pd.Series, dataset: str) -> str:
    if dataset == "plism":
        return f"T:{row.tissue_type}\nS:{row.stain_id} | Q:{row.scanner_id}"
    return f"{row.scanner_id}\ncase {row.case_id}"


def plot_nearest_neighbors(
    metadata: pd.DataFrame,
    pairs: list[tuple[int, list[int]]],
    dataset: str,
    destination: str | Path,
    title: str,
) -> None:
    from factorstain.visualization.domain_audit import read_audit_image

    columns = 1 + max((len(neighbors) for _, neighbors in pairs), default=0)
    fig, axes = plt.subplots(
        len(pairs),
        columns,
        figsize=(2.25 * columns, 2.25 * max(1, len(pairs))),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for row_number, (query, neighbors) in enumerate(pairs):
        positions = [query] + neighbors
        for column_number, position in enumerate(positions):
            row = metadata.iloc[position]
            axes[row_number, column_number].imshow(read_audit_image(row, dataset, 180))
            prefix = "QUERY" if column_number == 0 else f"NN {column_number}"
            axes[row_number, column_number].set_title(
                f"{prefix}\n{_sample_label(row, dataset)}",
                fontsize=6.5,
                color="#991b1b" if column_number == 0 else "black",
            )
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def compose_dashboard(
    panels: list[tuple[str, str | Path | None]],
    destination: str | Path,
    title: str,
    footer: str,
    columns: int = 2,
) -> None:
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(10 * columns, 7.1 * rows + 1.1), squeeze=False
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, (panel_title, path) in zip(axes.flat, panels):
        if path is not None and Path(path).exists():
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
        else:
            axis.text(
                0.5,
                0.5,
                "Panel unavailable",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#64748b",
            )
        axis.set_title(panel_title, fontsize=13, weight="bold", pad=7)
    fig.suptitle(title, fontsize=23, weight="bold", y=0.995)
    fig.text(
        0.5,
        0.008,
        footer,
        ha="center",
        va="bottom",
        fontsize=12,
        weight="bold",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.savefig(destination, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
