from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mmd_heatmap(
    matrix: pd.DataFrame,
    destination: str | Path,
    title: str,
    vmax: float | None = None,
) -> None:
    values = matrix.to_numpy(dtype=float)
    upper = float(np.nanmax(values)) if vmax is None else float(vmax)
    upper = max(upper, 1e-8)
    size = max(7.0, len(matrix) * 0.68)
    fig, axis = plt.subplots(figsize=(size, size * 0.88))
    image = axis.imshow(values, cmap="magma", vmin=0.0, vmax=upper, aspect="equal")
    axis.set_xticks(range(len(matrix)), matrix.columns, rotation=40, ha="right")
    axis.set_yticks(range(len(matrix)), matrix.index)
    midpoint = upper * 0.48
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=max(5.5, 9.0 - len(matrix) * 0.18),
                color="white" if value <= midpoint else "black",
            )
    axis.set_title(title, fontsize=14, weight="bold", pad=12)
    fig.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="reported unbiased multi-kernel MMD²",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gap_distributions(
    values: pd.DataFrame,
    destination: str | Path,
) -> None:
    order = [
        "scanner marginal",
        "scanner controlled",
        "stain marginal",
        "stain balanced",
    ]
    groups = [
        values.loc[values.analysis.eq(label), "mmd2"].to_numpy() for label in order
    ]
    fig, axis = plt.subplots(figsize=(11, 6.2))
    violin = axis.violinplot(groups, showmeans=True, showmedians=True, widths=0.8)
    colors = ["#60a5fa", "#1d4ed8", "#fbbf24", "#b45309"]
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.5)
    boxes = axis.boxplot(groups, widths=0.17, patch_artist=True, showfliers=True)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axis.set_xticks(
        range(1, len(order) + 1), [label.replace(" ", "\n", 1) for label in order]
    )
    axis.set_ylabel("reported unbiased multi-kernel MMD²")
    axis.set_title(
        "PLISM scanner and stain domain-gap distributions", fontsize=14, weight="bold"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.text(
        0.01,
        -0.17,
        "Domain counts and interventions differ; distribution heights should not be read as a causal scanner-vs-stain ranking.",
        transform=axis.transAxes,
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_centroid_vs_mmd(
    pairs: pd.DataFrame,
    spearman: float,
    destination: str | Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 6.2))
    axis.scatter(
        pairs.centroid_cosine_distance,
        pairs.controlled_mmd2,
        s=46,
        color="#2563eb",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    for _, row in pairs.iterrows():
        axis.annotate(
            f"{row.domain_a}–{row.domain_b}",
            (row.centroid_cosine_distance, row.controlled_mmd2),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            alpha=0.82,
        )
    axis.set_xlabel("scanner centroid cosine distance")
    axis.set_ylabel("controlled scanner MMD²")
    axis.set_title(
        f"Mean shift vs distribution shift (Spearman ρ={spearman:.3f})",
        fontsize=13,
        weight="bold",
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
