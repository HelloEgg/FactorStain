from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.color import rgb2hed

from factorstain.data.domain_audit import read_midog_patch


def read_audit_image(
    row: pd.Series | dict, dataset_name: str, display_size: int = 224
) -> Image.Image:
    row = pd.Series(row)
    if dataset_name == "plism":
        with Image.open(row.image_path) as opened:
            image = opened.convert("RGB")
    else:
        image = read_midog_patch(row)
    return ImageOps.fit(
        image, (display_size, display_size), method=Image.Resampling.BILINEAR
    )


def select_domain_examples(
    frame: pd.DataFrame, factor: str, count: int, seed: int
) -> pd.DataFrame:
    selections = []
    for domain_number, (domain, group) in enumerate(frame.groupby(factor, sort=True)):
        chosen = group.sample(
            min(count, len(group)), random_state=seed + domain_number
        ).copy()
        chosen["visual_factor"] = factor
        chosen["visual_domain"] = str(domain)
        chosen["visual_order"] = np.arange(len(chosen))
        selections.append(chosen)
    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def _label(row: pd.Series, dataset_name: str) -> str:
    if dataset_name == "plism":
        return f"S:{row.stain_id} | Q:{row.scanner_id}\nT:{row.tissue_type} | G:{str(row.aligned_group_id)[:10]}"
    return f"{row.scanner_id}\ncase {row.case_id} | ({int(row.x)}, {int(row.y)})"


def plot_domain_grid(
    selection: pd.DataFrame,
    factor: str,
    dataset_name: str,
    destination: str | Path,
    title: str,
    display_size: int = 180,
) -> None:
    domains = sorted(selection[factor].astype(str).unique())
    columns = int(selection.groupby(factor).size().max())
    fig, axes = plt.subplots(
        len(domains),
        columns,
        figsize=(2.5 * columns, 2.35 * len(domains)),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for row_number, domain in enumerate(domains):
        group = selection[selection[factor].astype(str).eq(domain)].sort_values(
            "visual_order"
        )
        for column_number, (_, row) in enumerate(group.iterrows()):
            axes[row_number, column_number].imshow(
                read_audit_image(row, dataset_name, display_size)
            )
            axes[row_number, column_number].set_title(
                _label(row, dataset_name), fontsize=6
            )
        axes[row_number, 0].text(
            -0.06,
            0.5,
            domain,
            transform=axes[row_number, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
            weight="bold",
        )
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_plism_controlled(comparisons: pd.DataFrame, figure_dir: Path) -> None:
    scanner = comparisons[
        comparisons.comparison_type.eq("same_morphology_same_stain_across_scanners")
    ]
    for comparison_id, group in scanner.groupby("comparison_id"):
        fig, axes = plt.subplots(
            1, len(group), figsize=(2.7 * len(group), 3), squeeze=False
        )
        for axis, (_, row) in zip(
            axes[0], group.sort_values("display_order").iterrows()
        ):
            axis.imshow(read_audit_image(row, "plism", 224))
            axis.axis("off")
            axis.set_title(str(row.scanner_id), fontsize=9)
        first = group.iloc[0]
        fig.suptitle(
            f"Same aligned morphology + same stain ({first.stain_id}) across scanners\nGroup {first.aligned_group_id} | {first.tissue_type}",
            fontsize=13,
            weight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            figure_dir
            / f"plism_same_tissue_same_stain_across_scanners_{int(comparison_id):02d}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)
    stains = comparisons[
        comparisons.comparison_type.eq("aligned_tissue_same_scanner_across_stains")
    ]
    for comparison_id, group in stains.groupby("comparison_id"):
        fig, axes = plt.subplots(
            1, len(group), figsize=(2.7 * len(group), 3), squeeze=False
        )
        for axis, (_, row) in zip(
            axes[0], group.sort_values("display_order").iterrows()
        ):
            axis.imshow(read_audit_image(row, "plism", 224))
            axis.axis("off")
            axis.set_title(str(row.stain_id), fontsize=9)
        first = group.iloc[0]
        fig.suptitle(
            f"Aligned tissue + same scanner ({first.scanner_id}) across serial stain conditions\nGroup {first.aligned_group_id} | {first.tissue_type}",
            fontsize=13,
            weight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"plism_tissue_across_stains_{int(comparison_id):02d}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def pixel_statistics(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    import cv2

    rows = []
    for _, item in frame.iterrows():
        image = (
            np.asarray(read_audit_image(item, dataset_name, 224), dtype=np.float32)
            / 255.0
        )
        uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        od = -np.log(np.clip(image, 1 / 255, 1))
        gray = cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        sx, sy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        hed = rgb2hed(image)
        row = {key: item[key] for key in ("sample_id", "scanner_id") if key in item}
        for key in ("stain_id", "tissue_type", "aligned_group_id", "case_id"):
            if key in item:
                row[key] = item[key]
        row.update(
            {
                "laplacian_variance": float(laplacian.var()),
                "sharpness": float(np.abs(laplacian).mean()),
                "edge_energy": float(np.sqrt(sx**2 + sy**2).mean()),
                "h_mean": float(hed[..., 0].mean()),
                "e_mean": float(hed[..., 1].mean()),
            }
        )
        for name, values in (("rgb", image), ("hsv", hsv), ("lab", lab), ("od", od)):
            for channel in range(3):
                row[f"{name}_{channel}_mean"] = float(values[..., channel].mean())
                row[f"{name}_{channel}_std"] = float(values[..., channel].std())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_color_statistics(
    statistics: pd.DataFrame, factor: str, destination: str | Path
) -> None:
    columns = [
        "rgb_0_mean",
        "rgb_1_mean",
        "rgb_2_mean",
        "hsv_0_mean",
        "hsv_1_mean",
        "lab_1_mean",
        "lab_2_mean",
        "od_0_mean",
        "od_1_mean",
        "od_2_mean",
        "h_mean",
        "e_mean",
    ]
    grouped = statistics.groupby(factor)[columns].mean()
    standardized = (grouped - grouped.mean()) / grouped.std().replace(0, 1)
    fig, axis = plt.subplots(figsize=(13, max(4.5, len(grouped) * 0.45)))
    image = axis.imshow(
        standardized, cmap="coolwarm", aspect="auto", vmin=-2.5, vmax=2.5
    )
    axis.set_xticks(range(len(columns)), columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(grouped)), grouped.index)
    axis.set_title(f"Raw color/OD statistics by {factor.replace('_id', '')}")
    fig.colorbar(image, ax=axis, label="domain mean z-score")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sharpness(
    statistics: pd.DataFrame, factor: str, destination: str | Path
) -> None:
    groups = [
        group.laplacian_variance.to_numpy() for _, group in statistics.groupby(factor)
    ]
    labels = [str(label) for label, _ in statistics.groupby(factor)]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(groups, tick_labels=labels, showfliers=False)
    axis.set_title(f"Laplacian sharpness by {factor.replace('_id', '')}")
    axis.set_ylabel("Laplacian variance")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
