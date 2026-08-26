from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.color import rgb2hed


def _read_rgb(path: str | Path, size: int = 224) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.fit(opened.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def appearance_statistics(index: pd.DataFrame, max_images: int = 5000, seed: int = 42) -> pd.DataFrame:
    import cv2

    sample = index[index.image_exists].sample(min(max_images, int(index.image_exists.sum())), random_state=seed)
    rows = []
    for _, item in sample.iterrows():
        image = _read_rgb(item.image_path)
        uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        od = -np.log(np.clip(image, 1 / 255, 1))
        hed = rgb2hed(image)
        gray = cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        row = {
            "image_id": item.image_id,
            "stain_id": item.stain_id,
            "scanner_id": item.scanner_id,
            "aligned_group_id": item.aligned_group_id,
            "edge_energy": float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2))),
            "laplacian_variance": float(laplacian.var()),
            "sharpness": float(np.mean(np.abs(laplacian))),
            "h_mean": float(hed[..., 0].mean()),
            "e_mean": float(hed[..., 1].mean()),
        }
        for label, array in (("rgb", image), ("hsv", hsv), ("lab", lab), ("od", od)):
            for channel in range(3):
                row[f"{label}_{channel}_mean"] = float(array[..., channel].mean())
                row[f"{label}_{channel}_std"] = float(array[..., channel].std())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_real_grid(index: pd.DataFrame, destination: str | Path, groups: int = 1) -> None:
    stains = sorted(index.stain_id.unique())
    scanners = sorted(index.scanner_id.unique())
    counts = index.groupby("aligned_group_id").size().sort_values(ascending=False)
    selected = counts.head(groups).index
    # The required 13×7 layout is preserved; multiple groups are exported as sequential figures if requested.
    for number, group_id in enumerate(selected):
        group = index[index.aligned_group_id == group_id]
        lookup = {(row.stain_id, row.scanner_id): row.image_path for _, row in group.iterrows() if row.image_exists}
        fig, axes = plt.subplots(len(stains), len(scanners), figsize=(2 * len(scanners), 1.7 * len(stains)))
        axes = np.asarray(axes).reshape(len(stains), len(scanners))
        for i, stain in enumerate(stains):
            for j, scanner in enumerate(scanners):
                axis = axes[i, j]
                axis.axis("off")
                if (stain, scanner) in lookup:
                    axis.imshow(_read_rgb(lookup[(stain, scanner)], size=160))
                else:
                    axis.set_facecolor("#eeeeee")
                    axis.text(0.5, 0.5, "missing", ha="center", va="center", color="#777777")
                if i == 0:
                    axis.set_title(str(scanner), fontsize=8)
                if j == 0:
                    axis.text(-0.08, 0.5, str(stain), transform=axis.transAxes, ha="right", va="center", fontsize=8)
        fig.suptitle(f"PLISM aligned morphology {group_id}: stains × scanners", fontsize=15, weight="bold")
        fig.tight_layout()
        path = Path(destination)
        if number:
            path = path.with_name(f"{path.stem}_group{number + 1}{path.suffix}")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_plism_summary(index: pd.DataFrame, destination: str | Path) -> None:
    stains, scanners = sorted(index.stain_id.unique()), sorted(index.scanner_id.unique())
    cell_counts = index.pivot_table(index="stain_id", columns="scanner_id", values="image_id", aggfunc="count", fill_value=0).reindex(index=stains, columns=scanners)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    index.tissue_type.value_counts().head(25).sort_values().plot.barh(ax=axes[0, 0], color="#4263eb")
    axes[0, 0].set_title(f"Tissue distribution (top 25; total n={len(index):,})")
    index.stain_id.value_counts().reindex(stains).plot.bar(ax=axes[0, 1], color="#e8590c")
    axes[0, 1].set_title("Stain distribution")
    index.scanner_id.value_counts().reindex(scanners).plot.bar(ax=axes[1, 0], color="#2b8a3e")
    axes[1, 0].set_title("Scanner distribution")
    image = axes[1, 1].imshow(cell_counts.eq(0), cmap="Reds", aspect="auto")
    axes[1, 1].set_xticks(range(len(scanners)), scanners, rotation=45, ha="right")
    axes[1, 1].set_yticks(range(len(stains)), stains)
    axes[1, 1].set_title("Missing acquisition cells (red = missing)")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_appearance_by_factor(statistics: pd.DataFrame, factor: str, destination: str | Path) -> None:
    columns = ["rgb_0_mean", "rgb_1_mean", "rgb_2_mean", "od_0_mean", "od_1_mean", "od_2_mean", "h_mean", "e_mean"]
    labels = ["R", "G", "B", "OD-R", "OD-G", "OD-B", "H", "E"]
    grouped = statistics.groupby(factor)[columns].mean()
    normalized = (grouped - grouped.mean()) / grouped.std().replace(0, 1)
    fig, axis = plt.subplots(figsize=(12, max(5, len(grouped) * 0.45)))
    image = axis.imshow(normalized, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(grouped)), grouped.index)
    axis.set_title(f"Standardized color/OD statistics by {factor.replace('_id', '')}")
    fig.colorbar(image, ax=axis, label="z-score")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sharpness(statistics: pd.DataFrame, destination: str | Path) -> None:
    groups = [group.laplacian_variance.to_numpy() for _, group in statistics.groupby("scanner_id")]
    labels = [str(label) for label, _ in statistics.groupby("scanner_id")]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(groups, tick_labels=labels, showfliers=False)
    axis.set_ylabel("Laplacian variance")
    axis.set_title("Sharpness by scanner")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)

