from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def infer_center_from_patient(patient_id: int) -> int:
    """Official CAMELYON17 training patients are partitioned in blocks of 20."""
    if not 0 <= patient_id <= 99:
        raise ValueError(f"Training patient must be in 000–099, got {patient_id}")
    return patient_id // 20


def build_camelyon_index(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    slides = [*root.rglob("*.tif"), *root.rglob("*.tiff")]
    rows = []
    for slide in slides:
        if "mask" in slide.stem.lower() or "annotation" in slide.stem.lower():
            continue
        match = re.search(r"patient[_-]?(\d{3})", slide.name, re.IGNORECASE)
        if not match:
            continue
        patient = int(match.group(1))
        if patient > 99:
            continue
        node_match = re.search(r"node[_-]?(\d+)", slide.name, re.IGNORECASE)
        rows.append(
            {
                "slide_id": slide.stem,
                "slide_path": str(slide.resolve()),
                "patient_id": patient,
                "node_id": int(node_match.group(1)) if node_match else -1,
                "center": infer_center_from_patient(patient),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["slide_id", "slide_path", "patient_id", "node_id", "center", "label", "label_text"])
    mask_stems = {
        re.sub(r"(?i)(_mask|_annotation)$", "", path.stem): path
        for path in [*root.rglob("*mask*.tif"), *root.rglob("*mask*.tiff")]
    }
    if mask_stems:
        frame["label"] = frame.slide_id.map(lambda value: int(value in mask_stems))
        frame["label_text"] = frame.label.map({0: "negative", 1: "metastasis"})
    label_files = [*root.rglob("*stage*.csv"), *root.rglob("*label*.csv")]
    for labels_path in label_files:
        labels = pd.read_csv(labels_path)
        normalized = {c.lower(): c for c in labels.columns}
        patient_col = next((normalized[k] for k in normalized if "patient" in k), None)
        label_col = next((normalized[k] for k in normalized if "stage" in k or "label" in k), None)
        if patient_col and label_col and "label" not in frame:
            label_map = {
                int(re.search(r"\d+", str(row[patient_col])).group()): str(row[label_col])
                for _, row in labels.iterrows()
                if re.search(r"\d+", str(row[patient_col]))
            }
            frame["label_text"] = frame.patient_id.map(label_map)
            frame["label"] = frame.label_text.fillna("negative").str.lower().ne("negative").astype(int)
            break
    if "label" not in frame:
        frame["label"] = -1
        frame["label_text"] = "unknown"
    return frame.sort_values(["patient_id", "node_id"]).reset_index(drop=True)


def tissue_coordinates(
    slide_path: str | Path,
    tile_size: int = 256,
    target_mpp: float = 0.5,
    tissue_threshold: float = 0.50,
    max_tiles: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate level-0 tile coordinates from a low-resolution HSV tissue mask."""
    try:
        import tiffslide
    except ImportError as exc:
        raise RuntimeError("tiffslide is required for CAMELYON preprocessing; run bash shell/setup.sh") from exc
    import cv2

    with tiffslide.TiffSlide(str(slide_path)) as slide:
        level = len(slide.level_dimensions) - 1
        width, height = slide.level_dimensions[level]
        thumbnail = np.asarray(slide.read_region((0, 0), level, (width, height)).convert("RGB"))
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 1] > 20) & (hsv[..., 2] < 245)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        downs = slide.level_downsamples[level]
        mpp_x = float(slide.properties.get("tiffslide.mpp-x", target_mpp))
        level0_stride = max(1, round(tile_size * target_mpp / mpp_x))
        scaled = max(1, round(level0_stride / downs))
        rows = []
        for y in range(0, height - scaled + 1, scaled):
            for x in range(0, width - scaled + 1, scaled):
                fraction = float(mask[y : y + scaled, x : x + scaled].mean())
                if fraction >= tissue_threshold:
                    rows.append({"x": round(x * downs), "y": round(y * downs), "tissue_fraction": fraction})
    frame = pd.DataFrame(rows)
    if max_tiles and len(frame) > max_tiles:
        frame = frame.sample(max_tiles, random_state=seed).sort_values(["y", "x"])
    frame["slide_path"] = str(Path(slide_path).resolve())
    frame["read_size"] = level0_stride
    return frame.reset_index(drop=True)
