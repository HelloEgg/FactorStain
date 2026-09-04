from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from sklearn.decomposition import NMF

from factorstain.data.stain import macenko_normalize
from factorstain.evaluation.renderer import _rgb_to_lab

from .base import AcquisitionMethod, FitContext


def as_uint8_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = array * scale
    return np.clip(array, 0, 255).astype(np.uint8)


def _load_rgb(path: str | Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if size:
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image).copy()


def histogram_normalize(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-channel RGB empirical-CDF matching via scikit-image."""
    try:
        from skimage import exposure

        matched = exposure.match_histograms(source, reference, channel_axis=-1)
    except ImportError:
        matched = np.empty_like(source, dtype=np.float64)
        for channel in range(3):
            _, inverse, counts = np.unique(
                source[..., channel], return_inverse=True, return_counts=True
            )
            target_values, target_counts = np.unique(
                reference[..., channel], return_counts=True
            )
            source_quantiles = np.cumsum(counts) / source[..., channel].size
            target_quantiles = np.cumsum(target_counts) / reference[..., channel].size
            interpolation = np.interp(source_quantiles, target_quantiles, target_values)
            matched[..., channel] = interpolation[inverse].reshape(source.shape[:2])
    return np.clip(matched, 0, 255).astype(np.uint8)


def reinhard_normalize(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reinhard Lab mean/std color transfer with numerical safeguards."""
    source_lab = _rgb_to_lab(source.astype(np.float64) / 255)
    target_lab = _rgb_to_lab(reference.astype(np.float64) / 255)
    source_mean = source_lab.reshape(-1, 3).mean(0)
    target_mean = target_lab.reshape(-1, 3).mean(0)
    source_std = source_lab.reshape(-1, 3).std(0)
    target_std = target_lab.reshape(-1, 3).std(0)
    transferred = (source_lab - source_mean) * (
        target_std / np.clip(source_std, 1e-6, None)
    ) + target_mean
    return np.clip(_lab_to_rgb(transferred) * 255, 0, 255).astype(np.uint8)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    delta = 6 / 29
    transformed = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(
        transformed > delta,
        transformed**3,
        3 * delta**2 * (transformed - 4 / 29),
    )
    xyz *= np.asarray([0.95047, 1.0, 1.08883])
    linear = (
        xyz
        @ np.asarray(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ]
        ).T
    )
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.cbrt(np.clip(linear, 0, None)) - 0.055,
    )


def _vahadane_matrix(rgb: np.ndarray, seed: int = 0) -> tuple[np.ndarray, NMF]:
    optical_density = -np.log(np.clip((rgb.reshape(-1, 3) + 1) / 256, 1e-6, 1))
    tissue = optical_density[(optical_density > 0.15).any(axis=1)]
    if len(tissue) < 20:
        raise ValueError("Insufficient tissue pixels for Vahadane estimation")
    if len(tissue) > 50_000:
        rng = np.random.default_rng(seed)
        tissue = tissue[rng.choice(len(tissue), 50_000, replace=False)]
    model = NMF(
        n_components=2,
        init="nndsvda",
        solver="cd",
        max_iter=300,
        random_state=seed,
        alpha_W=0.01,
        l1_ratio=1.0,
    ).fit(tissue)
    matrix = model.components_.T
    matrix /= np.clip(np.linalg.norm(matrix, axis=0, keepdims=True), 1e-8, None)
    if matrix[0, 0] < matrix[0, 1]:
        matrix = matrix[:, ::-1]
    return matrix, model


def vahadane_normalize(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Deterministic sparse non-negative stain separation and reconstruction."""
    source_matrix, source_model = _vahadane_matrix(source)
    target_matrix, target_model = _vahadane_matrix(reference)
    source_od = -np.log(np.clip((source.reshape(-1, 3) + 1) / 256, 1e-6, 1))
    target_od = -np.log(np.clip((reference.reshape(-1, 3) + 1) / 256, 1e-6, 1))
    source_concentrations = source_model.transform(source_od)
    target_concentrations = target_model.transform(target_od)
    # NMF component order can be permuted; matrices above define the canonical order.
    if not np.allclose(
        source_matrix,
        source_model.components_.T
        / np.clip(
            np.linalg.norm(source_model.components_.T, axis=0, keepdims=True),
            1e-8,
            None,
        ),
    ):
        source_concentrations = source_concentrations[:, ::-1]
    target_raw = target_model.components_.T
    target_raw /= np.clip(np.linalg.norm(target_raw, axis=0, keepdims=True), 1e-8, None)
    if not np.allclose(target_matrix, target_raw):
        target_concentrations = target_concentrations[:, ::-1]
    source_scale = np.percentile(source_concentrations, 99, axis=0)
    target_scale = np.percentile(target_concentrations, 99, axis=0)
    normalized = source_concentrations * (
        target_scale / np.clip(source_scale, 1e-8, None)
    )
    rendered = 255 * np.exp(-(target_matrix @ normalized.T).T)
    return np.clip(rendered.reshape(source.shape), 0, 255).astype(np.uint8)


def _design(rgb: np.ndarray, kind: str) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    r, g, b = values.T
    if kind == "affine":
        return np.column_stack([r, g, b, np.ones(len(values))])
    if kind == "polynomial":
        return np.column_stack(
            [r, g, b, r * r, g * g, b * b, r * g, r * b, g * b, np.ones(len(values))]
        )
    raise ValueError(kind)


def _sample_paired_pixels(
    pairs: pd.DataFrame, image_size: int, seed: int, maximum: int = 250_000
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left, right = [], []
    per_pair = max(256, maximum // max(1, len(pairs)))
    for pair in pairs.itertuples():
        source = _load_rgb(pair.source_path, image_size).reshape(-1, 3)
        target = _load_rgb(pair.target_path, image_size).reshape(-1, 3)
        count = min(per_pair, len(source))
        chosen = rng.choice(len(source), count, replace=False)
        left.append(source[chosen])
        right.append(target[chosen])
    if not left:
        raise ValueError("No clean training scanner pairs were supplied")
    return np.concatenate(left), np.concatenate(right)


@dataclass
class ColorTransform:
    kind: str
    payload: np.ndarray
    grid_size: int = 17

    def apply(self, image: np.ndarray) -> np.ndarray:
        rgb = as_uint8_rgb(image)
        if self.kind in {"affine", "polynomial"}:
            values = _design(rgb, self.kind) @ self.payload
            return np.clip(values.reshape(rgb.shape) * 255, 0, 255).astype(np.uint8)
        coordinates = (rgb.astype(np.float64) / 255.0 * (self.grid_size - 1)).reshape(
            -1, 3
        )
        mapped = np.column_stack(
            [
                map_coordinates(
                    self.payload[..., channel], coordinates.T, order=1, mode="nearest"
                )
                for channel in range(3)
            ]
        )
        return np.clip(mapped.reshape(rgb.shape) * 255, 0, 255).astype(np.uint8)


def fit_color_transform(
    pairs: pd.DataFrame,
    kind: str,
    image_size: int,
    seed: int,
    grid_size: int = 17,
) -> ColorTransform:
    source, target = _sample_paired_pixels(pairs, image_size, seed)
    if kind in {"affine", "polynomial"}:
        coefficients = np.linalg.lstsq(
            _design(source, kind), target.astype(np.float64) / 255.0, rcond=1e-6
        )[0]
        return ColorTransform(kind, coefficients)
    if kind != "lut":
        raise ValueError(f"Unknown scanner transform {kind}")
    bins = np.minimum(
        (source.astype(np.float64) / 256 * grid_size).astype(int), grid_size - 1
    )
    residual_sum = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float64)
    counts = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    residual = (target.astype(np.float64) - source.astype(np.float64)) / 255.0
    np.add.at(residual_sum, tuple(bins.T), residual)
    np.add.at(counts, tuple(bins.T), 1)
    observed = counts > 0
    residual_grid = residual_sum / np.clip(counts[..., None], 1, None)
    if observed.any() and not observed.all():
        nearest = distance_transform_edt(
            ~observed, return_distances=False, return_indices=True
        )
        residual_grid = residual_grid[tuple(nearest)]
    residual_grid = gaussian_filter(residual_grid, sigma=(0.75, 0.75, 0.75, 0))
    axis = np.linspace(0, 1, grid_size)
    identity = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return ColorTransform("lut", np.clip(identity + residual_grid, 0, 1), grid_size)


class ScannerTransformBank:
    def __init__(self, kind: str = "lut", grid_size: int = 17) -> None:
        self.kind = kind
        self.grid_size = grid_size
        self.transforms: dict[tuple[str, str], ColorTransform] = {}

    def fit(self, pairs: pd.DataFrame, image_size: int, seed: int) -> None:
        if pairs.empty:
            return
        for number, ((source, target), frame) in enumerate(
            pairs.groupby(["source_scanner", "target_scanner"], sort=True)
        ):
            self.transforms[(str(source), str(target))] = fit_color_transform(
                frame, self.kind, image_size, seed + number, self.grid_size
            )
        for number, (target, frame) in enumerate(
            pairs.groupby("target_scanner", sort=True)
        ):
            self.transforms[("__pooled__", str(target))] = fit_color_transform(
                frame, self.kind, image_size, seed + 1000 + number, self.grid_size
            )

    def apply(
        self, image: np.ndarray, source_scanner: str, target_scanner: str
    ) -> np.ndarray:
        if source_scanner == target_scanner:
            return as_uint8_rgb(image)
        transform = self.transforms.get((str(source_scanner), str(target_scanner)))
        transform = transform or self.transforms.get(
            ("__pooled__", str(target_scanner))
        )
        if transform is None:
            raise ValueError(
                f"No training-only scanner transform for {source_scanner}->{target_scanner}"
            )
        return transform.apply(image)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays, manifest = (
            {},
            {"kind": self.kind, "grid_size": self.grid_size, "keys": {}},
        )
        for number, (key, transform) in enumerate(sorted(self.transforms.items())):
            name = f"transform_{number}"
            arrays[name] = transform.payload
            manifest["keys"][name] = list(key)
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        manifest_path = path.with_suffix(".json")
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Scanner transform cache is incomplete: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != self.kind:
            raise ValueError(
                f"Scanner transform kind is {manifest.get('kind')}, expected {self.kind}"
            )
        if int(manifest.get("grid_size", -1)) != self.grid_size:
            raise ValueError(
                "Scanner transform grid size is "
                f"{manifest.get('grid_size')}, expected {self.grid_size}"
            )
        transforms: dict[tuple[str, str], ColorTransform] = {}
        with np.load(path, allow_pickle=False) as arrays:
            for name, key in manifest.get("keys", {}).items():
                if name not in arrays or len(key) != 2:
                    raise ValueError(f"Invalid scanner transform cache entry: {name}")
                transforms[(str(key[0]), str(key[1]))] = ColorTransform(
                    self.kind, arrays[name].copy(), self.grid_size
                )
        self.transforms = transforms


class NoAdaptMethod(AcquisitionMethod):
    method_name = "noadapt"

    def fit(self, context: FitContext, val_data: pd.DataFrame | None = None) -> None:
        self.context = context

    def translate(
        self,
        source_image: Image.Image | np.ndarray,
        source_stain: str,
        source_scanner: str,
        target_stain: str,
        target_scanner: str,
        *,
        track: str = "C",
    ) -> np.ndarray:
        return as_uint8_rgb(source_image)

    def supports_strict_composition(self) -> bool:
        return True


NORMALIZERS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "histogram": histogram_normalize,
    "reinhard": reinhard_normalize,
    "macenko": lambda source, target: np.asarray(macenko_normalize(source, target)),
    "vahadane": vahadane_normalize,
}


class ClassicalStainMethod(AcquisitionMethod):
    """Stain normalizer plus a separately learned scanner transform for Track C."""

    def __init__(
        self, method_name: str, scanner_kind: str = "lut", grid_size: int = 17
    ):
        if method_name not in NORMALIZERS:
            raise KeyError(method_name)
        self.method_name = method_name
        self.normalizer = NORMALIZERS[method_name]
        self.scanner_bank = ScannerTransformBank(scanner_kind, grid_size)
        self.references: dict[str, tuple[np.ndarray, str]] = {}

    def fit(self, context: FitContext, val_data: pd.DataFrame | None = None) -> None:
        for stain, details in context.reference_policy["stain_prototypes"].items():
            self.references[str(stain)] = (
                _load_rgb(details["image_path"], context.image_size),
                str(details["scanner_id"]),
            )
        self.scanner_bank.fit(context.scanner_pairs, context.image_size, context.seed)

    def translate(
        self,
        source_image: Image.Image | np.ndarray,
        source_stain: str,
        source_scanner: str,
        target_stain: str,
        target_scanner: str,
        *,
        track: str = "C",
    ) -> np.ndarray:
        source = as_uint8_rgb(source_image)
        if track == "B":
            return self.scanner_bank.apply(source, source_scanner, target_scanner)
        if target_stain not in self.references:
            raise ValueError(f"No training-only reference for stain {target_stain}")
        reference, reference_scanner = self.references[target_stain]
        stained = self.normalizer(source, reference)
        if track == "A":
            return stained
        return self.scanner_bank.apply(stained, reference_scanner, target_scanner)

    def supports_strict_composition(self) -> bool:
        return True


class ScannerOnlyMethod(AcquisitionMethod):
    def __init__(self, method_name: str, kind: str, grid_size: int = 17):
        self.method_name = method_name
        self.scanner_bank = ScannerTransformBank(kind, grid_size)

    def fit(self, context: FitContext, val_data: pd.DataFrame | None = None) -> None:
        self.scanner_bank.fit(context.scanner_pairs, context.image_size, context.seed)

    def translate(
        self,
        source_image: Image.Image | np.ndarray,
        source_stain: str,
        source_scanner: str,
        target_stain: str,
        target_scanner: str,
        *,
        track: str = "B",
    ) -> np.ndarray:
        if source_stain != target_stain and track != "B":
            raise ValueError("Scanner-only methods do not support stain transfer")
        return self.scanner_bank.apply(
            as_uint8_rgb(source_image), source_scanner, target_scanner
        )
