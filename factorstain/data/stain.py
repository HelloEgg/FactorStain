from __future__ import annotations

import numpy as np
from PIL import Image


def _stain_matrix(rgb: np.ndarray, alpha: float = 1.0, beta: float = 0.15) -> np.ndarray:
    pixels = rgb.reshape(-1, 3).astype(np.float64)
    optical_density = -np.log(np.clip((pixels + 1) / 256, 1e-6, 1))
    optical_density = optical_density[(optical_density > beta).all(axis=1)]
    if len(optical_density) < 10:
        raise ValueError("Insufficient tissue pixels for Macenko estimation")
    _, _, vectors = np.linalg.svd(optical_density, full_matrices=False)
    plane = vectors[:2].T
    projected = optical_density @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [alpha, 100 - alpha])
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    matrix = np.stack([first, second], axis=1)
    return matrix if matrix[0, 0] > matrix[0, 1] else matrix[:, ::-1]


def macenko_normalize(image: Image.Image | np.ndarray, reference: Image.Image | np.ndarray) -> Image.Image:
    """Normalize H&E stain basis/concentrations using the Macenko method."""
    source = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
    target = np.asarray(reference.convert("RGB") if isinstance(reference, Image.Image) else reference, dtype=np.uint8)
    source_matrix, target_matrix = _stain_matrix(source), _stain_matrix(target)
    source_od = -np.log(np.clip((source.reshape(-1, 3) + 1) / 256, 1e-6, 1)).T
    target_od = -np.log(np.clip((target.reshape(-1, 3) + 1) / 256, 1e-6, 1)).T
    source_concentrations = np.linalg.lstsq(source_matrix, source_od, rcond=None)[0]
    target_concentrations = np.linalg.lstsq(target_matrix, target_od, rcond=None)[0]
    source_scale = np.percentile(source_concentrations, 99, axis=1)
    target_scale = np.percentile(target_concentrations, 99, axis=1)
    normalized = source_concentrations * (target_scale / np.clip(source_scale, 1e-8, None))[:, None]
    rendered = 255 * np.exp(-target_matrix @ normalized)
    return Image.fromarray(np.clip(rendered.T.reshape(source.shape), 0, 255).astype(np.uint8))


def stain_augment(image: Image.Image | np.ndarray, strength: float = 0.15, seed: int | None = None) -> Image.Image:
    """Perturb estimated H/E concentrations while preserving the estimated stain basis."""
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image, dtype=np.uint8)
    matrix = _stain_matrix(rgb)
    optical_density = -np.log(np.clip((rgb.reshape(-1, 3) + 1) / 256, 1e-6, 1)).T
    concentrations = np.linalg.lstsq(matrix, optical_density, rcond=None)[0]
    rng = np.random.default_rng(seed)
    scale = rng.uniform(1 - strength, 1 + strength, size=(2, 1))
    bias = rng.uniform(-strength, strength, size=(2, 1)) * concentrations.std(axis=1, keepdims=True)
    rendered = 255 * np.exp(-matrix @ np.clip(concentrations * scale + bias, 0, None))
    return Image.fromarray(np.clip(rendered.T.reshape(rgb.shape), 0, 255).astype(np.uint8))

