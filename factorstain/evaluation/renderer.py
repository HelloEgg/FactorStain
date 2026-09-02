from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from factorstain.losses.factorial import edge_consistency_per_sample
from factorstain.models.dinov3 import load_feature_cache
from factorstain.training.renderer import _load_image, build_renderer


class EvaluationPairDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        index: pd.DataFrame,
        image_size: int,
        stain_map: dict,
        scanner_map: dict,
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.index = index
        self.image_size = image_size
        self.stain_map = stain_map
        self.scanner_map = scanner_map

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        pair = self.pairs.iloc[item]
        source = self.index.loc[int(pair.source_index)]
        target = self.index.loc[int(pair.target_index)]
        return {
            "source": _load_image(source.image_path, self.image_size),
            "target": _load_image(target.image_path, self.image_size),
            "source_stain": self.stain_map[source.stain_id],
            "source_scanner": self.scanner_map[source.scanner_id],
            "target_stain": self.stain_map[target.stain_id],
            "target_scanner": self.scanner_map[target.scanner_id],
            "source_stain_name": str(source.stain_id),
            "source_scanner_name": str(source.scanner_id),
            "target_stain_name": str(target.stain_id),
            "target_scanner_name": str(target.scanner_id),
            "group": str(target.aligned_group_id),
            "target_index": int(pair.target_index),
            "source_index": int(pair.source_index),
            "protocol": str(pair.protocol),
        }


def _cell_set(items: list[dict]) -> set[tuple[str, str]]:
    return {(str(item["stain_id"]), str(item["scanner_id"])) for item in items}


def build_evaluation_pairs(
    index: pd.DataFrame,
    split: dict,
    seed: int = 42,
    limit: int | None = None,
) -> pd.DataFrame:
    """Construct Protocol A, Protocol B, seen controls, and clean Track S pairs."""
    heldout = _cell_set(split.get("test_cells", split["heldout_cells"]))
    train_cells = _cell_set(split["train_cells"])
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    existing = (
        index[index.image_exists].copy() if "image_exists" in index else index.copy()
    )
    existing["cell"] = list(
        zip(existing.stain_id.astype(str), existing.scanner_id.astype(str))
    )
    for group_id, group in existing.groupby("aligned_group_id", sort=True):
        train_sources = group[group.cell.isin(train_cells)]
        if train_sources.empty:
            continue
        for target_index, target in group.iterrows():
            if target.cell in heldout and target.morphology_split == "train":
                protocol = "unseen_combo_seen_morphology"
            elif target.cell in heldout and target.morphology_split == "test":
                protocol = "unseen_combo_unseen_morphology"
            elif target.cell in train_cells and target.morphology_split == "test":
                protocol = "seen_combo_unseen_morphology"
            else:
                continue
            candidates = train_sources[train_sources.index != target_index]
            if candidates.empty:
                continue
            preferred = candidates[
                candidates.stain_id.ne(target.stain_id)
                & candidates.scanner_id.ne(target.scanner_id)
            ]
            chosen_from = preferred if not preferred.empty else candidates
            source_index = int(rng.choice(chosen_from.index.to_numpy()))
            rows.append(
                {
                    "source_index": source_index,
                    "target_index": int(target_index),
                    "protocol": protocol,
                    "aligned_group_id": str(group_id),
                }
            )
            if target.cell in heldout:
                scanner_sources = candidates[
                    candidates.stain_id.eq(target.stain_id)
                    & candidates.scanner_id.ne(target.scanner_id)
                ]
                if not scanner_sources.empty:
                    scanner_source = int(rng.choice(scanner_sources.index.to_numpy()))
                    rows.append(
                        {
                            "source_index": scanner_source,
                            "target_index": int(target_index),
                            "protocol": "controlled_scanner_transfer",
                            "aligned_group_id": str(group_id),
                        }
                    )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = pairs.drop_duplicates(
        ["source_index", "target_index", "protocol"]
    ).reset_index(drop=True)
    if limit:
        pairs = pd.concat(
            [
                frame.sample(min(limit, len(frame)), random_state=seed + number)
                for number, (_, frame) in enumerate(
                    pairs.groupby("protocol", sort=True)
                )
            ],
            ignore_index=True,
        )
    return pairs


class RealAcquisitionProbes:
    """Independent linear acquisition classifiers trained only on real images."""

    trained_on_generated = False

    def __init__(
        self,
        feature_cache: Path,
        metadata: pd.DataFrame,
        train_cells: set[tuple[str, str]],
    ) -> None:
        features, feature_ids, _ = load_feature_cache(feature_cache)
        id_column = "sample_id" if "sample_id" in metadata else "image_id"
        feature_frame = pd.DataFrame(
            {
                id_column: feature_ids.astype(str),
                "feature_position": np.arange(len(feature_ids)),
            }
        )
        merged = feature_frame.merge(
            metadata.assign(**{id_column: metadata[id_column].astype(str)}),
            on=id_column,
            how="inner",
        )
        merged = merged[
            merged.morphology_split.eq("train")
            & merged.apply(
                lambda row: (str(row.stain_id), str(row.scanner_id)) in train_cells,
                axis=1,
            )
        ]
        if merged.empty:
            raise RuntimeError(
                "No cached real training DINOv3 features are available for acquisition probes"
            )
        train_features = features[merged.feature_position.to_numpy()]
        self.stain_encoder = LabelEncoder().fit(metadata.stain_id.astype(str))
        self.scanner_encoder = LabelEncoder().fit(metadata.scanner_id.astype(str))
        self.stain = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        self.scanner = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        self.stain.fit(
            train_features, self.stain_encoder.transform(merged.stain_id.astype(str))
        )
        self.scanner.fit(
            train_features,
            self.scanner_encoder.transform(merged.scanner_id.astype(str)),
        )
        self.training_samples = len(merged)
        self.training_source = "real_training_images_only"

    @staticmethod
    def _complete_probabilities(
        pipeline, features: np.ndarray, classes: int
    ) -> np.ndarray:
        observed = pipeline.predict_proba(features)
        result = np.zeros((len(features), classes), dtype=np.float64)
        result[:, pipeline[-1].classes_.astype(int)] = observed
        return result

    def probabilities(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._complete_probabilities(
                self.stain, features, len(self.stain_encoder.classes_)
            ),
            self._complete_probabilities(
                self.scanner, features, len(self.scanner_encoder.classes_)
            ),
        )


def _hed_distance(generated: np.ndarray, target: np.ndarray) -> float:
    generated_hed, target_hed = _rgb_to_hed(generated), _rgb_to_hed(target)
    generated_stats = np.concatenate(
        [generated_hed[..., :2].mean((0, 1)), generated_hed[..., :2].std((0, 1))]
    )
    target_stats = np.concatenate(
        [target_hed[..., :2].mean((0, 1)), target_hed[..., :2].std((0, 1))]
    )
    return float(np.linalg.norm(generated_stats - target_stats))


def _delta_e(generated: np.ndarray, target: np.ndarray) -> float:
    difference = _rgb_to_lab(generated) - _rgb_to_lab(target)
    return float(np.linalg.norm(difference, axis=-1).mean())


def _rgb_to_hed(image: np.ndarray) -> np.ndarray:
    # Ruifrok-Johnston HED optical-density separation matrix.
    rgb_from_hed = np.asarray(
        [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
        dtype=np.float64,
    )
    optical_density = -np.log(np.clip(image, 1e-6, 1.0))
    return optical_density @ np.linalg.inv(rgb_from_hed).T


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    values = np.clip(image.astype(np.float64), 0, 1)
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    xyz = (
        linear
        @ np.asarray(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        ).T
    )
    xyz = xyz / np.asarray([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3 * delta**2) + 4 / 29,
    )
    return np.stack(
        [
            116 * transformed[..., 1] - 16,
            500 * (transformed[..., 0] - transformed[..., 1]),
            200 * (transformed[..., 1] - transformed[..., 2]),
        ],
        axis=-1,
    )


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    axes = tuple(range(left.ndim - 1)) if left.ndim == 3 else tuple(range(left.ndim))
    mean_left, mean_right = left.mean(axis=axes), right.mean(axis=axes)
    var_left, var_right = left.var(axis=axes), right.var(axis=axes)
    covariance = ((left - mean_left) * (right - mean_right)).mean(axis=axes)
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / (
        (mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2)
    )
    return float(np.mean(score))


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left - right) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def _sharpness_difference(left: np.ndarray, right: np.ndarray) -> float:
    def energy(image: np.ndarray) -> float:
        gray = image.mean(2)
        return float(
            np.mean(np.diff(gray, axis=1) ** 2) + np.mean(np.diff(gray, axis=0) ** 2)
        )

    return abs(energy(left) - energy(right))


def _factor_isolation(
    before_target: float,
    after_target: float,
    before_other: np.ndarray,
    after_other: np.ndarray,
    morphology: float,
) -> float:
    desired_change = np.clip((after_target - before_target + 1.0) / 2.0, 0.0, 1.0)
    other_preservation = 1.0 - 0.5 * np.abs(after_other - before_other).sum()
    return float(
        np.mean(
            [
                desired_change,
                np.clip(other_preservation, 0, 1),
                np.clip(morphology, 0, 1),
            ]
        )
    )


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


@torch.inference_mode()
def evaluate_model(
    model_name: str,
    checkpoint: Path,
    config: dict,
    index: pd.DataFrame,
    pairs: pd.DataFrame,
    dino,
    probes: RealAcquisitionProbes,
    lpips_model,
    device: torch.device,
    preview_limit: int = 15,
) -> tuple[pd.DataFrame, list[dict[str, torch.Tensor]], dict[str, Any]]:
    stain_map = {
        value: position
        for position, value in enumerate(sorted(index.stain_id.unique()))
    }
    scanner_map = {
        value: position
        for position, value in enumerate(sorted(index.scanner_id.unique()))
    }
    dataset = EvaluationPairDataset(
        pairs, index, config["image_size"], stain_map, scanner_map
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(config["batch_size"], 16)),
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = build_renderer(
        model_name,
        len(stain_map),
        len(scanner_map),
        config.get("model_width", 64),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    rows, previews = [], []
    generated_features, real_features, feature_metadata = [], [], []
    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        source_stain = batch["source_stain"].to(device)
        source_scanner = batch["source_scanner"].to(device)
        target_stain = batch["target_stain"].to(device)
        target_scanner = batch["target_scanner"].to(device)
        with _autocast(device):
            generated = model(source, target_stain, target_scanner)
            stain_cf = model(source, target_stain, source_scanner)
            scanner_cf = model(source, source_stain, target_scanner)
            combined = torch.cat([source, target, generated, stain_cf, scanner_cf])
            features = dino(combined)
        generated = generated.float()
        stain_cf = stain_cf.float()
        scanner_cf = scanner_cf.float()
        source_f, target_f, generated_f, stain_f, scanner_f = features.chunk(5)
        source_stain_p, source_scanner_p = probes.probabilities(source_f.cpu().numpy())
        generated_stain_p, generated_scanner_p = probes.probabilities(
            generated_f.cpu().numpy()
        )
        stain_cf_p, stain_cf_scanner_p = probes.probabilities(stain_f.cpu().numpy())
        scanner_cf_stain_p, scanner_cf_p = probes.probabilities(scanner_f.cpu().numpy())
        lpips_values = (
            lpips_model(generated * 2 - 1, target * 2 - 1)
            .flatten()
            .float()
            .cpu()
            .numpy()
        )
        source_generated_cosine = (
            F.cosine_similarity(source_f, generated_f).cpu().numpy()
        )
        target_generated_cosine = (
            F.cosine_similarity(target_f, generated_f).cpu().numpy()
        )
        edge_distance = edge_consistency_per_sample(generated, source).cpu().numpy()
        edge_similarity = 1.0 / (1.0 + edge_distance)
        generated_np = generated.cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
        target_np = target.cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
        source_np = source.cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
        for position in range(len(source)):
            target_stain_idx = int(target_stain[position])
            target_scanner_idx = int(target_scanner[position])
            source_stain_idx = int(source_stain[position])
            source_scanner_idx = int(source_scanner[position])
            stain_morphology = float(
                F.cosine_similarity(source_f[position], stain_f[position], dim=0)
                .clamp(0, 1)
                .item()
            )
            scanner_morphology = float(
                F.cosine_similarity(source_f[position], scanner_f[position], dim=0)
                .clamp(0, 1)
                .item()
            )
            stain_isolation = _factor_isolation(
                source_stain_p[position, target_stain_idx],
                stain_cf_p[position, target_stain_idx],
                source_scanner_p[position],
                stain_cf_scanner_p[position],
                stain_morphology,
            )
            scanner_isolation = _factor_isolation(
                source_scanner_p[position, target_scanner_idx],
                scanner_cf_p[position, target_scanner_idx],
                source_stain_p[position],
                scanner_cf_stain_p[position],
                scanner_morphology,
            )
            morphology_score = float(
                0.5 * np.clip(source_generated_cosine[position], 0, 1)
                + 0.5 * edge_similarity[position]
            )
            row = {
                "method": model_name,
                "protocol": batch["protocol"][position],
                "aligned_group_id": batch["group"][position],
                "source_index": int(batch["source_index"][position]),
                "target_index": int(batch["target_index"][position]),
                "source_stain_id": batch["source_stain_name"][position],
                "source_scanner_id": batch["source_scanner_name"][position],
                "target_stain_id": batch["target_stain_name"][position],
                "target_scanner_id": batch["target_scanner_name"][position],
                "l1": float(
                    np.abs(generated_np[position] - target_np[position]).mean()
                ),
                "lpips": float(lpips_values[position]),
                "ssim": float(_ssim(target_np[position], generated_np[position])),
                "psnr": _psnr(target_np[position], generated_np[position]),
                "dino_source_generated_cosine": float(
                    source_generated_cosine[position]
                ),
                "dino_target_generated_cosine": float(
                    target_generated_cosine[position]
                ),
                "edge_similarity": float(edge_similarity[position]),
                "grayscale_structure": float(
                    _ssim(
                        source_np[position].mean(2),
                        generated_np[position].mean(2),
                    )
                ),
                "morphology_preservation": morphology_score,
                "he_statistics_distance": _hed_distance(
                    generated_np[position], target_np[position]
                ),
                "color_delta_e": _delta_e(generated_np[position], target_np[position]),
                "sharpness_difference": _sharpness_difference(
                    generated_np[position], target_np[position]
                ),
                "stain_target_accuracy": float(
                    generated_stain_p[position].argmax() == target_stain_idx
                ),
                "scanner_target_accuracy": float(
                    generated_scanner_p[position].argmax() == target_scanner_idx
                ),
                "stain_preservation_accuracy": float(
                    generated_stain_p[position].argmax() == source_stain_idx
                ),
                "scanner_preservation_accuracy": float(
                    generated_scanner_p[position].argmax() == source_scanner_idx
                ),
                "stain_isolation_score": stain_isolation,
                "scanner_isolation_score": scanner_isolation,
                "factor_isolation_score": float(
                    0.5 * (stain_isolation + scanner_isolation)
                ),
            }
            rows.append(row)
            generated_features.append(generated_f[position].cpu().numpy())
            real_features.append(target_f[position].cpu().numpy())
            feature_metadata.append(
                {
                    "method": model_name,
                    "protocol": row["protocol"],
                    "aligned_group_id": row["aligned_group_id"],
                    "target_stain_id": row["target_stain_id"],
                    "target_scanner_id": row["target_scanner_id"],
                }
            )
            if len(previews) < preview_limit:
                previews.append(
                    {
                        "source": source[position].cpu(),
                        "target": target[position].cpu(),
                        "generated": generated[position].cpu(),
                        "stain_cf": stain_cf[position].cpu(),
                        "scanner_cf": scanner_cf[position].cpu(),
                        "source_index": row["source_index"],
                        "target_index": row["target_index"],
                        "protocol": row["protocol"],
                    }
                )
    feature_bundle = {
        "generated": np.asarray(generated_features, dtype=np.float32),
        "real": np.asarray(real_features, dtype=np.float32),
        "metadata": pd.DataFrame(feature_metadata),
    }
    return pd.DataFrame(rows), previews, feature_bundle
