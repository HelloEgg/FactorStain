#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from factorstain.baselines.benchmark import benchmark_output
from factorstain.baselines.features import pathorob_inspired_index
from factorstain.baselines.registry import TIER_METHODS, get_baseline, resolve_methods
from factorstain.evaluation.mmd import (
    compute_mmd2,
    controlled_scanner_indices,
    l2_normalize,
    stable_seed,
)
from factorstain.evaluation.renderer import (
    _delta_e,
    _factor_isolation,
    _hed_distance,
    _psnr,
    _rgb_to_lab,
    _sharpness_difference,
    _ssim,
)
from factorstain.metrics.statistics import paired_bootstrap
from factorstain.models.dinov3 import (
    load_feature_cache,
    load_frozen_dinov3_encoder,
)
from factorstain.utils.config import load_config
from factorstain.utils.runtime import atomic_json_dump

PRIMARY_COLUMNS = [
    "Method",
    "Registry Method",
    "Type",
    "Strict?",
    "External pretraining?",
    "Status",
    "Seeds",
    "DINO-MMD ↓",
    "DINO-MMD std",
    "Target Stain BA ↑",
    "Target Stain BA std",
    "Target Scanner BA ↑",
    "Target Scanner BA std",
    "Morphology ↑",
    "Morphology std",
    "OD Distance ↓",
    "OD Distance std",
    "Factor Isolation ↑",
    "Factor Isolation std",
    "Tissue Consistency ↑",
    "Tissue Consistency std",
    "Composite Score ↑",
    "Composite Score std",
    "Composite Domain-heavy ↑",
    "Composite Biology-heavy ↑",
    "GPU Hours",
]

SEEDED_IMAGE_METHODS = {
    "stainnet",
    "staingan",
    "cyclegan",
    "pix2pix",
    "histaugan",
    "cagan",
    "sastaindiff",
    "joint",
    "parallel",
    "factorstain",
}


def _planned_seed_count(config: dict, method: str) -> int:
    return (
        len(config.get("run_seeds", config["seeds"]))
        if method in SEEDED_IMAGE_METHODS
        else 1
    )


def _resolve(config: dict, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project = Path(config["paths"]["project_root"])
    candidate = project / path
    if candidate.exists():
        return candidate
    if path.parts and path.parts[0] == "outputs":
        return Path(config["paths"]["outputs_root"]).joinpath(*path.parts[1:])
    return candidate


def _output(config: dict) -> Path:
    return benchmark_output(config)


def _load_rgb(path: str, size: int) -> np.ndarray:
    with Image.open(path) as opened:
        return (
            np.asarray(
                opened.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )


def _edge_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_gray, right_gray = left.mean(2), right.mean(2)
    difference = np.mean(
        np.abs(np.diff(left_gray, axis=0) - np.diff(right_gray, axis=0))
    )
    difference += np.mean(
        np.abs(np.diff(left_gray, axis=1) - np.diff(right_gray, axis=1))
    )
    return float(1 / (1 + difference))


def _lab_histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_lab, right_lab = _rgb_to_lab(left), _rgb_to_lab(right)
    ranges = [(0, 100), (-128, 127), (-128, 127)]
    distances = []
    for channel, value_range in enumerate(ranges):
        left_hist, _ = np.histogram(
            left_lab[..., channel], 32, value_range, density=True
        )
        right_hist, _ = np.histogram(
            right_lab[..., channel], 32, value_range, density=True
        )
        distances.append(np.abs(left_hist - right_hist).mean())
    return float(np.mean(distances))


def _frequency_error(left: np.ndarray, right: np.ndarray) -> float:
    def spectrum(image: np.ndarray) -> np.ndarray:
        value = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image.mean(2)))))
        return value / np.clip(np.linalg.norm(value), 1e-8, None)

    return float(np.mean(np.abs(spectrum(left) - spectrum(right))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right)
        / np.clip(np.linalg.norm(left) * np.linalg.norm(right), 1e-12, None)
    )


@torch.inference_mode()
def _lpips_values(
    generation: pd.DataFrame, config: dict
) -> tuple[dict[str, float], str]:
    try:
        import lpips

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
        output: dict[str, float] = {}
        batch_size = max(1, min(config["batch_size"], 16))
        records = list(generation.itertuples(index=False))
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            generated = torch.stack(
                [
                    torch.from_numpy(
                        _load_rgb(row.generated_path, config["image_size"])
                    )
                    .permute(2, 0, 1)
                    .mul(2)
                    .sub(1)
                    for row in batch
                ]
            ).to(device)
            target = torch.stack(
                [
                    torch.from_numpy(_load_rgb(row.target_path, config["image_size"]))
                    .permute(2, 0, 1)
                    .mul(2)
                    .sub(1)
                    for row in batch
                ]
            ).to(device)
            values = model(generated, target).flatten().float().cpu().numpy()
            output.update(
                (str(row.generated_path), float(value))
                for row, value in zip(batch, values, strict=True)
            )
        return output, ""
    except Exception as exc:  # noqa: BLE001 - optional evaluator is reported, never imputed
        return {}, f"LPIPS unavailable: {type(exc).__name__}: {exc}"


class RealOnlyProbes:
    def __init__(
        self,
        features: np.ndarray,
        sample_ids: np.ndarray,
        full_index: pd.DataFrame,
        training: pd.DataFrame,
    ) -> None:
        id_column = "sample_id" if "sample_id" in full_index else "image_id"
        positions = pd.DataFrame(
            {
                id_column: sample_ids.astype(str),
                "feature_position": np.arange(len(features)),
            }
        )
        train_ids = set(training[id_column].astype(str))
        merged = positions.merge(
            full_index.assign(**{id_column: full_index[id_column].astype(str)}),
            on=id_column,
            how="inner",
        )
        merged = merged[merged[id_column].isin(train_ids)]
        self.encoders, self.models = {}, {}
        values = features[merged.feature_position.to_numpy()]
        for name, column in (
            ("stain", "stain_id"),
            ("scanner", "scanner_id"),
            ("tissue", "tissue_type"),
        ):
            encoder = LabelEncoder().fit(full_index[column].astype(str))
            targets = encoder.transform(merged[column].astype(str))
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42
                ),
            )
            model.fit(values, targets)
            self.encoders[name], self.models[name] = encoder, model
        self.training_source = "real_M1_training_images_only"

    def probabilities(self, name: str, features: np.ndarray) -> np.ndarray:
        model, encoder = self.models[name], self.encoders[name]
        observed = model.predict_proba(features)
        result = np.zeros((len(features), len(encoder.classes_)), dtype=np.float64)
        result[:, model[-1].classes_.astype(int)] = observed
        return result


def _real_feature_material(config: dict, out: Path):
    cache = _resolve(config, os.getenv("DINOV3_CACHE", config["dinov3_cache"]))
    index_path = _resolve(config, os.getenv("M1_INDEX", config["m1_index"]))
    if not cache.exists() or not index_path.exists():
        raise FileNotFoundError(f"DINO cache/index unavailable: {cache}, {index_path}")
    features, sample_ids, metadata = load_feature_cache(cache)
    index = pd.read_parquet(index_path).reset_index(drop=True)
    training = pd.read_parquet(out / "metadata" / "training_manifest.parquet")
    feature_lookup = {str(value): position for position, value in enumerate(sample_ids)}
    probes = RealOnlyProbes(features, sample_ids, index, training)
    return features, sample_ids, metadata, index, feature_lookup, probes


@torch.inference_mode()
def _encode_paths(config: dict, paths: list[str]) -> dict[str, np.ndarray]:
    unique = sorted({path for path in paths if path and Path(path).exists()})
    if not unique:
        return {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not config["fast_dev_run"]:
        raise RuntimeError("Full DINOv3 image evaluation requires CUDA")
    encoder = load_frozen_dinov3_encoder(
        config["dinov3_model_name"], device, config["dinov3_dtype"]
    )
    output: dict[str, np.ndarray] = {}
    batch_size = max(1, min(config["batch_size"], 32))
    for start in range(0, len(unique), batch_size):
        selected = unique[start : start + batch_size]
        tensors = []
        for path in selected:
            image = _load_rgb(path, config["image_size"])
            tensors.append(torch.from_numpy(image).permute(2, 0, 1))
        values = encoder(torch.stack(tensors).to(device)).cpu().numpy()
        output.update(zip(selected, values, strict=True))
    return output


def _status_records(out: Path, run_seeds: list[int] | None = None) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    selected_seeds = set(run_seeds or [])
    for path in sorted((out / "logs").glob("method_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            selected_seeds
            and payload["method"] in SEEDED_IMAGE_METHODS
            and payload.get("seed") not in selected_seeds
        ):
            continue
        grouped.setdefault(payload["method"], []).append(payload)
    records = {}
    for method, payloads in grouped.items():
        if len(payloads) == 1:
            records[method] = payloads[0]
            continue
        states = [payload.get("status", "FAILED") for payload in payloads]
        records[method] = {
            **payloads[-1],
            "status": (
                "COMPLETE"
                if all(state == "COMPLETE" for state in states)
                else "PARTIAL"
                if "COMPLETE" in states
                else states[-1]
            ),
            "seed_statuses": {
                str(payload.get("seed")): payload.get("status") for payload in payloads
            },
            "wall_seconds": sum(payload.get("wall_seconds", 0) for payload in payloads),
        }
    return records


def _generation_records(
    out: Path, selected: list[str], run_seeds: list[int] | None = None
) -> pd.DataFrame:
    frames = []
    selected_seeds = set(run_seeds or [])
    for method in selected:
        for path in sorted(
            (out / "generated" / method).glob("generation_manifest_seed*.csv")
        ):
            if path.exists() and path.stat().st_size:
                frame = pd.read_csv(path)
                if (
                    selected_seeds
                    and method in SEEDED_IMAGE_METHODS
                    and "seed" in frame
                ):
                    frame = frame[frame.seed.isin(selected_seeds)]
                if not frame.empty:
                    frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _evaluate_images(
    generation: pd.DataFrame, config: dict, out: Path
) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    if generation.empty:
        return pd.DataFrame(), {}, "No generated image manifests were available."
    dino_paths = generation.generated_path.astype(str).tolist()
    for column in ("stain_cf_path", "scanner_cf_path"):
        if column in generation:
            dino_paths.extend(generation[column].fillna("").astype(str).tolist())
    dino_error = ""
    lpips_by_path, lpips_error = _lpips_values(generation, config)
    try:
        _, _, _, _, lookup, probes = _real_feature_material(config, out)
        real_features, _, _, _, _, _ = _real_feature_material(config, out)
        generated_features = _encode_paths(config, dino_paths)
    except Exception as exc:  # noqa: BLE001 - aggregate optional evaluators fail closed
        lookup, probes, generated_features = {}, None, {}
        real_features = np.empty((0, 0))
        dino_error = f"DINO/probe evaluation unavailable: {type(exc).__name__}: {exc}"
    rows = []
    for record in generation.itertuples(index=False):
        source = _load_rgb(record.source_path, config["image_size"])
        target = _load_rgb(record.target_path, config["image_size"])
        generated = _load_rgb(record.generated_path, config["image_size"])
        edge = _edge_similarity(source, generated)
        grayscale = _ssim(source.mean(2), generated.mean(2))
        row = {
            **record._asdict(),
            "ssim": _ssim(target, generated),
            "psnr": _psnr(target, generated),
            "lpips": lpips_by_path.get(str(record.generated_path), np.nan),
            "color_delta_e": _delta_e(generated, target),
            "od_distance": _hed_distance(generated, target),
            "lab_histogram_distance": _lab_histogram_distance(generated, target),
            "edge_similarity": edge,
            "grayscale_structure": grayscale,
            "sharpness_error": _sharpness_difference(generated, target),
            "frequency_spectrum_error": _frequency_error(generated, target),
            "dino_source_generated_cosine": np.nan,
            "dino_target_generated_cosine": np.nan,
            "morphology_preservation": float(0.5 * max(0, grayscale) + 0.5 * edge),
            "stain_target_correct": np.nan,
            "scanner_target_correct": np.nan,
            "tissue_consistency": np.nan,
            "factor_isolation": np.nan,
        }
        generated_feature = generated_features.get(str(record.generated_path))
        source_position = lookup.get(str(record.source_sample_id))
        target_position = lookup.get(str(record.target_sample_id))
        if (
            generated_feature is not None
            and source_position is not None
            and target_position is not None
        ):
            source_feature = real_features[source_position]
            target_feature = real_features[target_position]
            source_cosine = _cosine(source_feature, generated_feature)
            target_cosine = _cosine(target_feature, generated_feature)
            row["dino_source_generated_cosine"] = source_cosine
            row["dino_target_generated_cosine"] = target_cosine
            row["morphology_preservation"] = float(
                0.5 * np.clip(source_cosine, 0, 1)
                + 0.25 * edge
                + 0.25 * max(0, grayscale)
            )
            stain_probability = probes.probabilities("stain", generated_feature[None])[
                0
            ]
            scanner_probability = probes.probabilities(
                "scanner", generated_feature[None]
            )[0]
            tissue_probability = probes.probabilities(
                "tissue", generated_feature[None]
            )[0]
            stain_encoder = probes.encoders["stain"]
            scanner_encoder = probes.encoders["scanner"]
            tissue_encoder = probes.encoders["tissue"]
            target_stain = int(
                stain_encoder.transform([str(record.target_stain_id)])[0]
            )
            target_scanner = int(
                scanner_encoder.transform([str(record.target_scanner_id)])[0]
            )
            target_tissue = int(tissue_encoder.transform([str(record.tissue_type)])[0])
            row["stain_target_correct"] = float(
                stain_probability.argmax() == target_stain
            )
            row["scanner_target_correct"] = float(
                scanner_probability.argmax() == target_scanner
            )
            row["tissue_consistency"] = float(
                tissue_probability.argmax() == target_tissue
            )
            stain_path = str(getattr(record, "stain_cf_path", "") or "")
            scanner_path = str(getattr(record, "scanner_cf_path", "") or "")
            if stain_path in generated_features and scanner_path in generated_features:
                source_stain_p = probes.probabilities("stain", source_feature[None])[0]
                source_scanner_p = probes.probabilities(
                    "scanner", source_feature[None]
                )[0]
                stain_feature = generated_features[stain_path]
                scanner_feature = generated_features[scanner_path]
                stain_p = probes.probabilities("stain", stain_feature[None])[0]
                stain_scanner_p = probes.probabilities("scanner", stain_feature[None])[
                    0
                ]
                scanner_stain_p = probes.probabilities("stain", scanner_feature[None])[
                    0
                ]
                scanner_p = probes.probabilities("scanner", scanner_feature[None])[0]
                stain_morph = np.clip(_cosine(source_feature, stain_feature), 0, 1)
                scanner_morph = np.clip(_cosine(source_feature, scanner_feature), 0, 1)
                stain_isolation = _factor_isolation(
                    source_stain_p[target_stain],
                    stain_p[target_stain],
                    source_scanner_p,
                    stain_scanner_p,
                    stain_morph,
                )
                scanner_isolation = _factor_isolation(
                    source_scanner_p[target_scanner],
                    scanner_p[target_scanner],
                    source_stain_p,
                    scanner_stain_p,
                    scanner_morph,
                )
                row["factor_isolation"] = 0.5 * (stain_isolation + scanner_isolation)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(out / "tables" / "per_episode_metrics.parquet", index=False)
    evaluator_note = "; ".join(value for value in (dino_error, lpips_error) if value)
    return metrics, generated_features, evaluator_note


def _mmd_rows(
    metrics: pd.DataFrame,
    generated_features: dict[str, np.ndarray],
    config: dict,
    out: Path,
) -> pd.DataFrame:
    if metrics.empty or not generated_features:
        return pd.DataFrame()
    try:
        real_features, _, _, _, lookup, _ = _real_feature_material(config, out)
    except Exception:  # noqa: BLE001 - missing optional DINO evaluator yields empty MMD
        return pd.DataFrame()
    rows = []
    strict = metrics[metrics.track.eq("C")]
    for keys, frame in strict.groupby(
        [
            "method",
            "display_name",
            "seed",
            "heldout_target_stain_id",
            "heldout_target_scanner_id",
        ],
        sort=True,
    ):
        generated, target = [], []
        for row in frame.itertuples():
            feature = generated_features.get(str(row.generated_path))
            position = lookup.get(str(row.target_sample_id))
            if feature is not None and position is not None:
                generated.append(feature)
                target.append(real_features[position])
        if len(generated) < 2:
            value, count = np.nan, len(generated)
        else:
            result = compute_mmd2(
                l2_normalize(np.asarray(generated)),
                l2_normalize(np.asarray(target)),
                seed=stable_seed(config["seed"], *keys),
            )
            value, count = result["reported_mmd2"], len(generated)
        rows.append(
            {
                "method": keys[0],
                "display_name": keys[1],
                "seed": keys[2],
                "target_stain_id": keys[3],
                "target_scanner_id": keys[4],
                "combination_id": f"{keys[3]}x{keys[4]}",
                "dino_mmd2": value,
                "n": count,
            }
        )
    return pd.DataFrame(rows)


def _composite(components: dict[str, float], weights: dict[str, float]) -> float:
    available = [
        (components[key], weight)
        for key, weight in weights.items()
        if np.isfinite(components.get(key, np.nan))
    ]
    if not available:
        return np.nan
    return float(
        np.average(
            [value for value, _ in available],
            weights=[weight for _, weight in available],
        )
    )


def _macro_primary(
    metrics: pd.DataFrame,
    mmd: pd.DataFrame,
    selected: list[str],
    statuses: dict[str, dict],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = (
        metrics[metrics.track.eq("C")].copy() if not metrics.empty else pd.DataFrame()
    )
    combinations = []
    if not strict.empty:
        strict["combination_id"] = (
            strict.heldout_target_stain_id.astype(str)
            + "x"
            + strict.heldout_target_scanner_id.astype(str)
        )
        combinations = (
            strict.groupby(
                ["method", "display_name", "seed", "combination_id"], sort=True
            )
            .agg(
                target_stain_ba=("stain_target_correct", "mean"),
                target_scanner_ba=("scanner_target_correct", "mean"),
                morphology_preservation=("morphology_preservation", "mean"),
                od_distance=("od_distance", "mean"),
                factor_isolation=("factor_isolation", "mean"),
                tissue_consistency=("tissue_consistency", "mean"),
                sample_count=("episode_id", "nunique"),
            )
            .reset_index()
        )
        if not mmd.empty:
            combinations = combinations.merge(
                mmd[["method", "seed", "combination_id", "dino_mmd2"]],
                on=["method", "seed", "combination_id"],
                how="left",
            )
    per_combination = pd.DataFrame(combinations)
    rows = []
    if not per_combination.empty:
        for (method, display), frame in per_combination.groupby(
            ["method", "display_name"], sort=True
        ):
            values = frame.mean(numeric_only=True)
            per_seed = frame.groupby("seed", sort=True).mean(numeric_only=True)
            standard = per_seed.std(ddof=1).fillna(0.0)
            spec = get_baseline(method)
            components = {
                "dino_mmd2": 1 / (1 + values.get("dino_mmd2", np.nan)),
                "target_stain_ba": values.get("target_stain_ba", np.nan),
                "target_scanner_ba": values.get("target_scanner_ba", np.nan),
                "morphology_preservation": values.get(
                    "morphology_preservation", np.nan
                ),
                "od_distance": 1 / (1 + values.get("od_distance", np.nan)),
                "factor_isolation": values.get("factor_isolation", np.nan),
                "tissue_consistency": values.get("tissue_consistency", np.nan),
            }
            seed_composites = []
            for _, seed_values in per_seed.iterrows():
                seed_components = {
                    "dino_mmd2": 1 / (1 + seed_values.get("dino_mmd2", np.nan)),
                    "target_stain_ba": seed_values.get("target_stain_ba", np.nan),
                    "target_scanner_ba": seed_values.get("target_scanner_ba", np.nan),
                    "morphology_preservation": seed_values.get(
                        "morphology_preservation", np.nan
                    ),
                    "od_distance": 1 / (1 + seed_values.get("od_distance", np.nan)),
                    "factor_isolation": seed_values.get("factor_isolation", np.nan),
                    "tissue_consistency": seed_values.get("tissue_consistency", np.nan),
                }
                seed_composites.append(
                    _composite(seed_components, config["composite"]["weights"])
                )

            budget = config.get("compute_budget", {}).get(method, {})
            rows.append(
                {
                    "Method": display,
                    "Registry Method": method,
                    "Type": spec.method_type,
                    "Strict?": True,
                    "External pretraining?": spec.external_pretraining,
                    "Status": statuses.get(method, {}).get("status", "COMPLETE"),
                    "Seeds": ",".join(
                        str(value) for value in sorted(frame.seed.unique())
                    ),
                    "DINO-MMD ↓": values.get("dino_mmd2", np.nan),
                    "DINO-MMD std": standard.get("dino_mmd2", 0.0),
                    "Target Stain BA ↑": values.get("target_stain_ba", np.nan),
                    "Target Stain BA std": standard.get("target_stain_ba", 0.0),
                    "Target Scanner BA ↑": values.get("target_scanner_ba", np.nan),
                    "Target Scanner BA std": standard.get("target_scanner_ba", 0.0),
                    "Morphology ↑": values.get("morphology_preservation", np.nan),
                    "Morphology std": standard.get("morphology_preservation", 0.0),
                    "OD Distance ↓": values.get("od_distance", np.nan),
                    "OD Distance std": standard.get("od_distance", 0.0),
                    "Factor Isolation ↑": values.get("factor_isolation", np.nan),
                    "Factor Isolation std": standard.get("factor_isolation", 0.0),
                    "Tissue Consistency ↑": values.get("tissue_consistency", np.nan),
                    "Tissue Consistency std": standard.get("tissue_consistency", 0.0),
                    "Composite Score ↑": _composite(
                        components, config["composite"]["weights"]
                    ),
                    "Composite Score std": float(np.nanstd(seed_composites, ddof=1))
                    if len(seed_composites) > 1
                    else 0.0,
                    "Composite Domain-heavy ↑": _composite(
                        components,
                        config["composite"]["sensitivity_weights"]["domain_fidelity"],
                    ),
                    "Composite Biology-heavy ↑": _composite(
                        components,
                        config["composite"]["sensitivity_weights"][
                            "biology_preservation"
                        ],
                    ),
                    "GPU Hours": budget.get("gpu_hours", np.nan)
                    * _planned_seed_count(config, method),
                }
            )
    present = {row["Registry Method"] for row in rows}
    for method in selected:
        spec = get_baseline(method)
        if not spec.supports_unseen_composition or method in present:
            continue
        rows.append(
            {
                "Method": (
                    f"{spec.display_name} + ScannerLUT (compositional adaptation)"
                    if method
                    in {
                        "histogram",
                        "reinhard",
                        "macenko",
                        "vahadane",
                        "stainnet",
                        "staingan",
                        "cyclegan",
                        "histaugan",
                        "cagan",
                        "sastaindiff",
                    }
                    else spec.display_name
                ),
                "Registry Method": method,
                "Type": spec.method_type,
                "Strict?": True,
                "External pretraining?": spec.external_pretraining,
                "Status": statuses.get(method, {}).get("status", "NOT_RUN"),
                "Seeds": ",".join(
                    str(seed) for seed in config.get("run_seeds", config["seeds"])
                )
                if method in SEEDED_IMAGE_METHODS
                else str(config["seed"]),
                "GPU Hours": config.get("compute_budget", {})
                .get(method, {})
                .get("gpu_hours", np.nan)
                * _planned_seed_count(config, method),
            }
        )
    primary = pd.DataFrame(rows).reindex(columns=PRIMARY_COLUMNS)
    return primary, per_combination


def _track_table(metrics: pd.DataFrame, track: str) -> pd.DataFrame:
    if metrics.empty or not metrics.track.eq(track).any():
        return pd.DataFrame()
    frame = metrics[metrics.track.eq(track)]
    if track == "A":
        columns = {
            "od_distance": "H/E OD Distance ↓",
            "stain_target_correct": "Target Stain BA ↑",
            "lab_histogram_distance": "Lab Histogram Distance ↓",
            "morphology_preservation": "Morphology ↑",
            "dino_source_generated_cosine": "DINO Morphology Similarity ↑",
            "edge_similarity": "Edge Structure ↑",
            "tissue_consistency": "Tissue Consistency ↑",
        }
    else:
        columns = {
            "psnr": "PSNR ↑",
            "ssim": "SSIM ↑",
            "lpips": "LPIPS ↓",
            "color_delta_e": "DeltaE ↓",
            "dino_target_generated_cosine": "DINO Similarity ↑",
            "edge_similarity": "Edge Similarity ↑",
            "sharpness_error": "Sharpness Error ↓",
            "frequency_spectrum_error": "Frequency Error ↓",
            "scanner_target_correct": "Scanner Target BA ↑",
        }
    result = (
        frame.groupby(["method", "display_name"], sort=True)[list(columns)]
        .mean()
        .reset_index()
    )
    return result.rename(columns={"display_name": "Method", **columns})


def _probe_score(
    features: np.ndarray,
    metadata: pd.DataFrame,
    label: str,
    train: np.ndarray,
    test: np.ndarray,
) -> float:
    encoder = LabelEncoder().fit(metadata[label].astype(str))
    targets = encoder.transform(metadata[label].astype(str))
    if len(np.unique(targets[train])) < 2 or not len(test):
        return np.nan
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    model.fit(features[train], targets[train])
    return float(balanced_accuracy_score(targets[test], model.predict(features[test])))


def _same_morphology_distance(features: np.ndarray, metadata: pd.DataFrame) -> float:
    values = l2_normalize(features)
    distances = []
    for positions in metadata.groupby(
        ["aligned_group_id", "stain_id"], sort=True
    ).groups.values():
        positions = np.asarray(list(positions), dtype=int)
        if len(positions) < 2:
            continue
        scanner = metadata.iloc[positions].scanner_id.astype(str).to_numpy()
        matrix = 1 - values[positions] @ values[positions].T
        mask = scanner[:, None] != scanner[None, :]
        if mask.any():
            distances.append(float(matrix[mask].mean()))
    return float(np.mean(distances)) if distances else np.nan


def _nn_tissue_retrieval(
    features: np.ndarray, metadata: pd.DataFrame, train: np.ndarray, test: np.ndarray
) -> float:
    if not len(train) or not len(test):
        return np.nan
    rng = np.random.default_rng(42)
    train = rng.choice(train, min(3000, len(train)), replace=False)
    test = rng.choice(test, min(2000, len(test)), replace=False)
    values = l2_normalize(features)
    nearest = train[(values[test] @ values[train].T).argmax(1)]
    labels = metadata.tissue_type.astype(str).to_numpy()
    return float(np.mean(labels[test] == labels[nearest]))


def _tissue_separation(
    features: np.ndarray, metadata: pd.DataFrame, test: np.ndarray
) -> float:
    frame = metadata.iloc[test]
    centroids = []
    for positions in frame.groupby("tissue_type", sort=True).groups.values():
        centroids.append(features[np.asarray(list(positions), dtype=int)].mean(0))
    if len(centroids) < 2:
        return np.nan
    centroids = l2_normalize(np.asarray(centroids))
    return float(
        (1 - centroids @ centroids.T)[np.triu_indices(len(centroids), 1)].mean()
    )


def _canonical_feature_sets(
    config: dict,
    out: Path,
    selected: list[str],
    metadata: pd.DataFrame,
    id_column: str,
) -> dict[str, tuple[str, int, np.ndarray]]:
    candidates: list[tuple[str, str, int, list[str]]] = []
    all_paths: list[str] = []
    required_ids = metadata[id_column].astype(str).tolist()
    for name in ("parallel", "factorstain"):
        if name not in selected:
            continue
        for manifest_path in sorted(
            (out / "generated" / name).glob("canonical_manifest_seed*.csv")
        ):
            manifest = pd.read_csv(manifest_path, dtype={id_column: str})
            seed = int(manifest.seed.iloc[0])
            if seed not in set(config.get("run_seeds", config["seeds"])):
                continue
            mapping = dict(
                zip(
                    manifest[id_column].astype(str),
                    manifest.generated_path.astype(str),
                    strict=True,
                )
            )
            if not all(value in mapping for value in required_ids):
                continue
            paths = [mapping[value] for value in required_ids]
            candidates.append((f"{name}_seed{seed}", name, seed, paths))
            all_paths.extend(paths)
    encoded = _encode_paths(config, all_paths) if candidates else {}
    result = {}
    for key, name, seed, paths in candidates:
        if all(path in encoded for path in paths):
            result[key] = (name, seed, np.stack([encoded[path] for path in paths]))
    return result


def _feature_table(
    config: dict, out: Path, selected: list[str], statuses: dict[str, dict]
) -> pd.DataFrame:
    rows = []
    try:
        raw, sample_ids, _, index, _, _ = _real_feature_material(config, out)
        id_column = "sample_id" if "sample_id" in index else "image_id"
        positions = pd.DataFrame(
            {id_column: sample_ids.astype(str), "feature_position": np.arange(len(raw))}
        )
        metadata = (
            positions.merge(
                index.assign(**{id_column: index[id_column].astype(str)}),
                on=id_column,
                how="inner",
            )
            .sort_values("feature_position")
            .reset_index(drop=True)
        )
        raw = raw[metadata.feature_position.to_numpy()]
        train_ids = set(
            pd.read_parquet(out / "metadata" / "training_manifest.parquet")[
                id_column
            ].astype(str)
        )
        train = np.flatnonzero(
            metadata[id_column].astype(str).isin(train_ids).to_numpy()
        )
        test = np.flatnonzero(metadata.morphology_split.eq("test").to_numpy())
        methods: dict[str, tuple[str, int | None, np.ndarray]] = {
            "raw_dinov3": ("raw_dinov3", None, raw)
        }
        for name in ("featmap", "scangen"):
            path = out / "generated" / name / "features.npz"
            if path.exists():
                values, ids, _ = load_feature_cache(path)
                mapping = {str(value): position for position, value in enumerate(ids)}
                if all(str(value) in mapping for value in metadata[id_column]):
                    methods[name] = (
                        name,
                        None,
                        values[[mapping[str(value)] for value in metadata[id_column]]],
                    )
        methods.update(
            _canonical_feature_sets(config, out, selected, metadata, id_column)
        )
        raw_separation = _tissue_separation(raw, metadata, test)
        for key, (name, seed, values) in methods.items():
            scanner_ba = _probe_score(values, metadata, "scanner_id", train, test)
            stain_ba = _probe_score(values, metadata, "stain_id", train, test)
            tissue_ba = _probe_score(values, metadata, "tissue_type", train, test)
            controlled_mmd = []
            for scanner_a, scanner_b in __import__("itertools").combinations(
                sorted(metadata.scanner_id.astype(str).unique()), 2
            ):
                try:
                    left, right, _, _ = controlled_scanner_indices(
                        metadata.iloc[test].reset_index(drop=True),
                        scanner_a,
                        scanner_b,
                        2000,
                        stable_seed(42, key, scanner_a, scanner_b),
                    )
                    test_values = values[test]
                    controlled_mmd.append(
                        compute_mmd2(
                            l2_normalize(test_values[left]),
                            l2_normalize(test_values[right]),
                            seed=stable_seed(42, key, scanner_a, scanner_b),
                        )["reported_mmd2"]
                    )
                except ValueError:
                    continue
            separation = _tissue_separation(values, metadata, test)
            rows.append(
                {
                    "Method": {
                        "raw_dinov3": "Raw DINOv3",
                        "featmap": "DINOv3 + FEATMAP",
                        "scangen": "DINOv3 + ScanGen",
                        "parallel": "OURS-Parallel canonicalized → DINOv3",
                        "factorstain": "OURS-Ordered canonicalized → DINOv3",
                    }[name],
                    "Registry Method": name,
                    "Status": "COMPLETE",
                    "Seeds": str(seed) if seed is not None else "42",
                    "Scanner Probe BA ↓": scanner_ba,
                    "Stain Probe BA ↓": stain_ba,
                    "Tissue Probe BA ↑": tissue_ba,
                    "Controlled Scanner MMD² ↓": np.mean(controlled_mmd)
                    if controlled_mmd
                    else np.nan,
                    "Aligned Cross-scanner Distance ↓": _same_morphology_distance(
                        values[test], metadata.iloc[test].reset_index(drop=True)
                    ),
                    "Nearest-neighbor Tissue Retrieval ↑": _nn_tissue_retrieval(
                        values, metadata, train, test
                    ),
                    "Between-tissue Separation Preserved ↑": separation / raw_separation
                    if raw_separation and np.isfinite(raw_separation)
                    else np.nan,
                    "PathoROB-inspired Index ↑": pathorob_inspired_index(
                        tissue_ba,
                        scanner_ba,
                        metadata.tissue_type.nunique(),
                        metadata.scanner_id.nunique(),
                    ),
                }
            )
        measured = pd.DataFrame(rows)
        aggregated = []
        for name, frame in measured.groupby("Registry Method", sort=False):
            if len(frame) == 1:
                aggregated.append(frame.iloc[0].to_dict())
                continue
            row = frame.iloc[0].to_dict()
            row["Seeds"] = ",".join(sorted(frame.Seeds.astype(str)))
            for column in frame.select_dtypes(include=[np.number]).columns:
                row[column] = float(frame[column].mean())
                row[f"{column} std"] = float(frame[column].std(ddof=1))
            aggregated.append(row)
        rows = aggregated
    except Exception as exc:  # noqa: BLE001 - feature track reports missing optional data
        rows.append(
            {
                "Method": "Raw DINOv3",
                "Registry Method": "raw_dinov3",
                "Status": f"UNAVAILABLE: {exc}",
            }
        )
    feature_declared = {
        "histofs": "NOT_APPLICABLE_TO_TILE_IMAGE_GENERATION; official method requires federated WSI bags/MIL",
        "phaet": "GATED_MODEL_LICENSE_REQUIRED",
        "mascaret": "GATED_MODEL_LICENSE_REQUIRED",
        "parallel": "NOT_RUN: canonicalization of the full Track-D cohort requires a complete M1 checkpoint run",
        "factorstain": "NOT_RUN: canonicalization of the full Track-D cohort requires a complete M1 checkpoint run",
    }
    present = {row["Registry Method"] for row in rows}
    for name, reason in feature_declared.items():
        if name in selected and name not in present:
            rows.append(
                {
                    "Method": get_baseline(name).display_name,
                    "Registry Method": name,
                    "Status": statuses.get(name, {}).get("status", reason),
                }
            )
    return pd.DataFrame(rows)


def _compute_budget(
    config: dict, selected: list[str], statuses: dict[str, dict], out: Path
) -> pd.DataFrame:
    training_count = len(
        pd.read_parquet(out / "metadata" / "training_manifest.parquet")
    )
    rows = []
    for name in selected:
        spec = get_baseline(name)
        budget = config.get("compute_budget", {}).get(name, {})
        status = statuses.get(name, {})
        seed_count = _planned_seed_count(config, name)
        rows.append(
            {
                "Method": spec.display_name,
                "Registry Method": name,
                "Status": status.get("status", "NOT_RUN"),
                "Parameter Count": {
                    "joint": 953347,
                    "parallel": 885891,
                    "factorstain": 890554,
                }.get(name, np.nan),
                "Training Images": training_count
                if spec.method_type in {"LEARNED PRIOR WORK", "OURS / INTERNAL"}
                else 0,
                "Epochs": 75
                if name in {"joint", "parallel", "factorstain"}
                else np.nan,
                "Optimizer": "AdamW"
                if name in {"joint", "parallel", "factorstain", "scangen"}
                else "upstream/default or N/A",
                "Learning Rate": 2e-4
                if name in {"joint", "parallel", "factorstain"}
                else 1e-3
                if name == "scangen"
                else np.nan,
                "Planned Seeds": seed_count,
                "Estimated GPU Hours per Seed/Run": budget.get("gpu_hours", np.nan),
                "Estimated Total GPU Hours": budget.get("gpu_hours", np.nan)
                * seed_count,
                "Observed Wall Hours": status.get("wall_seconds", np.nan) / 3600
                if status
                else np.nan,
                "Peak Memory GB": budget.get("peak_memory_gb", np.nan),
                "Estimated Disk GB": budget.get("disk_gb", np.nan),
                "External Pretraining Data": spec.external_pretraining,
                "Pretrained Weights Used": spec.external_pretraining != "NONE",
            }
        )
    return pd.DataFrame(rows)


def _paired_macro_combination_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict:
    """Cluster bootstrap groups, macro-averaging held-out cells each draw."""

    pivot = frame.pivot_table(
        index=["combination_id", "aligned_group_id"],
        columns="method",
        values=metric,
        aggfunc="mean",
    )
    paired = pivot[[method_a, method_b]].dropna().reset_index()
    if paired.empty:
        raise ValueError(f"No paired macro-combination samples for {metric}")
    paired["difference"] = paired[method_a] - paired[method_b]
    observed = float(paired.groupby("combination_id").difference.mean().mean())
    groups = paired.aligned_group_id.astype(str).unique()
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    for replicate in range(n_bootstrap):
        sampled = rng.choice(groups, len(groups), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        draw = paired.assign(
            weight=paired.aligned_group_id.astype(str).map(multiplicity).fillna(0)
        )
        draw = draw[draw.weight.gt(0)]
        by_combination = [
            np.average(values.difference, weights=values.weight)
            for _, values in draw.groupby("combination_id", sort=False)
        ]
        bootstrap[replicate] = float(np.mean(by_combination))
    alpha = 1 - confidence
    low, high = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric,
        "reported_metric": "macro_combination_composite_utility",
        "method_a": method_a,
        "method_b": method_b,
        "mean_difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_superiority": float((bootstrap > 0).mean()),
        "n_groups": len(groups),
        "n_combinations": int(paired.combination_id.nunique()),
        "direction": "positive means BEST_OURS is better",
        "outcome": "win" if low > 0 else "loss" if high < 0 else "tie",
    }


def _statistics(
    metrics: pd.DataFrame,
    primary: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, str | None, str | None]:
    complete = primary[primary["Composite Score ↑"].notna()]
    ours = complete[complete["Registry Method"].isin(["parallel", "factorstain"])]
    external = complete[
        ~complete["Registry Method"].isin(
            ["joint", "parallel", "factorstain", "noadapt"]
        )
    ]
    if ours.empty or external.empty or metrics.empty:
        return pd.DataFrame(), None, None
    best_ours = str(
        ours.sort_values("Composite Score ↑", ascending=False).iloc[0][
            "Registry Method"
        ]
    )
    best_external = str(
        external.sort_values("Composite Score ↑", ascending=False).iloc[0][
            "Registry Method"
        ]
    )
    strict = metrics[
        metrics.track.eq("C") & metrics.method.isin([best_ours, best_external])
    ].copy()
    strict["combination_id"] = (
        strict.target_stain_id.astype(str) + "x" + strict.target_scanner_id.astype(str)
    )
    utilities = {
        "target_stain_ba": ("stain_target_correct", 1),
        "target_scanner_ba": ("scanner_target_correct", 1),
        "morphology": ("morphology_preservation", 1),
        "od_distance": ("od_distance", -1),
        "factor_isolation": ("factor_isolation", 1),
        "tissue_consistency": ("tissue_consistency", 1),
    }
    rows = []
    replicates = (
        config["fast_dev_bootstrap_samples"]
        if config["fast_dev_run"]
        else config["bootstrap_samples"]
    )
    for label, (column, direction) in utilities.items():
        frame = strict[["method", "aligned_group_id", column]].dropna().copy()
        if frame.empty:
            continue
        frame["utility"] = frame[column] * direction
        try:
            result = paired_bootstrap(
                frame,
                "utility",
                best_ours,
                best_external,
                n_bootstrap=replicates,
                seed=stable_seed(config["seed"], label),
            )
        except ValueError:
            continue
        result.update(
            {
                "reported_metric": label,
                "direction": "positive means BEST_OURS is better",
                "outcome": "win"
                if result["ci_low"] > 0
                else "loss"
                if result["ci_high"] < 0
                else "tie",
            }
        )
        rows.append(result)
    sample_components = pd.DataFrame(
        {
            "target_stain": strict.stain_target_correct,
            "target_scanner": strict.scanner_target_correct,
            "morphology": strict.morphology_preservation,
            "od_fidelity": 1 / (1 + strict.od_distance),
            "factor_isolation": strict.factor_isolation,
            "tissue_consistency": strict.tissue_consistency,
        }
    )
    strict["sample_composite_utility"] = sample_components.mean(axis=1, skipna=True)
    composite_frame = strict[
        ["method", "aligned_group_id", "sample_composite_utility"]
    ].dropna()
    try:
        composite_result = paired_bootstrap(
            composite_frame,
            "sample_composite_utility",
            best_ours,
            best_external,
            n_bootstrap=replicates,
            seed=stable_seed(config["seed"], "sample_composite_utility"),
        )
        composite_result.update(
            {
                "reported_metric": "paired_sample_composite_utility",
                "direction": "positive means BEST_OURS is better",
                "outcome": "win"
                if composite_result["ci_low"] > 0
                else "loss"
                if composite_result["ci_high"] < 0
                else "tie",
            }
        )
        rows.append(composite_result)
    except ValueError:
        pass
    try:
        rows.append(
            _paired_macro_combination_bootstrap(
                strict[
                    [
                        "method",
                        "aligned_group_id",
                        "combination_id",
                        "sample_composite_utility",
                    ]
                ].dropna(),
                "sample_composite_utility",
                best_ours,
                best_external,
                replicates,
                stable_seed(config["seed"], "macro_combination_composite_utility"),
                config["decision"]["confidence"],
            )
        )
    except ValueError:
        pass
    return pd.DataFrame(rows), best_ours, best_external


def _sample_weighted_summary(metrics: pd.DataFrame, mmd: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or not metrics.track.eq("C").any():
        return pd.DataFrame()
    strict = metrics[metrics.track.eq("C")]
    summary = (
        strict.groupby(["method", "display_name"], sort=True)
        .agg(
            target_stain_ba=("stain_target_correct", "mean"),
            target_scanner_ba=("scanner_target_correct", "mean"),
            morphology_preservation=("morphology_preservation", "mean"),
            od_distance=("od_distance", "mean"),
            factor_isolation=("factor_isolation", "mean"),
            tissue_consistency=("tissue_consistency", "mean"),
            sample_count=("episode_id", "count"),
        )
        .reset_index()
    )
    if not mmd.empty:
        weighted_mmd = (
            mmd.assign(weighted=lambda frame: frame.dino_mmd2 * frame.n)
            .groupby("method", sort=True)
            .agg(weighted=("weighted", "sum"), n=("n", "sum"))
        )
        weighted_mmd["sample_weighted_dino_mmd2"] = (
            weighted_mmd.weighted / weighted_mmd.n.clip(lower=1)
        )
        summary = summary.merge(
            weighted_mmd[["sample_weighted_dino_mmd2"]],
            left_on="method",
            right_index=True,
            how="left",
        )
    return summary


def _placeholder(
    path: Path, title: str, message: str = "No complete results available"
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.axis("off")
    axis.text(0.5, 0.58, title, ha="center", va="center", fontsize=18, weight="bold")
    axis.text(0.5, 0.4, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _bar_figure(
    frame: pd.DataFrame, label: str, value: str, path: Path, title: str
) -> None:
    usable = frame.dropna(subset=[value]) if value in frame else pd.DataFrame()
    if usable.empty:
        _placeholder(path, title)
        return
    usable = usable.sort_values(value)
    fig, axis = plt.subplots(figsize=(10, max(4, 0.48 * len(usable))))
    axis.barh(usable[label].astype(str), usable[value], color="#2f6f8f")
    axis.set_xlabel(value)
    axis.set_title(title, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _qualitative_grid(generation: pd.DataFrame, path: Path, title: str) -> None:
    if generation.empty:
        _placeholder(path, title)
        return
    strict = generation[generation.track.eq("C")]
    if strict.empty:
        _placeholder(path, title)
        return
    episode = (
        strict.episode_id.astype(str)
        .sort_values()
        .iloc[len(strict.episode_id.unique()) // 2]
    )
    frame = (
        strict[strict.episode_id.astype(str).eq(str(episode))]
        .sort_values(["display_name", "seed"])
        .drop_duplicates("display_name")
    )
    first = frame.iloc[0]
    columns = [("Source", first.source_path), ("Real target", first.target_path)]
    columns.extend((row.display_name, row.generated_path) for row in frame.itertuples())
    columns = columns[:14]
    fig, axes = plt.subplots(
        1, len(columns), figsize=(3 * len(columns), 3.5), squeeze=False
    )
    for axis, (name, image_path) in zip(axes[0], columns, strict=True):
        axis.imshow(Image.open(image_path).convert("RGB"))
        axis.set_title(name, fontsize=8)
        axis.axis("off")
    fig.suptitle(title, weight="bold")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _best_median_worst(
    metrics: pd.DataFrame, generation: pd.DataFrame, path: Path
) -> None:
    strict = (
        metrics[metrics.track.eq("C")].copy() if not metrics.empty else pd.DataFrame()
    )
    if strict.empty or generation.empty:
        _placeholder(path, "Best / median / worst strict-composition examples")
        return
    candidates = [
        method
        for method in ("factorstain", "parallel", "macenko")
        if method in set(strict.method)
    ]
    reference_method = candidates[0] if candidates else min(strict.method.unique())
    reference = strict[strict.method.eq(reference_method)].copy()
    reference["selection_score"] = 0.5 * reference.morphology_preservation + 0.5 / (
        1 + reference.od_distance
    )
    episode_scores = (
        reference.groupby("episode_id").selection_score.mean().sort_values()
    )
    selected = [
        ("Worst", str(episode_scores.index[0])),
        ("Median", str(episode_scores.index[len(episode_scores) // 2])),
        ("Best", str(episode_scores.index[-1])),
    ]
    methods = sorted(generation[generation.track.eq("C")].display_name.unique())
    columns = ["Source", "Real target", *methods]
    fig, axes = plt.subplots(
        3, len(columns), figsize=(2.7 * len(columns), 8.5), squeeze=False
    )
    for row_number, (label, episode_id) in enumerate(selected):
        frame = (
            generation[
                generation.track.eq("C")
                & generation.episode_id.astype(str).eq(episode_id)
            ]
            .sort_values(["display_name", "seed"])
            .drop_duplicates("display_name")
        )
        if frame.empty:
            continue
        first = frame.iloc[0]
        image_paths = {
            "Source": first.source_path,
            "Real target": first.target_path,
            **dict(zip(frame.display_name, frame.generated_path, strict=True)),
        }
        for column_number, name in enumerate(columns):
            axis = axes[row_number, column_number]
            axis.axis("off")
            if name in image_paths and Path(image_paths[name]).exists():
                axis.imshow(Image.open(image_paths[name]).convert("RGB"))
            else:
                axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            if row_number == 0:
                axis.set_title(name, fontsize=8)
            if column_number == 0:
                axis.set_ylabel(label, fontsize=11, weight="bold")
    fig.suptitle(
        "Predefined selection: 0.5 morphology + 0.5 OD fidelity; no cherry-picking",
        weight="bold",
    )
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _figures(
    out: Path,
    primary: pd.DataFrame,
    per_combination: pd.DataFrame,
    stain: pd.DataFrame,
    scanner: pd.DataFrame,
    feature: pd.DataFrame,
    generation: pd.DataFrame,
    metrics: pd.DataFrame,
    best_ours: str | None,
    best_external: str | None,
    significant: bool,
) -> None:
    figures = out / "figures"
    _bar_figure(
        primary,
        "Method",
        "Composite Score ↑",
        figures / "primary_composition_ranking.png",
        "Strict-composition macro ranking",
    )
    _bar_figure(
        primary,
        "Method",
        "DINO-MMD ↓",
        figures / "real_vs_generated_mmd.png",
        "Real vs generated DINOv3 MMD²",
    )
    _bar_figure(
        stain,
        "Method",
        "Target Stain BA ↑",
        figures / "stain_track.png",
        "Track A — stain translation",
    )
    _bar_figure(
        scanner,
        "Method",
        "Scanner Target BA ↑",
        figures / "scanner_track.png",
        "Track B — controlled scanner transfer",
    )
    _bar_figure(
        feature,
        "Method",
        "PathoROB-inspired Index ↑",
        figures / "feature_robustness.png",
        "Track D — feature robustness",
    )
    _qualitative_grid(
        generation,
        figures / "generated_vs_real_grid.png",
        "Fixed held-out episode (no cherry-picking)",
    )

    if primary.dropna(subset=["Morphology ↑", "DINO-MMD ↓"]).empty:
        _placeholder(
            figures / "morphology_vs_domain_tradeoff.png",
            "Morphology vs domain fidelity",
        )
    else:
        frame = primary.dropna(subset=["Morphology ↑", "DINO-MMD ↓"])
        fig, axis = plt.subplots(figsize=(8, 6))
        fidelity = 1 / (1 + frame["DINO-MMD ↓"])
        axis.scatter(frame["Morphology ↑"], fidelity, s=70, color="#2f6f8f")
        for (_, row), y in zip(frame.iterrows(), fidelity, strict=True):
            axis.annotate(str(row["Method"]), (float(row["Morphology ↑"]), float(y)))
        axis.set_xlabel("Morphology preservation ↑")
        axis.set_ylabel("Target-domain fidelity 1/(1+MMD²) ↑")
        axis.set_title(
            "Morphology preservation vs target-domain fidelity", weight="bold"
        )
        fig.savefig(
            figures / "morphology_vs_domain_tradeoff.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)

    if per_combination.empty:
        _placeholder(figures / "per_combination_wins.png", "Per-combination wins")
    else:
        scored = per_combination.copy()
        scored["predefined_composite"] = scored.apply(
            lambda row: np.nanmean(
                [
                    1 / (1 + row.get("dino_mmd2", np.nan)),
                    row.get("target_stain_ba", np.nan),
                    row.get("target_scanner_ba", np.nan),
                    row.get("morphology_preservation", np.nan),
                    1 / (1 + row.get("od_distance", np.nan)),
                    row.get("factor_isolation", np.nan),
                    row.get("tissue_consistency", np.nan),
                ]
            ),
            axis=1,
        )
        pivot = scored.pivot_table(
            index="method", columns="combination_id", values="predefined_composite"
        )
        wins = pivot.eq(pivot.max(axis=0), axis=1).astype(int)
        fig, axis = plt.subplots(
            figsize=(max(8, 1.1 * wins.shape[1]), max(4, 0.5 * wins.shape[0]))
        )
        image = axis.imshow(wins, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        axis.set_xticks(range(wins.shape[1]), wins.columns, rotation=40, ha="right")
        axis.set_yticks(range(wins.shape[0]), wins.index)
        axis.set_title("Per-combination morphology win map")
        fig.colorbar(image, ax=axis, label="win")
        fig.savefig(figures / "per_combination_wins.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    _best_median_worst(metrics, generation, figures / "best_median_worst.png")
    panels = [
        (
            Path(
                out.parent
                / "m1_factorial"
                / "figures"
                / "heldout_combination_matrix.png"
            ),
            "A. Held-out split",
        ),
        (figures / "generated_vs_real_grid.png", "B. Fixed qualitative example"),
        (figures / "primary_composition_ranking.png", "C. Strict ranking"),
        (figures / "real_vs_generated_mmd.png", "D. DINO MMD"),
        (figures / "morphology_vs_domain_tradeoff.png", "E. Biology/domain trade-off"),
        (figures / "per_combination_wins.png", "F. Combination wins"),
        (figures / "scanner_track.png", "G. Scanner transfer"),
        (figures / "feature_robustness.png", "H. Feature robustness"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    for axis, (panel, title) in zip(axes.flat, panels, strict=True):
        axis.axis("off")
        axis.set_title(title, weight="bold")
        if panel.exists():
            axis.imshow(plt.imread(panel))
        else:
            axis.text(0.5, 0.5, "Unavailable", ha="center")
    footer = (
        f"BEST EXTERNAL METHOD: {best_external or 'pending'}    |    "
        f"BEST OURS METHOD: {best_ours or 'pending'}    |    "
        f"STATISTICALLY SIGNIFICANT: {'YES' if significant else 'NO / PENDING'}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=13, weight="bold")
    fig.suptitle("FactorStain External SOTA Benchmark", fontsize=22, weight="bold")
    fig.savefig(figures / "SOTA_SUMMARY_DASHBOARD.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _report(
    out: Path,
    selected: list[str],
    primary: pd.DataFrame,
    capability: pd.DataFrame,
    decision: dict,
    statuses: dict[str, dict],
    dino_note: str,
    feature: pd.DataFrame,
    per_combination: pd.DataFrame,
) -> None:
    capability_view = capability[
        [
            "display_name",
            "native_task",
            "supports_stain",
            "supports_scanner",
            "supports_unseen_composition",
            "requires_target_domain_samples",
            "primary_track",
            "availability",
        ]
    ]
    headers = capability_view.columns.tolist()
    capability_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in (
        capability_view.fillna("—").astype(str).itertuples(index=False, name=None)
    ):
        capability_lines.append(
            "| " + " | ".join(value.replace("|", "/") for value in row) + " |"
        )
    capability_md = "\n".join(capability_lines)
    status_lines = "\n".join(
        f"- **{name}**: `{statuses.get(name, {}).get('status', 'NOT_RUN')}` — {statuses.get(name, {}).get('reason', '')}"
        for name in selected
    )
    questions = [
        "Does a prior stain method solve strict composition?",
        "How strong is Macenko + ScannerLUT?",
        "How do classical and neural methods compare?",
        "Does HistAuGAN match explicit factorization?",
        "Does SAStainDiff offset its lack of scanner modeling?",
        "Does explicit scanner modeling help?",
        "Which is stronger: OURS-Parallel or OURS-Ordered?",
        "Are gains concentrated in particular combinations?",
        "Is domain fidelity bought at a morphology cost?",
        "Do FEATMAP/ScanGen remove the need for image correction?",
        "Is improvement over the strongest fair external baseline significant?",
    ]
    if not decision["decision_valid"] or primary.empty:
        answers = [
            "Pending a complete non-FAST run with all required fair baselines; no scientific conclusion is issued."
        ] * len(questions)
    else:
        indexed = primary.set_index("Registry Method")

        def score(name: str) -> float:
            return (
                float(indexed.loc[name, "Composite Score ↑"])
                if name in indexed.index
                else np.nan
            )

        best_ours, best_external = decision["best_ours"], decision["best_external"]
        classical = primary[primary.Type.eq("CLASSICAL")]["Composite Score ↑"].dropna()
        learned = primary[primary.Type.eq("LEARNED PRIOR WORK")][
            "Composite Score ↑"
        ].dropna()
        parallel, ordered = score("parallel"), score("factorstain")
        morphology_delta = (
            float(
                indexed.loc[best_ours, "Morphology ↑"]
                - indexed.loc[best_external, "Morphology ↑"]
            )
            if best_ours in indexed.index and best_external in indexed.index
            else np.nan
        )
        robust = (
            feature.dropna(subset=["PathoROB-inspired Index ↑"])
            if "PathoROB-inspired Index ↑" in feature
            else pd.DataFrame()
        )
        best_feature = (
            robust.sort_values("PathoROB-inspired Index ↑", ascending=False).iloc[0][
                "Method"
            ]
            if not robust.empty
            else "unavailable"
        )
        answers = [
            f"The strongest fair prior is `{best_external}` (composite {score(best_external):.4f}); compare its raw metrics with `{best_ours}` rather than treating stain fidelity alone as success.",
            f"Macenko + ScannerLUT has composite {score('macenko'):.4f}; its MMD, morphology, and OD values are shown in the primary table.",
            f"Mean composite is {classical.mean():.4f} for available classical methods and {learned.mean():.4f} for available prior neural methods.",
            f"HistAuGAN scores {score('histaugan'):.4f} versus best ours {score(best_ours):.4f} under the same macro protocol.",
            f"SAStainDiff scores {score('sastaindiff'):.4f}; its row is marked separately if external-data weights were used.",
            "The scanner track and strict +ScannerLUT rows quantify this directly; explicit modeling is beneficial only if scanner fidelity improves without material morphology loss.",
            f"OURS-Parallel={parallel:.4f}; OURS-Ordered={ordered:.4f}. The larger measured value is reported without privileging the ordered model.",
            f"Best ours wins {decision['winning_heldout_combinations']} held-out combinations; `per_combination.csv` shows whether gains are concentrated.",
            f"Best-ours minus best-external morphology is {morphology_delta:+.4f}; material degradation flag is `{decision['morphology_materially_degraded']}`.",
            f"The strongest measured feature row is `{best_feature}`; Track D reports acquisition reduction and biology preservation jointly, not image metrics.",
            f"Paired aligned-group bootstrap significant: `{decision['statistically_significant']}`; see `statistical_comparisons.csv` for differences and CIs.",
        ]
    question_text = "\n".join(
        f"{number}. **{question}** {answer}"
        for number, (question, answer) in enumerate(
            zip(questions, answers, strict=True), 1
        )
    )
    text = f"""# M1 external SOTA benchmark

## Decision

**{decision["status"]}** (decision valid: `{str(decision["decision_valid"]).lower()}`).

Best external strict method: `{decision.get("best_external") or "pending"}`. Best ours:
`{decision.get("best_ours") or "pending"}`. The decision compares best ours against best
fair external strict baseline—not against Joint.

{dino_note or "Independent DINOv3 stain/scanner/tissue probes were fitted on real M1 training images only."}

## Immutable data and leakage policy

The benchmark consumes `metadata/evaluation_manifest.parquet` once. Track C is the exact
M1 unseen-combination episode set; Track B is the exact controlled scanner subset; Track A
uses unseen-morphology, source-scanner stain targets. `metadata/reference_policy.json`
contains every reference ID and both split SHA256 hashes. Held-out joint targets are final
evaluation only and are excluded from references, LUT fitting, training, validation, early
stopping, and model selection. Oracle-target methods are segregated.

## Method execution status

{status_lines}

## Capability matrix

{capability_md}

## Paper-level questions

{question_text}

## Composite-score definition

Each available raw metric is mapped to [0,1] before a weighted average: accuracies and
preservation scores are used directly; MMD and OD distance use `1/(1+x)`. Equal weights
are primary. Domain-heavy and biology-heavy sensitivity columns are reported alongside.
Raw metrics—not the composite—control the scientific interpretation.
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_sota_benchmark.yaml")
    parser.add_argument("--method", default=os.getenv("METHOD", ""))
    parser.add_argument("--methods", default=os.getenv("METHODS", ""))
    parser.add_argument("--tier", default=os.getenv("TIER", "1"))
    args = parser.parse_args()
    config = load_config(args.config)
    out = _output(config)
    selected = resolve_methods(args.method, args.methods, args.tier)
    run_seeds = config.get("run_seeds", config["seeds"])
    statuses = _status_records(out, run_seeds)
    generation = _generation_records(out, selected, run_seeds)
    metrics, generated_features, dino_note = _evaluate_images(generation, config, out)
    mmd = _mmd_rows(metrics, generated_features, config, out)
    primary, per_combination = _macro_primary(metrics, mmd, selected, statuses, config)
    stain = _track_table(metrics, "A")
    scanner = _track_table(metrics, "B")
    feature = _feature_table(config, out, selected, statuses)
    statistics, best_ours, best_external = _statistics(metrics, primary, config)

    tables = out / "tables"
    primary.to_csv(tables / "PRIMARY_STRICT_COMPOSITION.csv", index=False)
    stain.to_csv(tables / "STAIN_TRACK.csv", index=False)
    scanner.to_csv(tables / "SCANNER_TRACK.csv", index=False)
    feature.to_csv(tables / "FEATURE_ROBUSTNESS_TRACK.csv", index=False)
    pd.DataFrame(columns=["Method", "Status", "Oracle target used?", "Reason"]).to_csv(
        tables / "ORACLE_DIAGNOSTIC.csv", index=False
    )
    per_combination.to_csv(tables / "per_combination.csv", index=False)
    _sample_weighted_summary(metrics, mmd).to_csv(
        tables / "sample_weighted_summary.csv", index=False
    )
    statistics.to_csv(tables / "statistical_comparisons.csv", index=False)
    _compute_budget(config, selected, statuses, out).to_csv(
        tables / "compute_budget.csv", index=False
    )
    significant = bool(
        not statistics.empty
        and (
            statistics.reported_metric.eq("macro_combination_composite_utility")
            & statistics.outcome.eq("win")
        ).any()
    )

    def method_complete(name: str) -> bool:
        payload = statuses.get(name, {})
        if payload.get("status") != "COMPLETE":
            return False
        if name not in SEEDED_IMAGE_METHODS:
            return True
        observed = payload.get("seed_statuses")
        if observed is None:
            observed = {str(payload.get("seed")): payload.get("status")}
        return all(
            observed.get(str(seed)) == "COMPLETE"
            for seed in config.get("run_seeds", config["seeds"])
        )

    required_tier_one = {
        name
        for name in TIER_METHODS[1]
        if get_baseline(name).availability
        not in {"NOT_APPLICABLE_TO_STRICT_IMAGE_COMPOSITION"}
    }
    all_required_selected = required_tier_one.issubset(selected)
    completed_required = all_required_selected and all(
        method_complete(name) for name in required_tier_one
    )
    full_seed_protocol = sorted(config.get("run_seeds", config["seeds"])) == sorted(
        config["seeds"]
    )
    decision_valid = bool(
        not config["fast_dev_run"]
        and full_seed_protocol
        and completed_required
        and best_ours
        and best_external
        and not dino_note
    )
    status = "INCOMPLETE"
    reasons = []
    if config["fast_dev_run"]:
        reasons.append(
            "FAST_DEV_RUN validates plumbing only; no scientific decision is permitted."
        )
    if not full_seed_protocol:
        reasons.append(
            "SOTA_SEEDS selected a subset of the preregistered three-seed protocol; "
            "results are diagnostic and decision_valid is false."
        )
    if not completed_required:
        reasons.append(
            "The complete predeclared Tier-1 roster was not selected and completed."
        )
    if not best_ours or not best_external:
        reasons.append(
            "Best ours versus best external strict comparison is unavailable."
        )
    wins = 0
    meaningful = 0
    winning_combinations = 0
    morphology_degraded = False
    relative_improvements: dict[str, float] = {}
    if best_ours and best_external and not primary.empty:
        ours_row = primary[primary["Registry Method"].eq(best_ours)].iloc[0]
        external_row = primary[primary["Registry Method"].eq(best_external)].iloc[0]
        principal = {
            "DINO-MMD": ("DINO-MMD ↓", False),
            "Target Stain BA": ("Target Stain BA ↑", True),
            "Target Scanner BA": ("Target Scanner BA ↑", True),
            "Morphology": ("Morphology ↑", True),
            "OD Distance": ("OD Distance ↓", False),
            "Factor Isolation": ("Factor Isolation ↑", True),
        }
        for label, (column, higher) in principal.items():
            ours_value, external_value = ours_row[column], external_row[column]
            if not np.isfinite(ours_value) or not np.isfinite(external_value):
                continue
            advantage = (
                ours_value - external_value if higher else external_value - ours_value
            )
            if advantage > 0:
                wins += 1
            relative = advantage / max(abs(float(external_value)), 1e-8)
            relative_improvements[label] = float(relative)
            meaningful += int(
                relative >= config["decision"]["meaningful_relative_improvement"]
            )
        morphology_degraded = bool(
            ours_row["Morphology ↑"]
            < external_row["Morphology ↑"]
            - config["decision"]["morphology_degradation_tolerance"]
        )
        combo = per_combination.groupby(["method", "combination_id"], sort=True).mean(
            numeric_only=True
        )
        for combination in sorted(per_combination.combination_id.unique()):
            ours_key, external_key = (
                (best_ours, combination),
                (
                    best_external,
                    combination,
                ),
            )
            if ours_key not in combo.index or external_key not in combo.index:
                continue
            ours_values, external_values = combo.loc[ours_key], combo.loc[external_key]
            combo_weights = config["composite"]["weights"]
            ours_components = {
                "dino_mmd2": 1 / (1 + ours_values.get("dino_mmd2", np.nan)),
                "target_stain_ba": ours_values.get("target_stain_ba", np.nan),
                "target_scanner_ba": ours_values.get("target_scanner_ba", np.nan),
                "morphology_preservation": ours_values.get(
                    "morphology_preservation", np.nan
                ),
                "od_distance": 1 / (1 + ours_values.get("od_distance", np.nan)),
                "factor_isolation": ours_values.get("factor_isolation", np.nan),
                "tissue_consistency": ours_values.get("tissue_consistency", np.nan),
            }
            external_components = {
                "dino_mmd2": 1 / (1 + external_values.get("dino_mmd2", np.nan)),
                "target_stain_ba": external_values.get("target_stain_ba", np.nan),
                "target_scanner_ba": external_values.get("target_scanner_ba", np.nan),
                "morphology_preservation": external_values.get(
                    "morphology_preservation", np.nan
                ),
                "od_distance": 1 / (1 + external_values.get("od_distance", np.nan)),
                "factor_isolation": external_values.get("factor_isolation", np.nan),
                "tissue_consistency": external_values.get("tissue_consistency", np.nan),
            }
            winning_combinations += int(
                _composite(ours_components, combo_weights)
                > _composite(external_components, combo_weights)
            )
    if decision_valid:
        if (
            wins >= config["decision"]["win_metrics"]
            and meaningful >= config["decision"]["meaningful_metrics"]
            and significant
            and not morphology_degraded
            and winning_combinations >= 2
        ):
            status = "STRONG_GO"
        elif significant and wins >= 3 and not morphology_degraded:
            status = "GO"
        else:
            ours_score = float(
                primary.loc[
                    primary["Registry Method"].eq(best_ours), "Composite Score ↑"
                ].iloc[0]
            )
            external_score = float(
                primary.loc[
                    primary["Registry Method"].eq(best_external), "Composite Score ↑"
                ].iloc[0]
            )
            status = (
                "GO_WITH_SCOPE_REDUCTION" if ours_score > external_score else "NO_GO"
            )
    decision = {
        "status": status,
        "decision_valid": decision_valid,
        "best_ours": best_ours,
        "best_external": best_external,
        "comparison_basis": "macro held-out-combination performance; best ours vs best fair external strict baseline",
        "statistically_significant": significant if decision_valid else False,
        "principal_metric_wins": wins,
        "meaningful_relative_improvements": meaningful,
        "relative_improvements": relative_improvements,
        "morphology_materially_degraded": morphology_degraded,
        "winning_heldout_combinations": winning_combinations,
        "reasons": reasons,
        "selected_methods": selected,
        "declared_scientific_seeds": config["seeds"],
        "executed_seeds": config.get("run_seeds", config["seeds"]),
        "fast_dev_run": config["fast_dev_run"],
    }
    atomic_json_dump(decision, out / "FINAL_DECISION.json")
    capability = pd.read_csv(out / "baseline_capability_matrix.csv")
    _figures(
        out,
        primary,
        per_combination,
        stain,
        scanner,
        feature,
        generation,
        metrics,
        best_ours,
        best_external,
        significant and decision_valid,
    )
    _report(
        out,
        selected,
        primary,
        capability,
        decision,
        statuses,
        dino_note,
        feature,
        per_combination,
    )
    print(
        f"Benchmark aggregation complete: {len(primary)} strict rows; "
        f"decision={status}, valid={decision_valid}"
    )


if __name__ == "__main__":
    main()
