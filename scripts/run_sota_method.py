#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from factorstain.baselines.base import FitContext, MethodUnavailable
from factorstain.baselines.classical import (
    ClassicalStainMethod,
    NoAdaptMethod,
    ScannerOnlyMethod,
)
from factorstain.baselines.external import OfficialSubprocessMethod
from factorstain.baselines.features import FEATMAPHarmonizer, ScanGenHarmonizer
from factorstain.baselines.internal import RendererAcquisitionMethod
from factorstain.baselines.registry import get_baseline
from factorstain.models.dinov3 import load_feature_cache, save_feature_cache
from factorstain.training.renderer import train_renderer
from factorstain.utils.config import load_config
from factorstain.utils.runtime import atomic_json_dump

IMAGE_TRACKS = {
    "noadapt": {"A", "B", "C"},
    "histogram": {"A", "C"},
    "reinhard": {"A", "C"},
    "macenko": {"A", "C"},
    "vahadane": {"A", "C"},
    "rgb_affine": {"B"},
    "polynomial": {"B"},
    "scanner_lut": {"B"},
    "stainnet": {"A", "C"},
    "staingan": {"A", "C"},
    "cyclegan": {"A", "C"},
    "pix2pix": {"B"},
    "histaugan": {"A", "C"},
    "cagan": {"A", "C"},
    "sastaindiff": {"A", "C"},
    "joint": {"A", "B", "C"},
    "parallel": {"A", "B", "C"},
    "factorstain": {"A", "B", "C"},
}
FEATURE_METHODS = {"scangen", "featmap"}
EXTERNAL_STAIN = {
    "stainnet",
    "staingan",
    "cyclegan",
    "histaugan",
    "cagan",
    "sastaindiff",
}


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
    return Path(config["paths"]["outputs_root"]) / config["milestone"]


def _fit_context(config: dict, out: Path) -> FitContext:
    policy = json.loads(
        (out / "metadata" / "reference_policy.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (out / "metadata" / "leakage_audit.json").read_text(encoding="utf-8")
    )
    required = (
        "evaluation_targets_disjoint_from_training",
        "unseen_morphology_groups_disjoint_from_training",
        "reference_and_scanner_fit_disjoint_from_heldout_targets",
    )
    if not all(audit.get(key, False) for key in required):
        raise RuntimeError(f"Leakage audit did not pass all strict checks: {audit}")
    return FitContext(
        train_index=pd.read_parquet(out / "metadata" / "training_manifest.parquet"),
        reference_policy=policy,
        scanner_pairs=pd.read_parquet(out / "metadata" / "scanner_fit_pairs.parquet"),
        output_dir=out / "checkpoints" / f"seed{config['seed']}",
        image_size=config["image_size"],
        seed=config["seed"],
        fast_dev_run=config["fast_dev_run"],
        metadata={"leakage_audit": audit, "benchmark_output": str(out)},
    )


def _create_method(name: str, config: dict, out: Path):
    if name == "noadapt":
        return NoAdaptMethod()
    if name in {"histogram", "reinhard", "macenko", "vahadane"}:
        return ClassicalStainMethod(name, "lut", config["lut_grid_size"])
    scanner_kinds = {
        "rgb_affine": "affine",
        "polynomial": "polynomial",
        "scanner_lut": "lut",
    }
    if name in scanner_kinds:
        return ScannerOnlyMethod(name, scanner_kinds[name], config["lut_grid_size"])
    if name in {"joint", "parallel", "factorstain"}:
        original = (
            Path(config["paths"]["outputs_root"])
            / config["m1_milestone"]
            / "checkpoints"
            / name
            / "best.pt"
        )
        local = (
            out
            / "checkpoints"
            / f"seed{config['seed']}"
            / "checkpoints"
            / name
            / "best.pt"
        )
        checkpoint = original if config["seed"] == 42 and original.exists() else local
        return RendererAcquisitionMethod(
            name, checkpoint, width=config.get("model_width", 64)
        )
    if name in EXTERNAL_STAIN:
        return OfficialSubprocessMethod(
            name, append_scanner_lut=True, grid_size=config["lut_grid_size"]
        )
    if name == "pix2pix":
        return OfficialSubprocessMethod(name, append_scanner_lut=False)
    raise MethodUnavailable(f"{name} has no image-level adapter")


def _ensure_internal_checkpoint(name: str, config: dict, out: Path) -> None:
    original = (
        Path(config["paths"]["outputs_root"])
        / config["m1_milestone"]
        / "checkpoints"
        / name
        / "best.pt"
    )
    if config["seed"] == 42 and original.exists():
        return
    seed_out = out / "checkpoints" / f"seed{config['seed']}"
    checkpoint = seed_out / "checkpoints" / name / "best.pt"
    if checkpoint.exists():
        return
    for child in ("logs", "figures", "checkpoints", "metadata", "tables"):
        (seed_out / child).mkdir(parents=True, exist_ok=True)
    training_config = load_config(config["m1_training_config"])
    training_config["seed"] = config["seed"]
    training_config["fast_dev_run"] = config["fast_dev_run"]
    training_config["paths"] = config["paths"]
    index_path = _resolve(config, os.getenv("M1_INDEX", config["m1_index"]))
    split_path = _resolve(config, config["combination_split"])
    index = pd.read_parquet(index_path).reset_index(drop=True)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_renderer(training_config, name, index, split, seed_out)
    if not checkpoint.exists():
        raise MethodUnavailable(
            f"Internal seed {config['seed']} training did not produce {checkpoint}"
        )


def _display_for_track(name: str, track: str) -> str:
    spec = get_baseline(name)
    if track == "C" and name in {
        "histogram",
        "reinhard",
        "macenko",
        "vahadane",
        *EXTERNAL_STAIN,
    }:
        return f"{spec.display_name} + ScannerLUT (compositional adaptation)"
    return spec.display_name


def _canonicalize_feature_cohort(
    name: str,
    method: RendererAcquisitionMethod,
    context: FitContext,
    config: dict,
    out: Path,
) -> tuple[int, str]:
    """Render every real feature-cache row to one acquisition-train cell.

    The canonical target is selected before evaluation from the most populated
    strict-training cell (lexical tie-break). Held-out targets are inputs only at
    final evaluation time and never influence the selected target or checkpoint.
    """

    counts = (
        context.train_index.groupby(["stain_id", "scanner_id"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["count", "stain_id", "scanner_id"],
            ascending=[False, True, True],
            kind="stable",
        )
    )
    if counts.empty:
        raise MethodUnavailable("No strict-training cell exists for canonicalization")
    target = counts.iloc[0]
    target_stain, target_scanner = str(target.stain_id), str(target.scanner_id)
    index = pd.read_parquet(_resolve(config, os.getenv("M1_INDEX", config["m1_index"])))
    if "image_exists" in index:
        index = index[index.image_exists].copy()
    id_column = "sample_id" if "sample_id" in index else "image_id"
    rows: list[dict] = []
    for record in index.itertuples(index=False):
        sample_id = str(getattr(record, id_column))
        safe_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
        destination = (
            out / "generated" / name / f"seed{config['seed']}" / "D" / f"{safe_id}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with Image.open(record.image_path) as opened:
                source = opened.convert("RGB").resize(
                    (config["image_size"], config["image_size"]),
                    Image.Resampling.BILINEAR,
                )
            generated = method.translate(
                source,
                str(record.stain_id),
                str(record.scanner_id),
                target_stain,
                target_scanner,
                track="C",
            )
            Image.fromarray(generated).save(destination)
        rows.append(
            {
                id_column: sample_id,
                "generated_path": str(destination),
                "source_stain_id": str(record.stain_id),
                "source_scanner_id": str(record.scanner_id),
                "target_stain_id": target_stain,
                "target_scanner_id": target_scanner,
                "seed": config["seed"],
            }
        )
    manifest = out / "generated" / name / f"canonical_manifest_seed{config['seed']}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return len(rows), f"{target_stain}x{target_scanner}"


def _run_images(name: str, config: dict, out: Path) -> dict:
    if name in {"joint", "parallel", "factorstain"}:
        _ensure_internal_checkpoint(name, config, out)
    context = _fit_context(config, out)
    method = _create_method(name, config, out)
    method.fit(context)
    manifest = pd.read_parquet(out / "metadata" / "evaluation_manifest.parquet")
    tracks = IMAGE_TRACKS[name]
    manifest = manifest[manifest.track.isin(tracks)].copy()
    rows, errors = [], []
    for episode in manifest.itertuples(index=False):
        destination = (
            out
            / "generated"
            / name
            / f"seed{config['seed']}"
            / str(episode.track)
            / f"{episode.episode_id}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            source = None
            if not destination.exists():
                with Image.open(episode.source_image_path) as opened:
                    source = opened.convert("RGB").resize(
                        (config["image_size"], config["image_size"]),
                        Image.Resampling.BILINEAR,
                    )
                    generated = method.translate(
                        source,
                        str(episode.source_stain_id),
                        str(episode.source_scanner_id),
                        str(episode.target_stain_id),
                        str(episode.target_scanner_id),
                        track=str(episode.track),
                    )
                Image.fromarray(generated).save(destination)
            stain_cf_path = ""
            scanner_cf_path = ""
            if str(episode.track) == "C":
                stain_cf = destination.with_name(destination.stem + "_stain_cf.png")
                scanner_cf = destination.with_name(destination.stem + "_scanner_cf.png")
                if not stain_cf.exists() or not scanner_cf.exists():
                    if source is None:
                        with Image.open(episode.source_image_path) as opened:
                            source = opened.convert("RGB").resize(
                                (config["image_size"], config["image_size"]),
                                Image.Resampling.BILINEAR,
                            )
                    stain_only = method.translate(
                        source,
                        str(episode.source_stain_id),
                        str(episode.source_scanner_id),
                        str(episode.target_stain_id),
                        str(episode.source_scanner_id),
                        track="A",
                    )
                    if isinstance(method, OfficialSubprocessMethod):
                        scanner_only = method.scanner_bank.apply(
                            np.asarray(source),
                            str(episode.source_scanner_id),
                            str(episode.target_scanner_id),
                        )
                    else:
                        scanner_only = method.translate(
                            source,
                            str(episode.source_stain_id),
                            str(episode.source_scanner_id),
                            str(episode.source_stain_id),
                            str(episode.target_scanner_id),
                            track="B",
                        )
                    Image.fromarray(stain_only).save(stain_cf)
                    Image.fromarray(scanner_only).save(scanner_cf)
                stain_cf_path, scanner_cf_path = str(stain_cf), str(scanner_cf)
            rows.append(
                {
                    "method": name,
                    "seed": config["seed"],
                    "display_name": _display_for_track(name, str(episode.track)),
                    "episode_id": episode.episode_id,
                    "track": episode.track,
                    "generated_path": str(destination),
                    "source_path": episode.source_image_path,
                    "target_path": episode.target_image_path,
                    "stain_cf_path": stain_cf_path,
                    "scanner_cf_path": scanner_cf_path,
                    "source_sample_id": episode.source_sample_id,
                    "target_sample_id": episode.target_sample_id,
                    "aligned_group_id": episode.target_aligned_group_id,
                    "tissue_type": episode.target_tissue_type,
                    "source_stain_id": episode.source_stain_id,
                    "source_scanner_id": episode.source_scanner_id,
                    "target_stain_id": episode.target_stain_id,
                    "target_scanner_id": episode.target_scanner_id,
                    "heldout_target_stain_id": episode.heldout_target_stain_id,
                    "heldout_target_scanner_id": episode.heldout_target_scanner_id,
                    "status": "COMPLETE",
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve every episode failure in audit
            errors.append(
                {
                    "episode_id": episode.episode_id,
                    "track": episode.track,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    result_path = (
        out / "generated" / name / f"generation_manifest_seed{config['seed']}.csv"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_path, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(
            out / "logs" / f"{name}_seed{config['seed']}_episode_errors.csv",
            index=False,
        )
    canonical_rows, canonical_target = 0, None
    if name in {"parallel", "factorstain"}:
        canonical_rows, canonical_target = _canonicalize_feature_cohort(
            name, method, context, config, out
        )
    return {
        "status": "COMPLETE"
        if rows and not errors
        else "PARTIAL"
        if rows
        else "FAILED",
        "generated_episodes": len(rows),
        "failed_episodes": len(errors),
        "tracks": sorted(
            {row["track"] for row in rows} | ({"D"} if canonical_rows else set())
        ),
        "result_manifest": str(result_path),
        "track_d_canonical_rows": canonical_rows,
        "track_d_canonical_target_train_cell": canonical_target,
    }


def _feature_frame(
    config: dict, out: Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cache_path = _resolve(config, os.getenv("DINOV3_CACHE", config["dinov3_cache"]))
    if not cache_path.exists():
        raise MethodUnavailable(f"Frozen DINOv3 feature cache is missing: {cache_path}")
    features, sample_ids, _ = load_feature_cache(cache_path)
    index_path = _resolve(config, os.getenv("M1_INDEX", config["m1_index"]))
    index = pd.read_parquet(index_path).copy()
    id_column = "sample_id" if "sample_id" in index else "image_id"
    feature_frame = pd.DataFrame(
        {
            id_column: sample_ids.astype(str),
            "feature_position": np.arange(len(sample_ids)),
        }
    )
    merged = feature_frame.merge(
        index.assign(**{id_column: index[id_column].astype(str)}),
        on=id_column,
        how="inner",
        validate="one_to_one",
    ).sort_values("feature_position")
    if len(merged) != len(features):
        features = features[merged.feature_position.to_numpy()]
        sample_ids = merged[id_column].astype(str).to_numpy()
    metadata = merged.reset_index(drop=True)
    train_ids = set(
        pd.read_parquet(out / "metadata" / "training_manifest.parquet")[
            id_column
        ].astype(str)
    )
    metadata["is_training"] = metadata[id_column].astype(str).isin(train_ids)
    return features, sample_ids, metadata


def _run_features(name: str, config: dict, out: Path) -> dict:
    features, sample_ids, metadata = _feature_frame(config, out)
    settings = config["feature_methods"]
    if name == "featmap":
        method = FEATMAPHarmonizer(settings["featmap_ridge"])
    elif name == "scangen":
        method = ScanGenHarmonizer(
            hidden=settings["scangen_hidden"],
            alpha=settings["scangen_alpha"],
            radius=settings["scangen_radius"],
            epochs=(
                settings["fast_dev_scangen_epochs"]
                if config["fast_dev_run"]
                else settings["scangen_epochs"]
            ),
            seed=config["seed"],
        )
    else:
        raise KeyError(name)
    training_positions = np.flatnonzero(metadata.is_training.to_numpy())
    if (
        name == "scangen"
        and len(training_positions) > settings["scangen_max_train_rows"]
    ):
        training_frame = metadata.iloc[training_positions]
        selected: list[int] = []
        for group in sorted(training_frame.aligned_group_id.astype(str).unique()):
            selected.extend(
                training_frame.index[
                    training_frame.aligned_group_id.astype(str).eq(group)
                ].tolist()
            )
            if len(selected) >= settings["scangen_max_train_rows"]:
                break
        training_positions = np.asarray(selected, dtype=int)
    training_metadata = metadata.iloc[training_positions].reset_index(drop=True)
    method.fit(features[training_positions], training_metadata)
    transformed = method.transform(features, metadata.reset_index(drop=True))
    destination = out / "generated" / name / "features.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_feature_cache(
        destination,
        transformed,
        sample_ids,
        {
            "method": name,
            "fit_rows": len(training_positions),
            "fit_policy": "M1 acquisition-train cells and morphology train only",
        },
    )
    metadata.to_parquet(destination.with_name("feature_metadata.parquet"), index=False)
    method.save(out / "checkpoints" / f"{name}.pt")
    return {
        "status": "COMPLETE",
        "feature_rows": len(transformed),
        "feature_dimension": transformed.shape[1],
        "tracks": ["D"],
        "result_manifest": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_sota_benchmark.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        if args.seed not in config.get("run_seeds", config["seeds"]):
            raise ValueError(
                f"Seed {args.seed} is not selected by SOTA_SEEDS/run_seeds"
            )
        config["seed"] = args.seed
    out = _output(config)
    name = get_baseline(args.method).method_name
    status_path = out / "logs" / f"method_{name}_seed{config['seed']}.json"
    if status_path.exists():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETE" and not os.getenv("FORCE_RERUN_METHOD"):
            print(f"{name}: already complete; resume skipped")
            return
    started = time.monotonic()
    payload = {
        "method": name,
        "display_name": get_baseline(name).display_name,
        "implementation_source": get_baseline(name).implementation_source,
        "seed": config["seed"],
    }
    try:
        if name in FEATURE_METHODS:
            payload.update(_run_features(name, config, out))
        elif name in IMAGE_TRACKS:
            payload.update(_run_images(name, config, out))
        else:
            spec = get_baseline(name)
            payload.update(
                {
                    "status": spec.availability,
                    "tracks": [spec.primary_track],
                    "reason": spec.notes,
                }
            )
    except MethodUnavailable as exc:
        payload.update({"status": "UNAVAILABLE", "reason": str(exc), "tracks": []})
    except Exception as exc:  # noqa: BLE001 - method status captures third-party failures
        payload.update(
            {
                "status": "FAILED",
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "tracks": [],
            }
        )
    payload["wall_seconds"] = time.monotonic() - started
    atomic_json_dump(payload, status_path)
    print(f"{name}: {payload['status']} ({payload['wall_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
