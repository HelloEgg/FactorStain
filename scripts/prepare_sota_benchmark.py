#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from factorstain.baselines.benchmark import benchmark_output_config
from factorstain.baselines.registry import BASELINES
from factorstain.evaluation.renderer import build_evaluation_pairs
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output
from factorstain.utils.runtime import atomic_json_dump

THIRD_PARTY_DIRECTORIES = {
    "stainnet": "StainNet",
    "staingan": "StainGAN",
    "cyclegan": "pytorch-CycleGAN-and-pix2pix",
    "pix2pix": "pytorch-CycleGAN-and-pix2pix",
    "histaugan": "HistAuGAN",
    "cagan": "CAGAN",
    "sastaindiff": "SAStainDiff",
    "histofs": "HistoFS",
}


def _resolve(config: dict, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    project = Path(config["paths"]["project_root"])
    candidate = project / path
    if candidate.exists():
        return candidate
    # Configs remain portable when OUTPUTS_ROOT points outside PROJECT_ROOT.
    parts = path.parts
    if parts and parts[0] == "outputs":
        return Path(config["paths"]["outputs_root"]).joinpath(*parts[1:])
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _observed_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cell_set(items: list[dict]) -> set[tuple[str, str]]:
    return {(str(item["stain_id"]), str(item["scanner_id"])) for item in items}


def _sample_id(frame: pd.DataFrame) -> pd.Series:
    column = "sample_id" if "sample_id" in frame else "image_id"
    return frame[column].astype(str)


def validate_split(index: pd.DataFrame, split: dict) -> dict[str, Any]:
    train = _cell_set(split["train_cells"])
    validation = _cell_set(split.get("validation_cells", []))
    test = _cell_set(split.get("test_cells", split["heldout_cells"]))
    if train & validation or train & test or validation & test:
        raise RuntimeError("M1 acquisition split cells overlap")
    train_stains = {stain for stain, _ in train}
    train_scanners = {scanner for _, scanner in train}
    missing_stains = sorted({stain for stain, _ in test} - train_stains)
    missing_scanners = sorted({scanner for _, scanner in test} - train_scanners)
    if missing_stains or missing_scanners:
        raise RuntimeError(
            f"Held-out factors are not individually seen: stains={missing_stains}, "
            f"scanners={missing_scanners}"
        )
    indexed_cells = set(
        zip(index.stain_id.astype(str), index.scanner_id.astype(str), strict=False)
    )
    if not test <= indexed_cells:
        raise RuntimeError("M1 test split contains cells absent from the frozen index")
    return {
        "train_cells": sorted([list(cell) for cell in train]),
        "validation_cells": sorted([list(cell) for cell in validation]),
        "test_cells": sorted([list(cell) for cell in test]),
        "target_stains_seen_in_train": not missing_stains,
        "target_scanners_seen_in_train": not missing_scanners,
        "cell_partitions_disjoint": True,
    }


def _enrich_manifest(pairs: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "sample_id",
        "image_id",
        "image_path",
        "stain_id",
        "scanner_id",
        "tissue_type",
        "aligned_group_id",
        "morphology_split",
    ]
    available = [field for field in fields if field in index]
    source = index[available].add_prefix("source_")
    target = index[available].add_prefix("target_")
    enriched = pairs.merge(
        source,
        left_on="source_index",
        right_index=True,
        how="left",
        validate="many_to_one",
    ).merge(
        target,
        left_on="target_index",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    enriched.insert(
        0, "episode_id", [f"m1-eval-{value:08d}" for value in range(len(enriched))]
    )
    return enriched


def _track_episodes(
    pairs: pd.DataFrame, index: pd.DataFrame, train_cells: set[tuple[str, str]]
) -> pd.DataFrame:
    """Derive valid A/B subsets while preserving exact M1 Track-C episodes."""
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        record = pair._asdict()
        joint_target = index.loc[int(pair.target_index)]
        record.update(
            {
                "heldout_target_stain_id": str(joint_target.stain_id),
                "heldout_target_scanner_id": str(joint_target.scanner_id),
            }
        )
        if str(pair.protocol).startswith("unseen_combo_"):
            rows.append({**record, "track": "C"})
        if str(pair.protocol) == "controlled_scanner_transfer":
            rows.append({**record, "track": "B"})
        if str(pair.protocol) != "unseen_combo_unseen_morphology":
            continue
        source = index.loc[int(pair.source_index)]
        target = index.loc[int(pair.target_index)]
        candidates = index[
            index.aligned_group_id.astype(str).eq(str(target.aligned_group_id))
            & index.stain_id.astype(str).eq(str(target.stain_id))
            & index.scanner_id.astype(str).eq(str(source.scanner_id))
            & index.morphology_split.eq("test")
        ]
        candidates = candidates[
            candidates.apply(
                lambda row: (str(row.stain_id), str(row.scanner_id)) in train_cells,
                axis=1,
            )
        ]
        if not candidates.empty:
            rows.append(
                {
                    **record,
                    "target_index": int(candidates.index[0]),
                    "protocol": "stain_translation_unseen_morphology",
                    "track": "A",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No scientifically valid Track A/B/C episodes remain")
    return result.drop_duplicates(
        ["source_index", "target_index", "protocol", "track"]
    ).reset_index(drop=True)


def _image_statistics(path: str, size: int = 64) -> np.ndarray:
    with Image.open(path) as opened:
        rgb = (
            np.asarray(
                opened.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
                dtype=np.float64,
            )
            / 255.0
        )
    optical_density = -np.log(np.clip(rgb, 1e-6, 1))
    return np.concatenate(
        [rgb.mean((0, 1)), rgb.std((0, 1)), optical_density.mean((0, 1))]
    )


def _reference_policy(
    train: pd.DataFrame,
    forbidden: pd.DataFrame,
    split_hashes: dict[str, str],
    pool_limit: int,
) -> dict[str, Any]:
    train = train.copy()
    train["_sample_id"] = _sample_id(train)
    prototypes: dict[str, dict[str, Any]] = {}
    for stain, frame in train.groupby("stain_id", sort=True):
        ordered = frame.assign(
            _key=frame["_sample_id"].map(
                lambda value: hashlib.sha256(f"42:{value}".encode()).hexdigest()
            )
        ).sort_values("_key")
        pool = ordered.head(pool_limit)
        statistics = np.stack([_image_statistics(path) for path in pool.image_path])
        median = np.median(statistics, axis=0)
        selected_position = int(np.linalg.norm(statistics - median, axis=1).argmin())
        selected = pool.iloc[selected_position]
        prototypes[str(stain)] = {
            "selection": "training-pool color-statistic medoid",
            "sample_id": str(selected["_sample_id"]),
            "image_id": str(selected.get("image_id", selected["_sample_id"])),
            "image_path": str(selected.image_path),
            "scanner_id": str(selected.scanner_id),
            "aligned_group_id": str(selected.aligned_group_id),
            "pool_sample_ids": pool["_sample_id"].astype(str).tolist(),
            "pool_cells": sorted(
                {
                    f"{row.stain_id}x{row.scanner_id}"
                    for row in pool[["stain_id", "scanner_id"]].itertuples()
                }
            ),
        }
    scanner_pools = {
        str(scanner): sorted(_sample_id(frame).tolist())
        for scanner, frame in train.groupby("scanner_id", sort=True)
    }
    forbidden_ids = sorted(_sample_id(forbidden).tolist())
    used_ids = sorted(
        {
            value
            for details in prototypes.values()
            for value in details["pool_sample_ids"]
        }
    )
    if set(used_ids) & set(forbidden_ids):
        raise RuntimeError("Held-out target leaked into target-stain reference pool")
    return {
        "policy_version": 1,
        "strict_rule": (
            "Target stain references and scanner transforms use only M1 acquisition-train "
            "cells with morphology_split=train. Real held-out joint targets are evaluation-only."
        ),
        "combination_split_sha256": split_hashes["combination_split"],
        "morphology_split_sha256": split_hashes["morphology_split"],
        "stain_prototypes": prototypes,
        "target_scanner_training_pools": scanner_pools,
        "reference_pool_sample_ids_sha256": _ids_sha256(used_ids),
        "forbidden_target_sample_ids": forbidden_ids,
        "forbidden_target_ids_sha256": _ids_sha256(forbidden_ids),
        "oracle_target_reference_allowed": False,
        "validation_target_reference_allowed": False,
    }


def _scanner_pairs(train: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    for (group, stain), frame in train.groupby(
        ["aligned_group_id", "stain_id"], sort=True
    ):
        for source in frame.itertuples():
            for target in frame.itertuples():
                if str(source.scanner_id) == str(target.scanner_id):
                    continue
                rows.append(
                    {
                        "aligned_group_id": str(group),
                        "stain_id": str(stain),
                        "source_sample_id": str(
                            getattr(source, "sample_id", source.Index)
                        ),
                        "target_sample_id": str(
                            getattr(target, "sample_id", target.Index)
                        ),
                        "source_scanner": str(source.scanner_id),
                        "target_scanner": str(target.scanner_id),
                        "source_path": str(source.image_path),
                        "target_path": str(target.image_path),
                    }
                )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise RuntimeError(
            "No clean same-group/same-stain training scanner pairs exist"
        )
    limited = []
    for number, (_, frame) in enumerate(
        pairs.groupby(["source_scanner", "target_scanner"], sort=True)
    ):
        if len(frame) > limit:
            frame = frame.sample(limit, random_state=seed + number)
        limited.append(frame)
    return pd.concat(limited, ignore_index=True)


def _literature_markdown() -> str:
    lines = [
        "# Literature baselines and provenance",
        "",
        "This file is generated from the frozen baseline registry. `source_commit` is the",
        "reviewed upstream HEAD at implementation time; gated model revisions are resolved",
        "only after authenticated license acceptance. Missing methods are never replaced by",
        "a look-alike model.",
        "",
        "| Method | Year | Source | Repository | Commit/revision | Availability | Scientific scope |",
        "|---|---:|---|---|---|---|---|",
    ]
    for spec in BASELINES.values():
        repository = (
            f"[{spec.official_repository}]({spec.official_repository})"
            if spec.official_repository
            else "—"
        )
        lines.append(
            f"| {spec.display_name} | {spec.paper_year or '—'} | {spec.implementation_source} | "
            f"{repository} | `{spec.source_commit or 'unresolved/not public'}` | "
            f"{spec.availability} | {spec.notes} |"
        )
    lines.extend(
        [
            "",
            "## Verification conclusions",
            "",
            "- PathoROB is a benchmark, not a model. PLISM outputs are labeled `PathoROB-inspired` unless the official definition is reproduced exactly.",
            "  Official benchmark code: https://github.com/bifold-pathomics/PathoROB at reviewed commit `6583cf0b0d902c8cc032308262fa3a3befdc0687`.",
            "- HistoFS is a federated WSI feature-augmentation/classification method; it is not placed in image-generation tables.",
            "- ScanGen is a downstream projection/loss, and FEATMAP is an embedding affine mapping; both are Track D only.",
            "- I-GAN's 2026 paper exists, but no official public code was found. It is `NOT_REPRODUCIBLE_FROM_AVAILABLE_RESOURCES`.",
            "- Phaet and Mascaret are official gated external-data models; their access terms and exact downloaded revisions must be recorded at execution time.",
            "- Pix2Pix is valid only for aligned same-group/same-stain scanner transfer; serial cross-stain sections are not treated as pixel pairs.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1_sota_benchmark.yaml")
    args = parser.parse_args()
    config = benchmark_output_config(load_config(args.config))
    out = prepare_output(config)
    for child in ("generated", "metadata", "tables", "figures", "checkpoints", "logs"):
        (out / child).mkdir(parents=True, exist_ok=True)

    split_path = _resolve(config, config["combination_split"])
    morphology_path = _resolve(config, config["morphology_split"])
    index_path = _resolve(config, os.getenv("M1_INDEX", config["m1_index"]))
    for required in (split_path, morphology_path, index_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Required frozen M1 artifact is missing: {required}. Run M1 first; this "
                "benchmark will not regenerate a split."
            )
    split_hashes = {
        "combination_split": _sha256(split_path),
        "morphology_split": _sha256(morphology_path),
    }
    split = json.loads(split_path.read_text(encoding="utf-8"))
    index = pd.read_parquet(index_path).reset_index(drop=True)
    audit = validate_split(index, split)
    train_cells = _cell_set(split["train_cells"])
    test_cells = _cell_set(split.get("test_cells", split["heldout_cells"]))
    cell = list(
        zip(index.stain_id.astype(str), index.scanner_id.astype(str), strict=False)
    )
    training_mask = index.morphology_split.eq("train") & pd.Series(cell).isin(
        train_cells
    )
    heldout_mask = pd.Series(cell).isin(test_cells)
    train = index[training_mask].copy()
    validation_mask = index.morphology_split.eq("val") & pd.Series(cell).isin(
        train_cells | _cell_set(split.get("validation_cells", []))
    )
    validation = index[validation_mask].copy()
    forbidden = index[heldout_mask].copy()
    if train.empty or forbidden.empty:
        raise RuntimeError(
            "Frozen M1 index has no strict training or held-out target rows"
        )

    pairs = build_evaluation_pairs(
        index,
        split,
        seed=config["seed"],
        limit=(
            config["fast_dev_eval_per_protocol"]
            if config["fast_dev_run"]
            else config["max_eval_per_protocol"]
        ),
    )
    manifest = _enrich_manifest(_track_episodes(pairs, index, train_cells), index)
    if manifest.empty:
        raise RuntimeError(
            "The exact M1 evaluation episode builder produced no episodes"
        )
    target_ids = set(manifest.target_sample_id.astype(str))
    train_ids = set(_sample_id(train))
    if target_ids & train_ids:
        raise RuntimeError("Evaluation target images leaked into strict training rows")
    unseen_groups = set(
        manifest.loc[
            manifest.protocol.eq("unseen_combo_unseen_morphology"),
            "target_aligned_group_id",
        ].astype(str)
    )
    train_groups = set(train.aligned_group_id.astype(str))
    if unseen_groups & train_groups:
        raise RuntimeError("aligned_group_id leakage in unseen-morphology protocol")

    scanner_pairs = _scanner_pairs(
        train, config["scanner_fit_pairs_per_mapping"], config["seed"]
    )
    policy = _reference_policy(
        train,
        forbidden,
        split_hashes,
        config["prototype_pool_limit"],
    )
    scanner_fit_ids = set(scanner_pairs.source_sample_id) | set(
        scanner_pairs.target_sample_id
    )
    if scanner_fit_ids & set(policy["forbidden_target_sample_ids"]):
        raise RuntimeError("Held-out target leaked into scanner calibration")
    policy["scanner_pair_sample_ids_sha256"] = _ids_sha256(sorted(scanner_fit_ids))
    policy["scanner_pair_count"] = len(scanner_pairs)

    manifest.to_parquet(out / "metadata" / "evaluation_manifest.parquet", index=False)
    train.assign(is_training=True).to_parquet(
        out / "metadata" / "training_manifest.parquet", index=False
    )
    validation.assign(is_training=False).to_parquet(
        out / "metadata" / "validation_manifest.parquet", index=False
    )
    scanner_pairs.to_parquet(
        out / "metadata" / "scanner_fit_pairs.parquet", index=False
    )
    atomic_json_dump(policy, out / "metadata" / "reference_policy.json")
    audit.update(
        {
            "combination_split_path": str(split_path),
            "morphology_split_path": str(morphology_path),
            "combination_split_sha256": split_hashes["combination_split"],
            "morphology_split_sha256": split_hashes["morphology_split"],
            "evaluation_episodes": len(manifest),
            "strict_training_samples": len(train),
            "strict_validation_samples": len(validation),
            "heldout_target_samples": len(forbidden),
            "evaluation_targets_disjoint_from_training": True,
            "unseen_morphology_groups_disjoint_from_training": True,
            "reference_and_scanner_fit_disjoint_from_heldout_targets": True,
        }
    )
    atomic_json_dump(audit, out / "metadata" / "leakage_audit.json")

    capability = pd.DataFrame([spec.to_dict() for spec in BASELINES.values()])
    capability.to_csv(out / "baseline_capability_matrix.csv", index=False)
    provenance = {}
    third_party_root = Path(config["paths"]["project_root"]) / "third_party"
    for name, spec in BASELINES.items():
        installed_path = (
            third_party_root / THIRD_PARTY_DIRECTORIES[name]
            if name in THIRD_PARTY_DIRECTORIES
            else None
        )
        observed = _observed_commit(installed_path) if installed_path else None
        if observed and spec.source_commit and observed != spec.source_commit:
            raise RuntimeError(
                f"Third-party commit drift for {name}: {observed} != {spec.source_commit}"
            )
        provenance[name] = {
            **spec.to_dict(),
            "resolved_commit": observed or spec.source_commit or None,
            "installed_path": str(installed_path) if observed else None,
            "source_checkout_verified": bool(observed),
            "provenance_status": spec.availability,
        }
    provenance["_evaluation_resources"] = {
        "PathoROB": {
            "classification": "benchmark_not_model",
            "official_repository": "https://github.com/bifold-pathomics/PathoROB",
            "reviewed_commit": "6583cf0b0d902c8cc032308262fa3a3befdc0687",
            "PLISM_metric_label": "PathoROB-inspired",
            "reason": "PLISM does not reproduce the official PathoROB datasets/protocol exactly.",
        }
    }
    atomic_json_dump(provenance, out / "metadata" / "method_provenance.json")
    (out / "LITERATURE_BASELINES.md").write_text(
        _literature_markdown(), encoding="utf-8"
    )
    track_counts = ", ".join(
        f"{track}={count}"
        for track, count in manifest.track.value_counts().sort_index().items()
    )
    print(f"Frozen combination split SHA256: {split_hashes['combination_split']}")
    print(f"Frozen morphology split SHA256:  {split_hashes['morphology_split']}")
    print(
        f"Prepared immutable evaluation manifest: {len(manifest):,} episodes; "
        f"tracks: {track_counts}; strict training rows: {len(train):,}; "
        f"scanner pairs: {len(scanner_pairs):,}"
    )


if __name__ == "__main__":
    main()
