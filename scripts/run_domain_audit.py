#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from factorstain.evaluation.domain_audit import (
    descriptive_silhouette,
    domain_centroids,
    grouped_probe,
    nearest_neighbors,
    paired_cluster_bootstrap,
    plism_controlled_distances,
    project_features,
    sampled_within_between_distances,
)
from factorstain.models.dinov3 import load_feature_cache
from factorstain.utils.domain_audit import (
    load_domain_audit_config,
    prepare_domain_audit_output,
)
from factorstain.utils.runtime import atomic_json_dump, collect_provenance
from factorstain.visualization.domain_features import (
    compose_dashboard,
    plot_centroid_heatmap,
    plot_distance_distribution,
    plot_nearest_neighbors,
    plot_probe_scores,
    plot_projection,
)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _align_metadata(
    metadata: pd.DataFrame, sample_ids: np.ndarray, cache_name: str
) -> pd.DataFrame:
    positions = pd.DataFrame(
        {"sample_id": sample_ids.astype(str), "feature_row": np.arange(len(sample_ids))}
    )
    if positions.sample_id.duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {cache_name}")
    result = positions.merge(
        metadata.assign(sample_id=metadata.sample_id.astype(str)),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if result.image_path.isna().any():
        raise ValueError(
            f"Feature cache {cache_name} contains sample IDs absent from its manifest"
        )
    return result.sort_values("feature_row").reset_index(drop=True)


def _appearance_features(statistics: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    excluded = {
        "sample_id",
        "scanner_id",
        "stain_id",
        "tissue_type",
        "aligned_group_id",
        "case_id",
    }
    columns = [
        column
        for column in statistics.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    values = statistics[columns].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median()).fillna(0.0)
    keep = values.std(axis=0) > 0
    return values.loc[:, keep].to_numpy(dtype=np.float32), list(values.columns[keep])


def _mean_off_diagonal(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy()
    return (
        float(values[~np.eye(len(values), dtype=bool)].mean())
        if len(values) > 1
        else 0.0
    )


def _metrics_rows(
    probes: list[dict],
    appearance_probes: list[dict],
    silhouettes: dict,
    summaries: dict,
) -> list[dict]:
    rows: list[dict] = []
    for probe in probes:
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "chance",
            "normalized_advantage",
            "group_overlap_count",
        ):
            rows.append(
                {
                    "dataset": probe["dataset"],
                    "analysis": "dinov3_linear_probe",
                    "target": probe["target"],
                    "metric": metric,
                    "value": probe[metric],
                }
            )
    for probe in appearance_probes:
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "chance",
            "normalized_advantage",
            "group_overlap_count",
        ):
            rows.append(
                {
                    "dataset": probe["dataset"],
                    "analysis": "raw_appearance_linear_probe",
                    "target": probe["target"],
                    "metric": metric,
                    "value": probe[metric],
                }
            )
    for key, value in silhouettes.items():
        dataset, target = key.split("/", 1)
        rows.append(
            {
                "dataset": dataset,
                "analysis": "pca_silhouette",
                "target": target,
                "metric": "silhouette",
                "value": value["value"],
            }
        )
    for dataset, values in summaries.items():
        for metric, value in values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "analysis": "feature_distance",
                    "target": "domain",
                    "metric": metric,
                    "value": value,
                }
            )
    return rows


def _recommendation(plism_stain: str, plism_scanner: str, midog_scanner: str) -> str:
    if plism_stain == "WEAK" and plism_scanner == "WEAK" and midog_scanner == "WEAK":
        return "DOMAIN_SIGNAL_WEAK_REVIEW_PROJECT"
    if plism_scanner != "WEAK" and plism_stain == "WEAK":
        return "PROCEED_BUT_SCANNER_FOCUS"
    if plism_stain != "WEAK" and plism_scanner == "WEAK":
        return "PROCEED_BUT_STAIN_FOCUS"
    return "PROCEED_TO_M0"


def _probe_lookup(probes: list[dict], dataset: str, target: str) -> dict:
    return next(
        item
        for item in probes
        if item["dataset"] == dataset and item["target"] == target
    )


def _report(
    destination: Path,
    config: dict,
    model: dict,
    probes: list[dict],
    appearance: list[dict],
    silhouettes: dict,
    controlled: pd.DataFrame,
    bootstraps: list[dict],
    centroid_summaries: dict,
    rejection: dict,
    recommendation: str,
) -> None:
    ps = _probe_lookup(probes, "plism", "stain")
    pq = _probe_lookup(probes, "plism", "scanner")
    pt = _probe_lookup(probes, "plism", "tissue")
    mq = _probe_lookup(probes, "midog21", "scanner-domain")
    aps = _probe_lookup(appearance, "plism", "stain")
    apq = _probe_lookup(appearance, "plism", "scanner")
    amq = _probe_lookup(appearance, "midog21", "scanner-domain")
    controlled_means = controlled.groupby("category").distance.mean().to_dict()
    same_name = "same morphology / same stain / different scanner"
    same_distance = controlled_means.get(same_name, float("nan"))
    tissue_relation = (
        "stronger"
        if pt["normalized_advantage"]
        > max(ps["normalized_advantage"], pq["normalized_advantage"])
        else "weaker"
    )
    bootstrap_text = (
        "; ".join(
            f"vs {item['comparison']}: difference {item['mean_difference']:.4f} (95% CI {item['ci_low']:.4f} to {item['ci_high']:.4f}, {item['n_groups']} aligned groups)"
            for item in bootstraps
            if item["n_groups"]
        )
        or "insufficient paired aligned groups for a confidence interval"
    )
    validity = (
        "FAST_DEV_RUN debugging output; not valid for scientific interpretation"
        if config["fast_dev_run"]
        else "full configured exploratory audit"
    )
    text = f"""# FactorStain M-1: Domain Feature Audit

Run status: **{validity}**. This preflight is exploratory and does not stop or change M0–M5.

## DINOv3 representation

- Model: `{model["model_name"]}`
- Feature dimension: {model["feature_dimension"]}
- Pooling strategy: `{model["pooling_strategy"]}`
- Processor: `{model["processor_class"]}`; input resolution `{model["input_resolution"]}`; processor size `{model["processor_size"]}`; crop `{model["crop_size"]}`
- Image normalization mean/std: `{model["image_mean"]}` / `{model["image_std"]}`
- The saved NPZ arrays contain raw frozen embeddings. L2 normalization is applied only for PCA, UMAP, cosine distances, and nearest-neighbor analysis.

## PLISM

1. **Are stain domains visibly different in raw image space?** The raw grids are the primary evidence. Basic pixel/color/OD/sharpness statistics predict stain with group-held-out BA {aps["balanced_accuracy"]:.3f} (chance {aps["chance"]:.3f}; {aps["domain_signal"]} measurable appearance signal). Visual confirmation remains necessary because a statistic is not a subjective visual judgment.
2. **Are scanner domains visibly different?** Basic appearance statistics predict scanner with aligned-group-held-out BA {apq["balanced_accuracy"]:.3f} (chance {apq["chance"]:.3f}; {apq["domain_signal"]}). Inspect `figures/plism/plism_random_by_scanner.png` and the same-tissue/same-stain grids.
3. **Are scanner domains separable in DINOv3 space?** {pq["domain_signal"]}: group-held-out BA {pq["balanced_accuracy"]:.3f}, normalized advantage {pq["normalized_advantage"]:.3f}, silhouette {silhouettes["plism/scanner"]["value"]:.3f}.
4. **Are stain domains separable in DINOv3 space?** {ps["domain_signal"]}: group-held-out BA {ps["balanced_accuracy"]:.3f}, normalized advantage {ps["normalized_advantage"]:.3f}, silhouette {silhouettes["plism/stain"]["value"]:.3f}.
5. **Is tissue identity stronger or weaker than acquisition identity?** Tissue is {tissue_relation} by normalized probe advantage: tissue {pt["normalized_advantage"]:.3f}, stain {ps["normalized_advantage"]:.3f}, scanner {pq["normalized_advantage"]:.3f}.
6. **For the same aligned morphology, how much does changing scanner move the embedding?** Mean cosine distance is {same_distance:.4f}. Cluster-bootstrap comparisons: {bootstrap_text}.
7. **Does scanner information survive morphology control?** The held-out probe has zero aligned-group overlap and BA {pq["balanced_accuracy"]:.3f}. The controlled pair analysis fixes aligned group and stain; its raw pair values are in `metadata/plism_controlled_feature_distances.csv`. This is stronger evidence than UMAP alone, but registration error can still contribute.

Scanner centroids have mean between-domain cosine distance {centroid_summaries["plism"]["between_scanner_centroid_distance"]:.4f}; mean normalized-feature within-scanner variance is {centroid_summaries["plism"]["within_scanner_feature_variance"]:.6f}. Stain centroids have mean between-domain distance {centroid_summaries["plism"]["between_stain_centroid_distance"]:.4f}; mean within-stain variance is {centroid_summaries["plism"]["within_stain_feature_variance"]:.6f}.

PLISM stain comparisons can be serial sections, not literally the same physical tissue section. Stain separation must therefore not be described as a perfectly isolated causal stain effect.

## MIDOG21

1. **Are scanner domains visually different?** The raw scanner grid is the primary evidence. Case-held-out basic appearance BA is {amq["balanced_accuracy"]:.3f} (chance {amq["chance"]:.3f}; {amq["domain_signal"]}).
2. **Are scanner-domain features separable using DINOv3?** {mq["domain_signal"]}, with PCA silhouette {silhouettes["midog21/scanner-domain"]["value"]:.3f}.
3. **How strong is the scanner-domain probe?** Case-held-out accuracy {mq["accuracy"]:.3f}, BA {mq["balanced_accuracy"]:.3f}, macro F1 {mq["macro_f1"]:.3f}, normalized advantage {mq["normalized_advantage"]:.3f}; train/test case overlap is zero.
4. **How large are scanner centroid differences?** Mean between-scanner centroid cosine distance is {centroid_summaries["midog21"]["between_scanner_centroid_distance"]:.4f}; mean within-scanner dispersion is {centroid_summaries["midog21"]["within_scanner_dispersion"]:.4f}, and normalized-feature within-scanner variance is {centroid_summaries["midog21"]["within_scanner_feature_variance"]:.6f}.
5. **What is limited by case composition?** Scanner groups contain different cases. Scanner and case are confounded, so MIDOG results measure scanner-domain separability, not a clean causal scanner effect. PLISM aligned comparisons are better suited to isolating acquisition effects.

## Sampling and safeguards

- Deterministic seed: {config["seed"]}; exact image paths and coordinates are saved in Parquet manifests.
- Sample quality/rejection summary: `{json.dumps(rejection, sort_keys=True)}`.
- All probes are group-held-out. PLISM uses `aligned_group_id`; MIDOG uses `case_id`. Every recorded overlap count is zero.
- PCA coordinates are deterministic; UMAP uses the fixed seed and the same coordinates are recolored across labels.
- Silhouette scores are descriptive and are not used alone for conclusions.

## Overall

Interpretation:

```text
PLISM stain domain signal    = {ps["domain_signal"]}
PLISM scanner domain signal  = {pq["domain_signal"]}
MIDOG scanner domain signal  = {mq["domain_signal"]}
```

Recommendation: **{recommendation}**

This recommendation is advisory only. M-1 never automatically terminates the FactorStain research pipeline.
"""
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m_minus1_domain_audit.yaml")
    args = parser.parse_args()
    config = load_domain_audit_config(args.config)
    out = prepare_domain_audit_output(config)
    metadata_dir, features_dir, figures = (
        out / "metadata",
        out / "features",
        out / "figures",
    )
    plism_features, plism_ids, plism_model = load_feature_cache(
        features_dir / "plism_dinov3.npz"
    )
    midog_features, midog_ids, midog_model = load_feature_cache(
        features_dir / "midog21_dinov3.npz"
    )
    for field in (
        "model_name",
        "feature_dimension",
        "pooling_strategy",
        "processor_class",
        "input_resolution",
    ):
        if plism_model.get(field) != midog_model.get(field):
            raise RuntimeError(
                f"PLISM and MIDOG caches disagree on DINOv3 {field}: {plism_model.get(field)!r} vs {midog_model.get(field)!r}"
            )
    plism = _align_metadata(
        pd.read_parquet(metadata_dir / "plism_samples.parquet"),
        plism_ids,
        "plism_dinov3.npz",
    )
    midog = _align_metadata(
        pd.read_parquet(metadata_dir / "midog_samples.parquet"),
        midog_ids,
        "midog21_dinov3.npz",
    )
    manifest = pd.concat(
        [plism.assign(dataset="plism"), midog.assign(dataset="midog21")],
        ignore_index=True,
        sort=False,
    )
    manifest["dinov3_model"] = plism_model["model_name"]
    manifest["feature_dimension"] = plism_model["feature_dimension"]
    manifest["pooling_strategy"] = plism_model["pooling_strategy"]
    _atomic_parquet(manifest, metadata_dir / "feature_manifest.parquet")

    plism_projection = project_features(
        plism_features, config["projection"], config["seed"]
    )
    midog_projection = project_features(
        midog_features, config["projection"], config["seed"] + 1
    )
    projection_manifest = pd.concat(
        [
            plism[["sample_id"]].assign(
                dataset="plism",
                pca_1=plism_projection["pca_2d"][:, 0],
                pca_2=plism_projection["pca_2d"][:, 1],
                umap_1=plism_projection["umap_2d"][:, 0],
                umap_2=plism_projection["umap_2d"][:, 1],
            ),
            midog[["sample_id"]].assign(
                dataset="midog21",
                pca_1=midog_projection["pca_2d"][:, 0],
                pca_2=midog_projection["pca_2d"][:, 1],
                umap_1=midog_projection["umap_2d"][:, 0],
                umap_2=midog_projection["umap_2d"][:, 1],
            ),
        ],
        ignore_index=True,
    )
    _atomic_parquet(projection_manifest, metadata_dir / "feature_projections.parquet")

    plism_figures = figures / "plism"
    midog_figures = figures / "midog21"
    for method, coordinates in (
        ("pca", plism_projection["pca_2d"]),
        ("umap", plism_projection["umap_2d"]),
    ):
        for target, column in (
            ("stain", "stain_id"),
            ("scanner", "scanner_id"),
            ("tissue", "tissue_type"),
        ):
            plot_projection(
                coordinates,
                plism[column],
                plism_figures / f"plism_dinov3_{method}_by_{target}.png",
                f"PLISM frozen DINOv3 {method.upper()} — colored by {target}",
            )
    plot_projection(
        midog_projection["pca_2d"],
        midog.scanner_id,
        midog_figures / "midog_dinov3_pca_by_scanner.png",
        "MIDOG21 frozen DINOv3 PCA — scanner domain",
    )
    plot_projection(
        midog_projection["umap_2d"],
        midog.scanner_id,
        midog_figures / "midog_dinov3_umap_by_scanner.png",
        "MIDOG21 frozen DINOv3 UMAP — scanner domain",
    )
    plot_projection(
        midog_projection["umap_2d"],
        midog.case_id,
        midog_figures / "midog_dinov3_umap_by_case.png",
        "MIDOG21 frozen DINOv3 UMAP — case identity",
    )

    probes, splits, confusion = [], [], {}
    for dataset, values, frame, targets, group_column in (
        (
            "plism",
            plism_features,
            plism,
            (
                ("stain", "stain_id"),
                ("scanner", "scanner_id"),
                ("tissue", "tissue_type"),
            ),
            "aligned_group_id",
        ),
        (
            "midog21",
            midog_features,
            midog,
            (("scanner-domain", "scanner_id"),),
            "case_id",
        ),
    ):
        for offset, (target, column) in enumerate(targets):
            result, matrix, split = grouped_probe(
                values,
                frame[column],
                frame[group_column],
                target,
                dataset,
                config["probe"]["test_size"],
                config["seed"] + offset,
                config["probe"]["max_iter"],
            )
            probes.append(result)
            splits.append(split.assign(sample_id=frame.sample_id.to_numpy()))
            confusion[f"{dataset}/{target}"] = matrix.tolist()
    _atomic_parquet(
        pd.concat(splits, ignore_index=True),
        metadata_dir / "probe_split_manifest.parquet",
    )

    appearance_probes, appearance_splits = [], []
    for dataset, frame, statistics_path, targets, group_column in (
        (
            "plism",
            plism,
            metadata_dir / "plism_pixel_statistics.parquet",
            (("stain", "stain_id"), ("scanner", "scanner_id")),
            "aligned_group_id",
        ),
        (
            "midog21",
            midog,
            metadata_dir / "midog_pixel_statistics.parquet",
            (("scanner-domain", "scanner_id"),),
            "case_id",
        ),
    ):
        statistics = frame[["sample_id"]].merge(
            pd.read_parquet(statistics_path),
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        appearance_values, appearance_columns = _appearance_features(statistics)
        for offset, (target, column) in enumerate(targets):
            result, _, split = grouped_probe(
                appearance_values,
                frame[column],
                frame[group_column],
                target,
                dataset,
                config["probe"]["test_size"],
                config["seed"] + 30 + offset,
                config["probe"]["max_iter"],
            )
            result["input_columns"] = appearance_columns
            appearance_probes.append(result)
            appearance_splits.append(
                split.assign(
                    sample_id=frame.sample_id.to_numpy(),
                    feature_space="raw_pixel_statistics",
                )
            )
    _atomic_parquet(
        pd.concat(appearance_splits, ignore_index=True),
        metadata_dir / "appearance_probe_split_manifest.parquet",
    )

    silhouettes = {}
    for offset, (key, values, labels) in enumerate(
        (
            ("plism/stain", plism_projection["pca_reduced"], plism.stain_id),
            ("plism/scanner", plism_projection["pca_reduced"], plism.scanner_id),
            ("plism/tissue", plism_projection["pca_reduced"], plism.tissue_type),
            (
                "midog21/scanner-domain",
                midog_projection["pca_reduced"],
                midog.scanner_id,
            ),
        )
    ):
        silhouettes[key] = descriptive_silhouette(
            values,
            labels,
            config["projection"]["silhouette_max_samples"],
            config["seed"] + offset,
        )

    scanner_matrix, _, scanner_dispersion = domain_centroids(
        plism_features, plism.scanner_id
    )
    stain_matrix, _, stain_dispersion = domain_centroids(plism_features, plism.stain_id)
    midog_matrix, _, midog_dispersion = domain_centroids(
        midog_features, midog.scanner_id
    )
    scanner_matrix.to_csv(metadata_dir / "plism_scanner_centroid_distances.csv")
    stain_matrix.to_csv(metadata_dir / "plism_stain_centroid_distances.csv")
    midog_matrix.to_csv(metadata_dir / "midog_scanner_centroid_distances.csv")
    scanner_dispersion.assign(factor="scanner").to_csv(
        metadata_dir / "plism_domain_dispersion.csv", index=False
    )
    stain_dispersion.assign(factor="stain").to_csv(
        metadata_dir / "plism_stain_dispersion.csv", index=False
    )
    midog_dispersion.to_csv(metadata_dir / "midog_scanner_dispersion.csv", index=False)
    plot_centroid_heatmap(
        scanner_matrix,
        plism_figures / "plism_scanner_centroid_distance.png",
        "PLISM scanner centroid cosine distances",
    )
    plot_centroid_heatmap(
        stain_matrix,
        plism_figures / "plism_stain_centroid_distance.png",
        "PLISM stain centroid cosine distances",
    )
    plot_centroid_heatmap(
        midog_matrix,
        midog_figures / "midog_scanner_centroid_distance.png",
        "MIDOG21 scanner-domain centroid cosine distances",
    )

    controlled = plism_controlled_distances(plism_features, plism, config["seed"])
    controlled.to_csv(
        metadata_dir / "plism_controlled_feature_distances.csv", index=False
    )
    plot_distance_distribution(
        controlled,
        plism_figures / "plism_feature_distance_controlled.png",
        "PLISM controlled frozen-DINOv3 feature movement",
    )
    reference = "same morphology / same stain / different scanner"
    bootstraps = [
        paired_cluster_bootstrap(
            controlled,
            reference,
            comparison,
            config["statistics"]["bootstrap_samples"],
            config["statistics"]["confidence"],
            config["seed"] + offset,
        )
        for offset, comparison in enumerate(
            (
                "different morphology / same stain / same scanner",
                "different morphology / same stain / different scanner",
            )
        )
    ]
    pd.DataFrame(bootstraps).to_csv(
        metadata_dir / "plism_controlled_bootstrap.csv", index=False
    )

    midog_distances = sampled_within_between_distances(
        midog_features, midog.scanner_id, midog.sample_id, 20000, config["seed"]
    )
    midog_distances.to_csv(
        metadata_dir / "midog_within_between_scanner_distances.csv", index=False
    )
    plot_distance_distribution(
        midog_distances,
        midog_figures / "midog_within_between_scanner_distance.png",
        "MIDOG21 within- vs between-scanner feature distances",
    )

    plot_probe_scores(
        [
            _probe_lookup(probes, "plism", target)
            for target in ("scanner", "stain", "tissue")
        ],
        plism_figures / "plism_linear_probe_scores.png",
        "PLISM aligned-group-held-out frozen-feature probes",
    )
    plot_probe_scores(
        [_probe_lookup(probes, "midog21", "scanner-domain")],
        midog_figures / "midog_linear_probe_scores.png",
        "MIDOG21 case-held-out scanner-domain probe",
    )
    plism_nn = nearest_neighbors(plism_features, plism.scanner_id, seed=config["seed"])
    midog_nn = nearest_neighbors(midog_features, midog.scanner_id, seed=config["seed"])
    plot_nearest_neighbors(
        plism,
        plism_nn,
        "plism",
        plism_figures / "plism_nn_examples.png",
        "PLISM nearest DINOv3 neighbors (unconstrained retrieval)",
    )
    plot_nearest_neighbors(
        midog,
        midog_nn,
        "midog21",
        midog_figures / "midog_nn_examples.png",
        "MIDOG21 nearest DINOv3 neighbors (unconstrained retrieval)",
    )

    centroid_summaries = {
        "plism": {
            "between_scanner_centroid_distance": _mean_off_diagonal(scanner_matrix),
            "between_stain_centroid_distance": _mean_off_diagonal(stain_matrix),
            "within_scanner_feature_variance": float(
                scanner_dispersion.within_domain_feature_variance.mean()
            ),
            "within_stain_feature_variance": float(
                stain_dispersion.within_domain_feature_variance.mean()
            ),
            "within_scanner_dispersion": float(
                scanner_dispersion.within_domain_mean_cosine_distance.mean()
            ),
            "within_stain_dispersion": float(
                stain_dispersion.within_domain_mean_cosine_distance.mean()
            ),
        },
        "midog21": {
            "between_scanner_centroid_distance": _mean_off_diagonal(midog_matrix),
            "within_scanner_feature_variance": float(
                midog_dispersion.within_domain_feature_variance.mean()
            ),
            "within_scanner_dispersion": float(
                midog_dispersion.within_domain_mean_cosine_distance.mean()
            ),
            "within_scanner_pair_distance": float(
                midog_distances[
                    midog_distances.category.eq("within scanner")
                ].distance.mean()
            ),
            "between_scanner_pair_distance": float(
                midog_distances[
                    midog_distances.category.eq("between scanners")
                ].distance.mean()
            ),
        },
    }
    ps, pq, pt, mq = (
        _probe_lookup(probes, "plism", "stain"),
        _probe_lookup(probes, "plism", "scanner"),
        _probe_lookup(probes, "plism", "tissue"),
        _probe_lookup(probes, "midog21", "scanner-domain"),
    )
    recommendation = _recommendation(
        ps["domain_signal"], pq["domain_signal"], mq["domain_signal"]
    )
    rejection = {}
    quality_path = metadata_dir / "sampling_quality.csv"
    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        for dataset, subset in quality.groupby("dataset"):
            rejection[str(dataset)] = {
                "evaluated": len(subset),
                "rejected": int((subset.reason != "accepted").sum()),
                "rejection_rate": float((subset.reason != "accepted").mean()),
                "reasons": subset.reason.value_counts().to_dict(),
            }
    metrics_rows = _metrics_rows(
        probes, appearance_probes, silhouettes, centroid_summaries
    )
    pd.DataFrame(metrics_rows).to_csv(out / "metrics.csv", index=False)
    payload = {
        "milestone": "m_minus1_domain_audit",
        "audit_valid": not config["fast_dev_run"],
        "fast_dev_run": config["fast_dev_run"],
        "model": plism_model,
        "sampling": {
            "plism_features": len(plism),
            "midog21_features": len(midog),
            "rejection": rejection,
        },
        "probes": probes,
        "appearance_probes": appearance_probes,
        "silhouettes": silhouettes,
        "controlled_plism": {
            "category_means": controlled.groupby("category").distance.mean().to_dict(),
            "bootstrap": bootstraps,
        },
        "centroid_and_dispersion": centroid_summaries,
        "confusion_matrices": confusion,
        "signals": {
            "plism_stain": ps["domain_signal"],
            "plism_scanner": pq["domain_signal"],
            "plism_tissue": pt["domain_signal"],
            "midog_scanner_domain": mq["domain_signal"],
        },
        "recommendation": recommendation,
        "provenance": collect_provenance(
            config["seed"], {"plism": len(plism), "midog21": len(midog)}
        ),
        "caveats": [
            "PLISM stain conditions may be serial sections.",
            "MIDOG21 scanner domains contain different cases, so scanner and case are confounded.",
            "UMAP and silhouette are descriptive; group-held-out probes and PLISM aligned comparisons carry more weight.",
        ],
    }
    atomic_json_dump(payload, out / "metrics.json")
    _report(
        out / "REPORT.md",
        config,
        plism_model,
        probes,
        appearance_probes,
        silhouettes,
        controlled,
        bootstraps,
        centroid_summaries,
        rejection,
        recommendation,
    )

    controlled_visual = sorted(
        plism_figures.glob("plism_same_tissue_same_stain_across_scanners_*.png")
    )
    plism_footer = f"Measured domain signal — scanner: {pq['domain_signal']} | stain: {ps['domain_signal']} | tissue: {pt['domain_signal']}"
    compose_dashboard(
        [
            ("Raw patches by stain", plism_figures / "plism_random_by_stain.png"),
            (
                "Controlled: same morphology + stain",
                controlled_visual[0] if controlled_visual else None,
            ),
            ("UMAP by scanner", plism_figures / "plism_dinov3_umap_by_scanner.png"),
            ("Same UMAP by stain", plism_figures / "plism_dinov3_umap_by_stain.png"),
            ("Same UMAP by tissue", plism_figures / "plism_dinov3_umap_by_tissue.png"),
            ("Group-held-out probes", plism_figures / "plism_linear_probe_scores.png"),
            (
                "Scanner centroid distances",
                plism_figures / "plism_scanner_centroid_distance.png",
            ),
            (
                "Controlled feature distances",
                plism_figures / "plism_feature_distance_controlled.png",
            ),
        ],
        plism_figures / "PLISM_DOMAIN_SUMMARY.png",
        "PLISM DOMAIN FEATURE SUMMARY",
        plism_footer,
    )
    midog_footer = f"Measured scanner-domain signal: {mq['domain_signal']} | Scanner groups contain different cases; this is not a fully causal scanner test."
    compose_dashboard(
        [
            (
                "Raw scanner-domain patches",
                midog_figures / "midog_random_by_scanner.png",
            ),
            ("UMAP by scanner", midog_figures / "midog_dinov3_umap_by_scanner.png"),
            ("PCA by scanner", midog_figures / "midog_dinov3_pca_by_scanner.png"),
            ("Case-held-out probe", midog_figures / "midog_linear_probe_scores.png"),
            (
                "Scanner centroid distances",
                midog_figures / "midog_scanner_centroid_distance.png",
            ),
            (
                "Within vs between distances",
                midog_figures / "midog_within_between_scanner_distance.png",
            ),
        ],
        midog_figures / "MIDOG21_DOMAIN_SUMMARY.png",
        "MIDOG21 DOMAIN FEATURE SUMMARY",
        midog_footer,
    )
    master_footer = f"PLISM scanner signal: {pq['domain_signal']} | PLISM stain signal: {ps['domain_signal']} | MIDOG scanner-domain signal: {mq['domain_signal']}\nRecommendation: {recommendation}"
    compose_dashboard(
        [
            ("PLISM raw domains", plism_figures / "plism_random_by_stain.png"),
            ("MIDOG21 raw domains", midog_figures / "midog_random_by_scanner.png"),
            (
                "PLISM UMAP by scanner",
                plism_figures / "plism_dinov3_umap_by_scanner.png",
            ),
            (
                "MIDOG21 UMAP by scanner",
                midog_figures / "midog_dinov3_umap_by_scanner.png",
            ),
            (
                "PLISM same UMAP by stain",
                plism_figures / "plism_dinov3_umap_by_stain.png",
            ),
            (
                "MIDOG21 PCA by scanner",
                midog_figures / "midog_dinov3_pca_by_scanner.png",
            ),
            (
                "PLISM group-held-out probes",
                plism_figures / "plism_linear_probe_scores.png",
            ),
            (
                "MIDOG21 case-held-out probe",
                midog_figures / "midog_linear_probe_scores.png",
            ),
            (
                "PLISM controlled distance",
                plism_figures / "plism_feature_distance_controlled.png",
            ),
            (
                "MIDOG21 centroid distance",
                midog_figures / "midog_scanner_centroid_distance.png",
            ),
        ],
        figures / "SUMMARY_DASHBOARD.png",
        "M-1 DOMAIN FEATURE AUDIT",
        master_footer,
    )

    print("=" * 60)
    print("FactorStain M-1 Domain Audit Complete")
    print("=" * 60)
    print(f"\nDINOv3:\n{plism_model['model_name']}")
    print(
        f"\nPLISM:\n  stain domain signal   : {ps['domain_signal']}\n  scanner domain signal : {pq['domain_signal']}\n  tissue signal         : {pt['domain_signal']}"
    )
    print(
        f"\nMIDOG21:\n  scanner-domain signal : {mq['domain_signal']}\n  NOTE: case/scanner confounding exists"
    )
    print(
        f"\nMain dashboard:\n{figures / 'SUMMARY_DASHBOARD.png'}\n\nPLISM dashboard:\n{plism_figures / 'PLISM_DOMAIN_SUMMARY.png'}\n\nMIDOG dashboard:\n{midog_figures / 'MIDOG21_DOMAIN_SUMMARY.png'}\n\nReport:\n{out / 'REPORT.md'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
