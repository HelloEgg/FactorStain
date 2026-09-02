#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from factorstain.evaluation.domain_audit import domain_centroids
from factorstain.evaluation.mmd import (
    balanced_domain_indices,
    balanced_stain_indices,
    benjamini_hochberg,
    bootstrap_mmd2,
    compute_mmd2,
    controlled_scanner_indices,
    l2_normalize,
    matrix_from_pair_records,
    off_diagonal_summary,
    permutation_test_mmd2,
    stable_seed,
)
from factorstain.models.dinov3 import load_feature_cache
from factorstain.utils.config import load_config, save_resolved_config
from factorstain.utils.runtime import atomic_json_dump, collect_provenance
from factorstain.visualization.domain_features import compose_dashboard
from factorstain.visualization.mmd import (
    plot_centroid_vs_mmd,
    plot_gap_distributions,
    plot_mmd_heatmap,
)


def _load_config(path: str | Path) -> dict:
    config = load_config(path)
    settings = config["mmd"]
    if os.getenv("MAX_SAMPLES_PER_DOMAIN"):
        settings["max_samples_per_domain"] = int(os.environ["MAX_SAMPLES_PER_DOMAIN"])
    if os.getenv("MMD_BOOTSTRAPS"):
        settings["bootstrap_replicates"] = int(os.environ["MMD_BOOTSTRAPS"])
    if os.getenv("MMD_PERMUTATIONS"):
        settings["permutation_replicates"] = int(os.environ["MMD_PERMUTATIONS"])
    config["run_mmd_permutation"] = os.getenv("RUN_MMD_PERMUTATION", "0") == "1"
    config["run_pca_sensitivity"] = (
        os.getenv("RUN_MMD_PCA", "0") == "1" or settings["run_pca_sensitivity"]
    )
    return config


def _prepare_output(config: dict) -> Path:
    out = Path(config["paths"]["outputs_root"]) / config["milestone"]
    for child in ("tables", "figures", "logs"):
        (out / child).mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, out / "config_resolved.yaml")
    return out


def _locate_source(config: dict) -> tuple[Path, Path, Path]:
    source = Path(config["paths"]["outputs_root"]) / config["source_milestone"]
    feature = source / "features" / "plism_dinov3.npz"
    metadata = source / "metadata" / "plism_samples.parquet"
    if not feature.exists():
        matches = sorted(source.rglob("plism_dinov3.npz")) if source.exists() else []
        if len(matches) == 1:
            feature = matches[0]
    if not metadata.exists():
        matches = (
            sorted(source.rglob("plism_samples.parquet")) if source.exists() else []
        )
        if len(matches) == 1:
            metadata = matches[0]
    missing = [str(path) for path in (feature, metadata) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The MMD stage only reuses the completed M-1 audit and will not extract features. "
            f"Missing cached input(s): {missing}. Copy/mount the existing outputs or complete "
            "bash shell/m_minus1_domain_audit.sh first."
        )
    return source, feature, metadata


def _align_metadata(metadata: pd.DataFrame, sample_ids: np.ndarray) -> pd.DataFrame:
    required = {
        "sample_id",
        "scanner_id",
        "stain_id",
        "tissue_type",
        "aligned_group_id",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"PLISM sample metadata is missing required MMD columns: {sorted(missing)}"
        )
    positions = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "feature_position": np.arange(len(sample_ids)),
        }
    )
    aligned = positions.merge(
        metadata.assign(sample_id=metadata.sample_id.astype(str)),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if aligned.scanner_id.isna().any():
        missing_ids = (
            aligned.loc[aligned.scanner_id.isna(), "sample_id"].head().tolist()
        )
        raise ValueError(
            f"Cached feature IDs are absent from PLISM metadata: {missing_ids}"
        )
    return aligned.sort_values("feature_position").reset_index(drop=True)


def _selection(
    mode: str,
    metadata: pd.DataFrame,
    domain_a: str,
    domain_b: str,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    if mode == "scanner_marginal":
        left, right, info = balanced_domain_indices(
            metadata.scanner_id, domain_a, domain_b, max_samples, seed
        )
        return left, right, None, None, info
    if mode == "scanner_controlled":
        left, right, clusters, info = controlled_scanner_indices(
            metadata, domain_a, domain_b, max_samples, seed
        )
        return left, right, clusters, None, info
    if mode == "stain_marginal":
        left, right, info = balanced_domain_indices(
            metadata.stain_id, domain_a, domain_b, max_samples, seed
        )
        return left, right, None, None, info
    if mode == "stain_balanced":
        left, right, strata, info = balanced_stain_indices(
            metadata, domain_a, domain_b, max_samples, seed
        )
        return left, right, None, strata, info
    raise KeyError(f"Unknown MMD analysis mode {mode!r}")


def _evaluate_pairs(
    features: np.ndarray,
    metadata: pd.DataFrame,
    domains: list[str],
    mode: str,
    config: dict,
    bootstrap: bool = False,
    permutation: bool = False,
) -> pd.DataFrame:
    settings, rows = config["mmd"], []
    bootstrap_replicates = (
        settings["fast_dev_bootstrap_replicates"]
        if config["fast_dev_run"]
        else settings["bootstrap_replicates"]
    )
    total = len(domains) * (len(domains) - 1) // 2
    for pair_number, (domain_a, domain_b) in enumerate(
        combinations(domains, 2), start=1
    ):
        pair_seed = stable_seed(config["seed"], mode, domain_a, domain_b)
        left_indices, right_indices, clusters, strata, info = _selection(
            mode,
            metadata,
            domain_a,
            domain_b,
            settings["max_samples_per_domain"],
            pair_seed,
        )
        left, right = features[left_indices], features[right_indices]
        result = compute_mmd2(
            left,
            right,
            settings["kernel_scales"],
            settings["bandwidth_distance_pairs"],
            pair_seed,
        )
        row = {
            "analysis": mode,
            "domain_a": domain_a,
            "domain_b": domain_b,
            **info,
            **result,
        }
        if bootstrap:
            row.update(
                bootstrap_mmd2(
                    left,
                    right,
                    result["sigma"],
                    settings["kernel_scales"],
                    bootstrap_replicates,
                    stable_seed(pair_seed, "bootstrap"),
                    cluster_ids=clusters,
                    strata_ids=strata,
                )
            )
        if permutation:
            row.update(
                permutation_test_mmd2(
                    left,
                    right,
                    result["sigma"],
                    settings["kernel_scales"],
                    settings["permutation_replicates"],
                    stable_seed(pair_seed, "permutation"),
                )
            )
        rows.append(row)
        print(
            f"[{mode} {pair_number:>3}/{total}] {domain_a} vs {domain_b}: "
            f"MMD²={result['reported_mmd2']:.6f}, n={info['n_used']}"
        )
    frame = pd.DataFrame(rows)
    if permutation and len(frame):
        frame["permutation_p_value_fdr"] = benjamini_hochberg(
            frame.permutation_p_value.to_numpy()
        )
    return frame


def _write_matrix(matrix: pd.DataFrame, path: Path) -> None:
    matrix.to_csv(path, index=True, index_label="domain")


def _pair_extreme(records: pd.DataFrame, largest: bool) -> dict:
    position = (
        records.reported_mmd2.idxmax() if largest else records.reported_mmd2.idxmin()
    )
    row = records.loc[position]
    return {
        "domain_a": str(row.domain_a),
        "domain_b": str(row.domain_b),
        "mmd2": float(row.reported_mmd2),
    }


def _off_diagonal_values(matrix: pd.DataFrame, label: str) -> pd.DataFrame:
    values = matrix.to_numpy()[np.triu_indices(len(matrix), k=1)]
    return pd.DataFrame({"analysis": label, "mmd2": values})


def _load_centroid_matrix(
    source: Path,
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanners: list[str],
) -> tuple[pd.DataFrame, str]:
    cached = source / "metadata" / "plism_scanner_centroid_distances.csv"
    if cached.exists():
        matrix = pd.read_csv(cached, index_col=0)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str)
        return matrix.reindex(index=scanners, columns=scanners), str(cached)
    matrix, _, _ = domain_centroids(features, metadata.scanner_id)
    return matrix.reindex(
        index=scanners, columns=scanners
    ), "computed from reused cached features"


def _centroid_pair_table(
    controlled: pd.DataFrame,
    centroid_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    rows = []
    for _, pair in controlled.iterrows():
        rows.append(
            {
                "domain_a": pair.domain_a,
                "domain_b": pair.domain_b,
                "centroid_cosine_distance": float(
                    centroid_matrix.loc[str(pair.domain_a), str(pair.domain_b)]
                ),
                "controlled_mmd2": float(pair.reported_mmd2),
            }
        )
    frame = pd.DataFrame(rows)
    correlation = spearmanr(frame.centroid_cosine_distance, frame.controlled_mmd2)
    return frame, float(correlation.statistic), float(correlation.pvalue)


def _pair_order_correlation(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[float, float]:
    merged = left[["domain_a", "domain_b", "reported_mmd2"]].merge(
        right[["domain_a", "domain_b", "reported_mmd2"]],
        on=["domain_a", "domain_b"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    result = spearmanr(merged.reported_mmd2_left, merged.reported_mmd2_right)
    return float(result.statistic), float(result.pvalue)


def _scanner_probe_accuracy(source: Path) -> float | None:
    path = source / "metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for probe in payload.get("probes", []):
        if probe.get("dataset") == "plism" and probe.get("target") == "scanner":
            value = probe.get("balanced_accuracy")
            return float(value) if value is not None else None
    return None


def _run_pca_sensitivity(
    features: np.ndarray,
    metadata: pd.DataFrame,
    scanners: list[str],
    stains: list[str],
    config: dict,
    tables: Path,
) -> dict:
    dimensions = min(
        config["mmd"]["pca_dimensions"], features.shape[1], len(features) - 1
    )
    reduced = PCA(n_components=dimensions, random_state=config["seed"]).fit_transform(
        features
    )
    outputs = {}
    for name, domains in (
        ("scanner_marginal", scanners),
        ("scanner_controlled", scanners),
        ("stain_marginal", stains),
        ("stain_balanced", stains),
    ):
        details = _evaluate_pairs(reduced, metadata, domains, name, config)
        matrix = matrix_from_pair_records(details, domains, "reported_mmd2")
        path = tables / f"{name}_pca{dimensions}.csv"
        _write_matrix(matrix, path)
        details.to_csv(tables / f"{name}_pca{dimensions}_details.csv", index=False)
        outputs[name] = str(path)
    return {"dimensions": dimensions, "tables": outputs}


def _write_report(
    path: Path,
    config: dict,
    feature_path: Path,
    feature_metadata: dict,
    scanner_summary: dict,
    controlled_summary: dict,
    stain_summary: dict,
    balanced_summary: dict,
    scanner_largest: dict,
    scanner_smallest: dict,
    stain_largest: dict,
    stain_smallest: dict,
    scanner_order: tuple[float, float],
    stain_order: tuple[float, float],
    centroid_result: tuple[float, float],
    probe_accuracy: float | None,
    controlled: pd.DataFrame,
    stain_marginal: pd.DataFrame,
    stain_balanced: pd.DataFrame,
    centroid_source: str,
    pca_sensitivity: dict | None,
) -> None:
    significant_bootstrap = int((controlled.ci_lower > 0).sum())
    delta = (
        stain_balanced.reported_mmd2.to_numpy()
        - stain_marginal.reported_mmd2.to_numpy()
    )
    reduced_pairs = int((delta < 0).sum())
    probe_text = (
        f"The existing group-held-out scanner probe BA is {probe_accuracy:.3f}."
        if probe_accuracy is not None
        else "The prior scanner probe metric was not present in the cached audit metrics."
    )
    permutation_text = "Permutation testing was not requested."
    if "permutation_p_value_fdr" in controlled:
        discoveries = int((controlled.permutation_p_value_fdr < 0.05).sum())
        permutation_text = (
            f"Optional controlled-scanner permutation testing found {discoveries}/{len(controlled)} "
            "pairs below FDR-adjusted p<0.05."
        )
    pca_text = (
        f"PCA sensitivity was enabled at {pca_sensitivity['dimensions']} dimensions; separate tables are listed in metrics.json."
        if pca_sensitivity
        else "PCA sensitivity was not enabled; set RUN_MMD_PCA=1 to generate separate PCA tables."
    )
    report = f"""# PLISM Frozen-DINOv3 MMD Domain Analysis

## Method

This analysis reuses `{feature_path}` without feature extraction. The primary representation is the original {feature_metadata.get("feature_dimension", "unknown")}-dimensional frozen DINOv3 embedding after L2 normalization; UMAP coordinates are never used.

For each domain pair, sample sizes are equalized and capped at {config["mmd"]["max_samples_per_domain"]}. The pair-specific Gaussian bandwidth is the square root of the median non-zero squared distance from a deterministic pooled subsample. The primary kernel averages RBF kernels at σ/2, σ, and 2σ. Reported values are `max(unbiased MMD², 0)`; signed unbiased estimates and the single-σ secondary estimate remain in the detailed CSV files.

Controlled scanner confidence intervals use {config["mmd"]["fast_dev_bootstrap_replicates"] if config["fast_dev_run"] else config["mmd"]["bootstrap_replicates"]} aligned-group cluster-bootstrap replicates. Balanced stain intervals bootstrap independently within exact `(tissue_type, scanner_id)` strata. Bandwidth is fixed at the pairwise median estimate during bootstrap. {permutation_text} {pca_text}

## Scanner

- Largest controlled scanner gap: **{scanner_largest["domain_a"]} ↔ {scanner_largest["domain_b"]}**, MMD²={scanner_largest["mmd2"]:.6f}.
- Smallest controlled scanner gap: **{scanner_smallest["domain_a"]} ↔ {scanner_smallest["domain_b"]}**, MMD²={scanner_smallest["mmd2"]:.6f}.
- Marginal scanner distribution: mean={scanner_summary["mean"]:.6f}, median={scanner_summary["median"]:.6f}, IQR={scanner_summary["iqr"]:.6f}.
- Controlled scanner distribution: mean={controlled_summary["mean"]:.6f}, median={controlled_summary["median"]:.6f}, IQR={controlled_summary["iqr"]:.6f}.
- Marginal-vs-controlled pair ordering: Spearman ρ={scanner_order[0]:.3f}, p={scanner_order[1]:.3g}. This directly quantifies whether ordering remains similar after matching aligned morphology and stain.
- {significant_bootstrap}/{len(controlled)} controlled pairs have a clipped bootstrap 95% lower bound greater than zero. This is a bootstrap interval statement, not a formal hypothesis test; use `RUN_MMD_PERMUTATION=1` for the optional FDR-controlled permutation analysis.

The controlled matrix is the scientifically stronger scanner result because selection fixes `aligned_group_id` and `stain_id` before comparing distributions. The marginal matrix can still reflect composition differences.

## Relation to centroid distance and scanner decoding

Scanner centroid cosine distance and controlled MMD have Spearman ρ={centroid_result[0]:.3f} (p={centroid_result[1]:.3g}); centroid source: `{centroid_source}`. Centroid distance measures mean shift, whereas MMD detects broader distribution differences. A high MMD with a small centroid distance suggests that scanner shift changes distribution shape or dispersion rather than only the global feature centroid.

{probe_text} The linear probe asks whether scanner identity can be decoded, centroid distance asks how far scanner means move, and MMD asks whether the full feature distributions differ. They are complementary rather than redundant, and a single global probe accuracy cannot be equated with every pairwise MMD value.

## Stain

- Largest tissue/scanner-balanced stain gap: **{stain_largest["domain_a"]} ↔ {stain_largest["domain_b"]}**, MMD²={stain_largest["mmd2"]:.6f}.
- Smallest balanced stain gap: **{stain_smallest["domain_a"]} ↔ {stain_smallest["domain_b"]}**, MMD²={stain_smallest["mmd2"]:.6f}.
- Marginal stain distribution: mean={stain_summary["mean"]:.6f}, median={stain_summary["median"]:.6f}, IQR={stain_summary["iqr"]:.6f}.
- Balanced stain distribution: mean={balanced_summary["mean"]:.6f}, median={balanced_summary["median"]:.6f}, IQR={balanced_summary["iqr"]:.6f}.
- Balancing changed pairwise MMD by mean {delta.mean():+.6f} and median {np.median(delta):+.6f}; {reduced_pairs}/{len(delta)} pairs decreased.
- Marginal-vs-balanced ordering: Spearman ρ={stain_order[0]:.3f}, p={stain_order[1]:.3g}.

Exact balancing reduces tissue/scanner composition confounding, but PLISM stain conditions may use serial sections. Serial-section morphology remains a limitation, so balanced stain MMD is not a perfectly isolated causal stain effect.

## Overall interpretation

The numeric scanner and stain summaries above are reported directly; no universal LOW/MODERATE/HIGH thresholds are assigned because kernel MMD scale depends on representation, bandwidth, and the domain sets. The scanner-vs-stain violin plot is descriptive only: the number and nature of scanner and stain domains differ.

Primary scientific outputs:

- `figures/scanner_mmd_controlled_heatmap.png`
- `figures/stain_mmd_balanced_heatmap.png`
- `figures/MMD_SUMMARY.png`
- `tables/scanner_mmd_controlled.csv`
- `tables/stain_mmd_balanced.csv`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m_minus1_mmd.yaml")
    args = parser.parse_args()
    started = time.monotonic()
    config = _load_config(args.config)
    out = _prepare_output(config)
    tables, figures = out / "tables", out / "figures"
    source, feature_path, metadata_path = _locate_source(config)
    print(f"Detected cached PLISM DINOv3 features: {feature_path}")
    print(f"Detected PLISM sample metadata: {metadata_path}")

    raw_features, sample_ids, feature_metadata = load_feature_cache(feature_path)
    metadata = _align_metadata(pd.read_parquet(metadata_path), sample_ids)
    features = l2_normalize(raw_features)
    scanners = sorted(metadata.scanner_id.astype(str).unique())
    stains = sorted(metadata.stain_id.astype(str).unique())
    if len(scanners) < 2 or len(stains) < 2:
        raise RuntimeError(
            f"MMD requires at least two scanner and stain domains; found scanners={scanners}, stains={stains}"
        )

    scanner_marginal = _evaluate_pairs(
        features, metadata, scanners, "scanner_marginal", config
    )
    scanner_controlled = _evaluate_pairs(
        features,
        metadata,
        scanners,
        "scanner_controlled",
        config,
        bootstrap=True,
        permutation=config["run_mmd_permutation"],
    )
    stain_marginal = _evaluate_pairs(
        features, metadata, stains, "stain_marginal", config
    )
    stain_balanced = _evaluate_pairs(
        features,
        metadata,
        stains,
        "stain_balanced",
        config,
        bootstrap=True,
    )

    if config["fast_dev_run"]:
        for frame in (scanner_controlled, stain_balanced):
            frame["fast_dev_run"] = True

    matrices = {
        "scanner_mmd": matrix_from_pair_records(
            scanner_marginal, scanners, "reported_mmd2"
        ),
        "scanner_mmd_controlled": matrix_from_pair_records(
            scanner_controlled, scanners, "reported_mmd2"
        ),
        "stain_mmd": matrix_from_pair_records(stain_marginal, stains, "reported_mmd2"),
        "stain_mmd_balanced": matrix_from_pair_records(
            stain_balanced, stains, "reported_mmd2"
        ),
    }
    for name, matrix in matrices.items():
        _write_matrix(matrix, tables / f"{name}.csv")
    _write_matrix(
        matrix_from_pair_records(scanner_controlled, scanners, "ci_lower"),
        tables / "scanner_mmd_ci_lower.csv",
    )
    _write_matrix(
        matrix_from_pair_records(scanner_controlled, scanners, "ci_upper"),
        tables / "scanner_mmd_ci_upper.csv",
    )
    _write_matrix(
        matrix_from_pair_records(stain_balanced, stains, "ci_lower"),
        tables / "stain_mmd_ci_lower.csv",
    )
    _write_matrix(
        matrix_from_pair_records(stain_balanced, stains, "ci_upper"),
        tables / "stain_mmd_ci_upper.csv",
    )
    _write_matrix(
        matrix_from_pair_records(scanner_controlled, scanners, "bootstrap_mean_mmd2"),
        tables / "scanner_mmd_bootstrap_mean.csv",
    )
    _write_matrix(
        matrix_from_pair_records(stain_balanced, stains, "bootstrap_mean_mmd2"),
        tables / "stain_mmd_bootstrap_mean.csv",
    )
    for name, frame in (
        ("scanner_mmd_details", scanner_marginal),
        ("scanner_mmd_controlled_details", scanner_controlled),
        ("stain_mmd_details", stain_marginal),
        ("stain_mmd_balanced_details", stain_balanced),
    ):
        frame.to_csv(tables / f"{name}.csv", index=False)
    if config["run_mmd_permutation"]:
        scanner_controlled[
            [
                "domain_a",
                "domain_b",
                "observed_raw_mmd2",
                "permutation_p_value",
                "permutation_p_value_fdr",
                "permutations",
            ]
        ].to_csv(tables / "scanner_mmd_permutation.csv", index=False)

    scanner_vmax = max(
        matrices["scanner_mmd"].to_numpy().max(),
        matrices["scanner_mmd_controlled"].to_numpy().max(),
    )
    stain_vmax = max(
        matrices["stain_mmd"].to_numpy().max(),
        matrices["stain_mmd_balanced"].to_numpy().max(),
    )
    plot_mmd_heatmap(
        matrices["scanner_mmd"],
        figures / "scanner_mmd_heatmap.png",
        "PLISM scanner MMD² — marginal",
        scanner_vmax,
    )
    plot_mmd_heatmap(
        matrices["scanner_mmd_controlled"],
        figures / "scanner_mmd_controlled_heatmap.png",
        "PLISM scanner MMD² — controlled by aligned morphology + stain",
        scanner_vmax,
    )
    plot_mmd_heatmap(
        matrices["stain_mmd"],
        figures / "stain_mmd_heatmap.png",
        "PLISM stain MMD² — marginal",
        stain_vmax,
    )
    plot_mmd_heatmap(
        matrices["stain_mmd_balanced"],
        figures / "stain_mmd_balanced_heatmap.png",
        "PLISM stain MMD² — tissue/scanner balanced",
        stain_vmax,
    )
    distribution_values = pd.concat(
        [
            _off_diagonal_values(matrices["scanner_mmd"], "scanner marginal"),
            _off_diagonal_values(
                matrices["scanner_mmd_controlled"], "scanner controlled"
            ),
            _off_diagonal_values(matrices["stain_mmd"], "stain marginal"),
            _off_diagonal_values(matrices["stain_mmd_balanced"], "stain balanced"),
        ],
        ignore_index=True,
    )
    distribution_values.to_csv(tables / "domain_gap_distributions.csv", index=False)
    plot_gap_distributions(distribution_values, figures / "scanner_vs_stain_mmd.png")

    centroid_matrix, centroid_source = _load_centroid_matrix(
        source, features, metadata, scanners
    )
    centroid_pairs, centroid_rho, centroid_p = _centroid_pair_table(
        scanner_controlled, centroid_matrix
    )
    centroid_pairs.to_csv(tables / "scanner_centroid_vs_mmd.csv", index=False)
    plot_centroid_vs_mmd(
        centroid_pairs,
        centroid_rho,
        figures / "scanner_centroid_vs_mmd.png",
    )

    summaries = {
        "scanner_marginal": off_diagonal_summary(matrices["scanner_mmd"]),
        "scanner_controlled": off_diagonal_summary(matrices["scanner_mmd_controlled"]),
        "stain_marginal": off_diagonal_summary(matrices["stain_mmd"]),
        "stain_balanced": off_diagonal_summary(matrices["stain_mmd_balanced"]),
    }
    scanner_largest = _pair_extreme(scanner_controlled, True)
    scanner_smallest = _pair_extreme(scanner_controlled, False)
    stain_largest = _pair_extreme(stain_balanced, True)
    stain_smallest = _pair_extreme(stain_balanced, False)
    scanner_order = _pair_order_correlation(scanner_marginal, scanner_controlled)
    stain_order = _pair_order_correlation(stain_marginal, stain_balanced)
    probe_accuracy = _scanner_probe_accuracy(source)
    pca_sensitivity = None
    if config["run_pca_sensitivity"]:
        pca_sensitivity = _run_pca_sensitivity(
            features, metadata, scanners, stains, config, tables
        )

    footer = (
        f"Largest controlled scanner gap: {scanner_largest['domain_a']} ↔ {scanner_largest['domain_b']} "
        f"(MMD²={scanner_largest['mmd2']:.6f}) | Smallest: {scanner_smallest['domain_a']} ↔ "
        f"{scanner_smallest['domain_b']} ({scanner_smallest['mmd2']:.6f})\n"
        f"Largest balanced stain gap: {stain_largest['domain_a']} ↔ {stain_largest['domain_b']} "
        f"(MMD²={stain_largest['mmd2']:.6f})"
    )
    compose_dashboard(
        [
            ("Scanner marginal MMD²", figures / "scanner_mmd_heatmap.png"),
            (
                "Scanner controlled MMD² — primary",
                figures / "scanner_mmd_controlled_heatmap.png",
            ),
            ("Stain marginal MMD²", figures / "stain_mmd_heatmap.png"),
            (
                "Stain tissue/scanner-balanced MMD² — primary",
                figures / "stain_mmd_balanced_heatmap.png",
            ),
            (
                "Scanner vs stain pair distributions",
                figures / "scanner_vs_stain_mmd.png",
            ),
            (
                "Centroid distance vs controlled MMD²",
                figures / "scanner_centroid_vs_mmd.png",
            ),
        ],
        figures / "MMD_SUMMARY.png",
        "PLISM FROZEN-DINOv3 MMD DOMAIN ANALYSIS",
        footer,
    )

    metrics = {
        "milestone": config["milestone"],
        "audit_valid": not config["fast_dev_run"],
        "fast_dev_run": config["fast_dev_run"],
        "feature_source": str(feature_path),
        "metadata_source": str(metadata_path),
        "feature_metadata": feature_metadata,
        "representation": "L2-normalized original frozen DINOv3 embedding",
        "kernel": {
            "type": "pairwise median-bandwidth multi-scale Gaussian RBF",
            "scales": config["mmd"]["kernel_scales"],
            "estimator": "unbiased MMD^2",
            "reported_transform": "max(raw_mmd2, 0)",
        },
        "domains": {"scanners": scanners, "stains": stains},
        "summaries": summaries,
        "extrema": {
            "controlled_scanner_largest": scanner_largest,
            "controlled_scanner_smallest": scanner_smallest,
            "balanced_stain_largest": stain_largest,
            "balanced_stain_smallest": stain_smallest,
        },
        "correlations": {
            "scanner_marginal_vs_controlled_spearman": {
                "rho": scanner_order[0],
                "p_value": scanner_order[1],
            },
            "stain_marginal_vs_balanced_spearman": {
                "rho": stain_order[0],
                "p_value": stain_order[1],
            },
            "scanner_centroid_vs_controlled_mmd_spearman": {
                "rho": centroid_rho,
                "p_value": centroid_p,
                "centroid_source": centroid_source,
            },
        },
        "existing_scanner_probe_balanced_accuracy": probe_accuracy,
        "ci_matrix_semantics": {
            "scanner": "controlled scanner aligned-group cluster bootstrap",
            "stain": "tissue/scanner-balanced within-stratum bootstrap",
        },
        "permutation_enabled": config["run_mmd_permutation"],
        "pca_sensitivity": pca_sensitivity,
        "elapsed_seconds": time.monotonic() - started,
        "provenance": collect_provenance(
            config["seed"], {"plism_features": len(features)}
        ),
        "caveats": [
            "Controlled scanner MMD fixes aligned_group_id and stain_id and is stronger than marginal scanner MMD.",
            "Balanced stain MMD matches tissue_type and scanner_id, but serial-section morphology remains confounded.",
            "Scanner and stain pair distributions involve different domain sets and are not directly causal rankings.",
        ],
    }
    atomic_json_dump(metrics, out / "metrics.json")
    _write_report(
        out / "REPORT.md",
        config,
        feature_path,
        feature_metadata,
        summaries["scanner_marginal"],
        summaries["scanner_controlled"],
        summaries["stain_marginal"],
        summaries["stain_balanced"],
        scanner_largest,
        scanner_smallest,
        stain_largest,
        stain_smallest,
        scanner_order,
        stain_order,
        (centroid_rho, centroid_p),
        probe_accuracy,
        scanner_controlled,
        stain_marginal,
        stain_balanced,
        centroid_source,
        pca_sensitivity,
    )

    print("=" * 52)
    print("PLISM DINOv3 MMD Domain Analysis Complete")
    print("=" * 52)
    print(f"\nScanner domains: {len(scanners)}\nStain domains: {len(stains)}")
    print(
        f"\nLargest controlled scanner MMD:\n  {scanner_largest['domain_a']} vs "
        f"{scanner_largest['domain_b']} : {scanner_largest['mmd2']:.6f}"
    )
    print(
        f"\nSmallest controlled scanner MMD:\n  {scanner_smallest['domain_a']} vs "
        f"{scanner_smallest['domain_b']} : {scanner_smallest['mmd2']:.6f}"
    )
    print(
        f"\nLargest balanced stain MMD:\n  {stain_largest['domain_a']} vs "
        f"{stain_largest['domain_b']} : {stain_largest['mmd2']:.6f}"
    )
    print(
        f"\nOutputs:\n  {figures / 'MMD_SUMMARY.png'}\n"
        f"  {tables / 'scanner_mmd_controlled.csv'}\n"
        f"  {tables / 'stain_mmd_balanced.csv'}\n  {out / 'REPORT.md'}"
    )
    print("=" * 52)


if __name__ == "__main__":
    main()
