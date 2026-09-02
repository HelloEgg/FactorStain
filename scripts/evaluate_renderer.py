#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from factorstain.evaluation.mmd import compute_mmd2, l2_normalize, stable_seed
from factorstain.evaluation.renderer import (
    RealAcquisitionProbes,
    build_evaluation_pairs,
    evaluate_model,
)
from factorstain.metrics.statistics import paired_bootstrap, summarize_samples
from factorstain.models.dinov3 import load_frozen_dinov3_encoder
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output, update_master
from factorstain.utils.runtime import atomic_json_dump, collect_provenance

SAMPLE_PRIMARY = {
    "stain_target_accuracy": 1,
    "scanner_target_accuracy": 1,
    "morphology_preservation": 1,
    "factor_isolation_score": 1,
}
PRIMARY = {"dino_mmd2": -1, **SAMPLE_PRIMARY}
TRACK_S = {
    "lpips": -1,
    "ssim": 1,
    "scanner_target_accuracy": 1,
    "morphology_preservation": 1,
}


def _cell_set(items: list[dict]) -> set[tuple[str, str]]:
    return {(str(item["stain_id"]), str(item["scanner_id"])) for item in items}


def _locate_dino_cache(config: dict) -> Path:
    source = Path(config["paths"]["outputs_root"]) / "m_minus1_domain_audit"
    direct = source / "features" / "plism_dinov3.npz"
    if direct.exists():
        return direct
    matches = sorted(source.rglob("plism_dinov3.npz")) if source.exists() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        "M1 reuses the M-1 DINOv3 cache for real-only acquisition classifiers. "
        f"Expected {direct}."
    )


def _tensor_image(axis, tensor: torch.Tensor, title: str) -> None:
    axis.imshow(tensor.permute(1, 2, 0).numpy().clip(0, 1))
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _representative(previews: dict[str, list[dict]], protocol: str) -> dict[str, dict]:
    selected = {}
    for method, values in previews.items():
        match = next((value for value in values if value["protocol"] == protocol), None)
        if match is not None:
            selected[method] = match
    return selected


def _comparison_grid(
    previews: dict[str, list[dict]],
    protocol: str,
    destination: Path,
    title: str,
) -> None:
    selected = _representative(previews, protocol)
    if not selected:
        return
    methods = [
        name
        for name in ("joint", "parallel", "factorstain", "reverse")
        if name in selected
    ]
    first = selected[methods[0]]
    fig, axes = plt.subplots(1, len(methods) + 2, figsize=(3 * (len(methods) + 2), 3.4))
    _tensor_image(axes[0], first["source"], "Source")
    for position, method in enumerate(methods, start=1):
        _tensor_image(axes[position], selected[method]["generated"], method.title())
    _tensor_image(axes[-1], first["target"], "Real target")
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _factorial_figure(previews: dict[str, list[dict]], destination: Path) -> None:
    selected = _representative(previews, "unseen_combo_seen_morphology")
    if not selected:
        selected = _representative(previews, "unseen_combo_unseen_morphology")
    if "factorstain" not in selected:
        return
    ordered = selected["factorstain"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    _tensor_image(axes[0, 0], ordered["source"], "REAL A × X")
    _tensor_image(axes[0, 1], ordered["scanner_cf"], "FactorStain A × Y")
    _tensor_image(axes[1, 0], ordered["stain_cf"], "FactorStain B × X")
    _tensor_image(axes[1, 1], ordered["generated"], "FactorStain B × Y\nHELD OUT")
    for column, method in enumerate(("joint", "parallel"), start=2):
        if method in selected:
            _tensor_image(
                axes[0, column],
                selected[method]["generated"],
                f"{method.title()} B × Y",
            )
        else:
            axes[0, column].axis("off")
    _tensor_image(axes[1, 2], ordered["generated"], "FactorStain B × Y")
    _tensor_image(axes[1, 3], ordered["target"], "REAL B × Y")
    fig.suptitle(
        "Compositional acquisition completion: stain first, scanner second",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _metric_figures(metrics: pd.DataFrame, mmd: pd.DataFrame, figures: Path) -> None:
    plot_metrics = [
        "stain_target_accuracy",
        "scanner_target_accuracy",
        "morphology_preservation",
        "factor_isolation_score",
    ]
    protocols = [
        value
        for value in (
            "seen_combo_unseen_morphology",
            "unseen_combo_seen_morphology",
            "unseen_combo_unseen_morphology",
        )
        if value in metrics.protocol.unique()
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    for axis, metric in zip(axes, plot_metrics):
        values = metrics.groupby(["method", "protocol"])[metric].mean().unstack()
        values.reindex(columns=protocols).plot.bar(ax=axis)
        axis.set_title(metric.replace("_", " ") + " ↑")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=25)
        if axis is not axes[0] and axis.get_legend():
            axis.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)))
    fig.suptitle(
        "Seen versus unseen combination performance", fontsize=16, weight="bold"
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.96])
    fig.savefig(
        figures / "seen_vs_unseen_performance.png", dpi=190, bbox_inches="tight"
    )
    plt.close(fig)

    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")]
    aggregate = unseen.groupby("method")[plot_metrics].mean()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for axis, metric in zip(axes, plot_metrics):
        axis.bar(aggregate.index, aggregate[metric], color="#4263eb")
        axis.set_title(metric.replace("_", " ") + " ↑")
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle(
        "Model comparison on held-out stain × scanner cells", fontsize=16, weight="bold"
    )
    fig.tight_layout()
    fig.savefig(figures / "model_comparison_metrics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    mmd_valid = mmd.dropna(subset=["dino_mmd2"])
    if len(mmd_valid):
        values = mmd_valid.pivot_table(
            index="combination_id", columns="method", values="dino_mmd2"
        )
        axis = values.plot.bar(
            figsize=(max(10, len(values) * 0.75), 5.8),
            color=["#495057", "#4263eb", "#2b8a3e", "#f08c00"][: len(values.columns)],
        )
        fig = axis.figure
        axis.tick_params(axis="x", rotation=45)
    else:
        fig, axis = plt.subplots(figsize=(9, 5.5))
        axis.text(0.5, 0.5, "Insufficient per-cell FAST samples for MMD", ha="center")
    axis.set_ylabel("DINOv3 MMD² ↓")
    axis.set_title("Generated versus real held-out-domain distributions")
    fig.tight_layout()
    fig.savefig(figures / "unseen_combo_mmd.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    for metric, filename, title in (
        ("factor_isolation_score", "factor_isolation.png", "Factor Isolation Score ↑"),
        (
            "morphology_preservation",
            "morphology_preservation.png",
            "Morphology preservation ↑",
        ),
    ):
        values = unseen.groupby("method")[metric].agg(["mean", "sem"])
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            values.index,
            values["mean"],
            yerr=1.96 * values["sem"].fillna(0),
            color="#2b8a3e",
            capsize=4,
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=190, bbox_inches="tight")
        plt.close(fig)


def _factor_isolation_examples(
    previews: dict[str, list[dict]], metrics: pd.DataFrame, destination: Path
) -> None:
    selected = _representative(previews, "unseen_combo_seen_morphology")
    if "factorstain" not in selected:
        selected = _representative(previews, "unseen_combo_unseen_morphology")
    if "factorstain" not in selected:
        return
    preview = selected["factorstain"]
    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")]
    values = unseen.groupby("method").factor_isolation_score.mean()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    _tensor_image(axes[0], preview["source"], "Source A × X")
    _tensor_image(axes[1], preview["stain_cf"], "Stain-only B × X")
    _tensor_image(axes[2], preview["scanner_cf"], "Scanner-only A × Y")
    axes[3].bar(values.index, values, color="#2b8a3e")
    axes[3].set_title("Factor Isolation Score ↑")
    axes[3].tick_params(axis="x", rotation=25)
    fig.suptitle(
        "Desired factor changes; non-target factor and morphology should remain",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _failure_grid(previews: list[dict], scores: list[float], destination: Path) -> None:
    if not previews:
        return
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), squeeze=False)
    labels = ("Best", "Median", "Worst")
    for position in range(15):
        preview = previews[min(position, len(previews) - 1)]
        score = scores[min(position, len(scores) - 1)]
        row, column = divmod(position, 5)
        generated = preview["generated"].permute(1, 2, 0).numpy().clip(0, 1)
        target = preview["target"].permute(1, 2, 0).numpy().clip(0, 1)
        axes[row, column].imshow(np.concatenate([generated, target], axis=1))
        axes[row, column].set_title(
            f"{labels[row]}\nscore={score:.3f}\ngenerated | real", fontsize=8
        )
        axes[row, column].axis("off")
    fig.suptitle(
        "FactorStain best / median / worst audit (automatic selection)",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _compute_mmd(feature_bundles: dict[str, dict], seed: int) -> pd.DataFrame:
    rows = []
    for method, bundle in feature_bundles.items():
        metadata = bundle["metadata"]
        unseen = metadata.protocol.str.startswith("unseen_combo")
        for (stain, scanner), positions in (
            metadata[unseen]
            .groupby(["target_stain_id", "target_scanner_id"])
            .groups.items()
        ):
            indices = np.asarray(list(positions), dtype=int)
            row = {
                "method": method,
                "target_stain_id": stain,
                "target_scanner_id": scanner,
                "combination_id": f"{stain} × {scanner}",
                "n": len(indices),
            }
            if len(indices) >= 2:
                result = compute_mmd2(
                    l2_normalize(bundle["generated"][indices]),
                    l2_normalize(bundle["real"][indices]),
                    seed=stable_seed(seed, method, stain, scanner),
                )
                row.update(
                    {
                        "dino_mmd2": result["reported_mmd2"],
                        "raw_mmd2": result["raw_mmd2"],
                        "sigma": result["sigma"],
                    }
                )
            else:
                row.update({"dino_mmd2": np.nan, "raw_mmd2": np.nan, "sigma": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_comparisons(
    metrics: pd.DataFrame,
    mmd: pd.DataFrame,
    strongest: dict[str, str],
    replicates: int,
    seed: int,
) -> list[dict]:
    results = []
    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")].copy()
    for metric, direction in SAMPLE_PRIMARY.items():
        frame = unseen.copy()
        frame["score"] = direction * frame[metric]
        result = paired_bootstrap(
            frame,
            "score",
            "factorstain",
            strongest[metric],
            n_bootstrap=replicates,
            seed=stable_seed(seed, metric),
        )
        result.update(
            {
                "source_metric": metric,
                "direction": direction,
                "statistical_unit": "aligned_group_id",
            }
        )
        results.append(result)
    mmd_frame = mmd.dropna(subset=["dino_mmd2"]).copy()
    if len(mmd_frame):
        mmd_frame["aligned_group_id"] = mmd_frame.combination_id
        mmd_frame["score"] = -mmd_frame.dino_mmd2
        try:
            result = paired_bootstrap(
                mmd_frame,
                "score",
                "factorstain",
                strongest["dino_mmd2"],
                n_bootstrap=replicates,
                seed=stable_seed(seed, "dino_mmd2"),
            )
            result.update(
                {
                    "source_metric": "dino_mmd2",
                    "direction": -1,
                    "statistical_unit": "heldout_combination",
                }
            )
            results.append(result)
        except ValueError:
            pass
    return results


def _decide(
    metrics: pd.DataFrame,
    mmd: pd.DataFrame,
    config: dict,
) -> tuple[dict, list[dict], dict[str, dict]]:
    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")]
    sample_means = unseen.groupby("method")[list(SAMPLE_PRIMARY)].mean()
    mmd_means = mmd.groupby("method").dino_mmd2.mean()
    means = {
        method: {
            **sample_means.loc[method].to_dict(),
            "dino_mmd2": float(mmd_means.get(method, np.nan)),
        }
        for method in sample_means.index
    }
    baselines = [name for name in ("joint", "parallel") if name in means]
    if "factorstain" not in means or not baselines:
        decision = {
            "status": "NO_GO",
            "decision_valid": not config["fast_dev_run"],
            "reasons": ["The required three-model comparison is incomplete."],
        }
        return decision, [], means
    strongest = {}
    for metric, direction in PRIMARY.items():
        finite = [name for name in baselines if np.isfinite(means[name][metric])]
        strongest[metric] = (
            max(finite, key=lambda name: direction * means[name][metric])
            if finite
            else baselines[0]
        )
    replicates = (
        config.get("fast_dev_bootstrap_samples", 50)
        if config["fast_dev_run"]
        else max(1000, config.get("bootstrap_samples", 1000))
    )
    try:
        bootstraps = _bootstrap_comparisons(
            metrics, mmd, strongest, replicates, config["seed"]
        )
    except ValueError:
        bootstraps = []
    wins, relative_wins = [], []
    for metric, direction in PRIMARY.items():
        factor_value = means["factorstain"][metric]
        baseline_value = means[strongest[metric]][metric]
        if not (np.isfinite(factor_value) and np.isfinite(baseline_value)):
            continue
        improvement = direction * (factor_value - baseline_value)
        relative = improvement / max(abs(baseline_value), 0.05)
        if improvement > 0:
            wins.append(metric)
        if relative >= config["decision"]["relative_improvement"]:
            relative_wins.append(metric)
    consistent = [row["source_metric"] for row in bootstraps if row["ci_low"] > 0]
    seen = metrics[metrics.protocol.eq("seen_combo_unseen_morphology")]
    unseen_specific = []
    if len(seen):
        seen_means = seen.groupby("method")[list(SAMPLE_PRIMARY)].mean()
        for metric, direction in SAMPLE_PRIMARY.items():
            baseline = strongest[metric]
            if baseline in seen_means.index:
                unseen_advantage = direction * (
                    means["factorstain"][metric] - means[baseline][metric]
                )
                seen_advantage = direction * (
                    seen_means.loc["factorstain", metric]
                    - seen_means.loc[baseline, metric]
                )
                if unseen_advantage > seen_advantage:
                    unseen_specific.append(metric)
    track = metrics[metrics.protocol.eq("controlled_scanner_transfer")]
    track_means = (
        track.groupby("method")[list(TRACK_S)].mean() if len(track) else pd.DataFrame()
    )
    track_wins = []
    if "factorstain" in track_means.index:
        for metric, direction in TRACK_S.items():
            candidates = [name for name in baselines if name in track_means.index]
            if candidates:
                baseline = max(
                    candidates,
                    key=lambda name: direction * track_means.loc[name, metric],
                )
                if (
                    direction
                    * (
                        track_means.loc["factorstain", metric]
                        - track_means.loc[baseline, metric]
                    )
                    > 0
                ):
                    track_wins.append(metric)
    full_go = (
        len(wins) >= config["decision"]["win_metrics"]
        and len(relative_wins) >= config["decision"]["relative_improvement_metrics"]
        and len(consistent) >= config["decision"]["consistent_metrics"]
        and len(unseen_specific) >= 2
    )
    parallel_close = False
    if "parallel" in means:
        substantive = 0
        for metric, direction in PRIMARY.items():
            factor_value, parallel_value = (
                means["factorstain"][metric],
                means["parallel"][metric],
            )
            if np.isfinite(factor_value) and np.isfinite(parallel_value):
                relative = (
                    direction
                    * (factor_value - parallel_value)
                    / max(abs(parallel_value), 0.05)
                )
                substantive += relative >= config["decision"]["relative_improvement"]
        parallel_close = substantive < 2
    if full_go and not parallel_close:
        status = "STRONG_GO" if len(wins) == 5 and len(consistent) >= 3 else "GO"
    elif (
        len(track_wins) == len(TRACK_S)
        or parallel_close
        or "stain_target_accuracy" not in wins
    ):
        status = "GO_WITH_SCOPE_REDUCTION"
    else:
        status = "NO_GO"
    reasons = [
        f"FactorStain wins {len(wins)}/5 primary metrics: {wins}.",
        f"At least 5% relative improvement occurs on {len(relative_wins)} metrics: {relative_wins}.",
        f"Paired bootstrap lower CI exceeds zero on {len(consistent)} metrics: {consistent}.",
        f"Advantage is larger on unseen than seen combinations for {len(unseen_specific)} metrics: {unseen_specific}.",
        f"Track S wins are {len(track_wins)}/4: {track_wins}.",
        f"Parallel conditioning is effectively close to ordered FactorStain: {parallel_close}.",
    ]
    if config["fast_dev_run"]:
        status = "GO_WITH_SCOPE_REDUCTION"
        reasons.insert(
            0,
            "FAST_DEV_RUN validates plumbing only; no scientific decision is permitted.",
        )
    decision = {
        "status": status,
        "decision_valid": not config["fast_dev_run"],
        "primary_metric": "heldout_composition_composite",
        "observed": len(wins),
        "threshold": config["decision"]["win_metrics"],
        "wins": wins,
        "relative_improvement_metrics": relative_wins,
        "bootstrap_consistent_metrics": consistent,
        "unseen_specific_metrics": unseen_specific,
        "track_s_wins": track_wins,
        "strongest_baseline_by_metric": strongest,
        "reasons": reasons,
        "recommended_next_step": (
            "Proceed with full stain×scanner factorization."
            if status in {"STRONG_GO", "GO"}
            else "Prioritize the controlled scanner operator and treat cross-stain claims as inconclusive."
            if status == "GO_WITH_SCOPE_REDUCTION"
            else "Stop expansion and inspect held-out-combination failures."
        ),
    }
    return decision, bootstraps, means


def _decide_m2(
    metrics: pd.DataFrame, mmd: pd.DataFrame, config: dict
) -> tuple[dict, list[dict], dict[str, dict]]:
    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")]
    sample_means = unseen.groupby("method")[list(SAMPLE_PRIMARY)].mean()
    mmd_means = mmd.groupby("method").dino_mmd2.mean()
    means = {
        method: {
            **sample_means.loc[method].to_dict(),
            "dino_mmd2": float(mmd_means.get(method, np.nan)),
        }
        for method in sample_means.index
    }
    required = {"factorstain", "reverse", "parallel", "joint"}
    missing = required - set(means)
    bootstraps = []
    if missing:
        status, order_wins, factorized = "NO_GO", [], False
        reasons = [f"Required M2 models are missing: {sorted(missing)}."]
    else:
        order_wins = [
            metric
            for metric, direction in SAMPLE_PRIMARY.items()
            if all(
                direction * (means["factorstain"][metric] - means[other][metric]) > 0
                for other in ("reverse", "parallel")
            )
        ]
        joint_wins = {
            method: [
                metric
                for metric, direction in SAMPLE_PRIMARY.items()
                if direction * (means[method][metric] - means["joint"][metric]) > 0
            ]
            for method in ("factorstain", "reverse")
        }
        factorized = any(len(values) >= 3 for values in joint_wins.values())
        replicates = (
            50
            if config["fast_dev_run"]
            else max(1000, config.get("bootstrap_samples", 1000))
        )
        for other in ("reverse", "parallel", "joint"):
            for metric, direction in SAMPLE_PRIMARY.items():
                frame = unseen.copy()
                frame["score"] = direction * frame[metric]
                result = paired_bootstrap(
                    frame,
                    "score",
                    "factorstain",
                    other,
                    n_bootstrap=replicates,
                    seed=stable_seed(config["seed"], "m2", other, metric),
                )
                result.update(
                    {
                        "source_metric": metric,
                        "direction": direction,
                        "statistical_unit": "aligned_group_id",
                    }
                )
                bootstraps.append(result)
        status = (
            "NO_GO"
            if not factorized
            else "GO"
            if len(order_wins) >= config["decision"].get("win_metrics", 3)
            else "GO_WITH_SCOPE_REDUCTION"
        )
        reasons = [
            f"S→Q beats both Q→S and parallel on {len(order_wins)}/4 metrics: {order_wins}.",
            f"Factorized models show substantive gains over joint conditioning: {factorized}; wins={joint_wins}.",
        ]
    if config["fast_dev_run"]:
        status = "GO_WITH_SCOPE_REDUCTION"
        reasons.insert(0, "FAST_DEV_RUN is not a valid M2 scientific decision.")
    decision = {
        "status": status,
        "decision_valid": not config["fast_dev_run"],
        "primary_metric": "ordered_factorization_metric_wins",
        "observed": len(order_wins),
        "threshold": config["decision"].get("win_metrics", 3),
        "wins": order_wins,
        "relative_improvement_metrics": [],
        "bootstrap_consistent_metrics": [
            row["source_metric"] for row in bootstraps if row["ci_low"] > 0
        ],
        "unseen_specific_metrics": [],
        "track_s_wins": [],
        "strongest_baseline_by_metric": {},
        "reasons": reasons,
        "recommended_next_step": (
            "Retain physical S→Q ordering."
            if status == "GO"
            else "Reduce ordering claims or stop renderer expansion."
        ),
    }
    return decision, bootstraps, means


def _summary_dashboard(figures: Path, decision: dict, scores: dict[str, float]) -> None:
    sources = [
        ("heldout_combination_matrix.png", "Global split"),
        ("factorial_2x2_examples.png", "2×2 composition"),
        ("controlled_scanner_examples.png", "Track S"),
        ("seen_vs_unseen_performance.png", "Seen vs unseen"),
        ("unseen_combo_mmd.png", "DINOv3 MMD²"),
        ("factor_isolation.png", "Factor isolation"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    for axis, (filename, title) in zip(axes.flat, sources):
        axis.axis("off")
        path = figures / filename
        if path.exists():
            with Image.open(path) as image:
                axis.imshow(image.copy())
        axis.set_title(title, weight="bold")
    decision_axis = axes.flat[6]
    decision_axis.axis("off")
    color = {
        "STRONG_GO": "#087f5b",
        "GO": "#2b8a3e",
        "GO_WITH_SCOPE_REDUCTION": "#f08c00",
        "NO_GO": "#c92a2a",
    }[decision["status"]]
    decision_axis.text(
        0.5, 0.72, "M1 DECISION", ha="center", fontsize=16, weight="bold"
    )
    decision_axis.text(
        0.5,
        0.52,
        decision["status"],
        ha="center",
        fontsize=22,
        weight="bold",
        color=color,
    )
    decision_axis.text(
        0.5,
        0.25,
        f"valid={decision['decision_valid']}\nprimary wins={decision['observed']}/5",
        ha="center",
        fontsize=12,
    )
    score_axis = axes.flat[7]
    score_axis.axis("off")
    score_axis.text(0.05, 0.85, "Held-out composite scores", fontsize=15, weight="bold")
    for position, method in enumerate(("joint", "parallel", "factorstain")):
        score_axis.text(
            0.08,
            0.65 - position * 0.18,
            f"{method.upper()} unseen score: {scores.get(method, float('nan')):.4f}",
            fontsize=13,
        )
    fig.suptitle(
        "M1 FACTORIAL COMPOSITION PILOT — ONE-PAGE AUDIT", fontsize=22, weight="bold"
    )
    fig.tight_layout()
    fig.savefig(figures / "SUMMARY_DASHBOARD.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    out: Path,
    split: dict,
    episode_stats: dict,
    means: dict[str, dict],
    track_means: dict,
    mmd: pd.DataFrame,
    decision: dict,
    parameter_counts: dict,
) -> None:
    heldout = ", ".join(
        f"{row['stain_id']}×{row['scanner_id']}" for row in split["test_cells"]
    )
    report = f"""# M1 — Factorial Composition Pilot

## Experimental contract

- Held-out stain×scanner combinations: **{len(split["test_cells"])}** of {split["observed_cell_count"]} observed cells ({heldout}).
- Every held-out stain and scanner remains individually present in training: **yes**.
- Morphology split groups: {episode_stats["morphology_group_counts"]}; aligned_group_id leakage: **none**.
- Protocol A is an unseen combination on training-split morphology. Protocol B is an unseen combination on a completely unseen aligned_group_id.
- Models use the same split, optimization budget, 256×256 inputs, and evaluation pairs. Parameter counts: {parameter_counts}.

## Results answering the M1 questions

1. **How many combinations were held out?** {len(split["test_cells"])} test cells; {len(split["validation_cells"])} separate validation cells.
2. **Were factors individually seen?** Yes; the split constructor verifies every stain and scanner remains in `train_cells`.
3. **Seen-combination performance:** see `tables/overall_metrics.csv` and `figures/seen_vs_unseen_performance.png`.
4. **Unseen-combination performance:** {means}.
5. **Did the advantage increase on unseen cells?** Metrics satisfying this condition: {decision["unseen_specific_metrics"]}.
6. **Controlled Scanner Operator:** Track S means are {track_means}; its pairs share aligned_group_id and stain_id.
7. **Was morphology preserved?** Reported with source/generated DINOv3 cosine, edge similarity, grayscale structure, and their transparent composite.
8. **Were requested factors correct?** Independent stain/scanner logistic probes were fit only on cached real training DINOv3 features, never generated images.
9. **Did generated distributions approach real domains?** Per-held-out-cell DINOv3 MMD² is in `tables/per_combination_metrics.csv`; lower is better.
10. **Failure cases:** automatically ranked best/median/worst and worst-case panels are saved; no examples were hand selected.
11. **Final decision:** **{decision["status"]}** (valid={decision["decision_valid"]}).

## Factor Isolation Score

For each one-factor intervention, `desired_change = clip((P_after(target)-P_before(target)+1)/2, 0, 1)`, `other_preservation = 1 - total_variation(P_after(other), P_before(other))`, and `morphology = source/intervention DINOv3 cosine`. The factor-specific score is the arithmetic mean of these three terms; the reported Factor Isolation Score is the mean of stain-only and scanner-only scores. Higher is better and every term is visible in the per-sample table.

## PLISM supervision limitation

Same-stain scanner targets correspond to the same aligned stained tissue and receive strong L1, SSIM, LPIPS, DINOv3, and edge supervision. Stain-changing pairs may be serial sections: they receive target color/optical-density distribution losses plus source DINOv3, edge, and grayscale morphology losses, not pixel-perfect supervision. Cross-stain PSNR/SSIM are diagnostic only and never primary decision metrics.

## Decision rationale

{chr(10).join(f"- {reason}" for reason in decision["reasons"])}

{decision["status"]}
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")


def _write_m2_report(
    out: Path,
    means: dict[str, dict],
    decision: dict,
    parameter_counts: dict[str, int],
) -> None:
    text = f"""# M2 — Ordered Renderer Ablation

FactorStain S→Q, reverse Q→S, parallel, and joint renderers use the same global split and evaluation pairs.

- Unseen-combination means: {means}
- Parameter counts: {parameter_counts}
- Ordered wins: {decision["wins"]}
- Decision valid: {decision["decision_valid"]}

{chr(10).join(f"- {reason}" for reason in decision["reasons"])}

{decision["status"]}
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config = load_config(args.config)
    out = prepare_output(config)
    index_path = out / "metadata" / "plism_index_with_splits.parquet"
    index = pd.read_parquet(index_path)
    split = json.loads(
        (out / "splits" / f"combination_split_seed{config['seed']}.json").read_text(
            encoding="utf-8"
        )
    )
    episode_stats = json.loads(
        (out / "metadata" / "factorial_episode_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    limit = (
        config.get("fast_dev_eval_per_protocol", 8)
        if config["fast_dev_run"]
        else config.get("max_eval_per_protocol", 5000)
    )
    pairs = build_evaluation_pairs(index, split, config["seed"], limit)
    required_protocols = {
        "unseen_combo_seen_morphology",
        "unseen_combo_unseen_morphology",
        "controlled_scanner_transfer",
    }
    missing_protocols = (
        required_protocols - set(pairs.protocol) if len(pairs) else required_protocols
    )
    if missing_protocols:
        raise RuntimeError(
            f"PLISM cannot form required M1 evaluation protocols: {sorted(missing_protocols)}"
        )
    allow_cpu = config["fast_dev_run"] and os.getenv("ALLOW_CPU_FAST_DEV", "0") == "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(
            "M1 evaluation requires CUDA; CPU is only allowed for explicit FAST pipeline tests"
        )
    auxiliary = config.get(
        "auxiliary",
        {
            "dinov3_model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "dinov3_dtype": "bfloat16",
        },
    )
    dino = load_frozen_dinov3_encoder(
        auxiliary["dinov3_model_name"],
        device,
        auxiliary.get("dinov3_dtype", "bfloat16"),
    )
    probes = RealAcquisitionProbes(
        _locate_dino_cache(config), index, _cell_set(split["train_cells"])
    )
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    except Exception as exc:
        raise RuntimeError(
            f"LPIPS is mandatory for Track S and could not initialize: {exc}"
        ) from exc
    models = list(config["models"])
    if (
        out / "checkpoints" / "reverse" / "best.pt"
    ).exists() and "reverse" not in models:
        models.append("reverse")
    all_metrics, previews, feature_bundles, parameter_counts = [], {}, {}, {}
    feature_dir = out / "evaluation_features"
    feature_dir.mkdir(exist_ok=True)
    for model_name in models:
        checkpoint = out / "checkpoints" / model_name / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Required checkpoint is missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parameter_counts[model_name] = int(
            sum(value.numel() for value in payload["model"].values())
        )
        frame, model_previews, bundle = evaluate_model(
            model_name,
            checkpoint,
            config,
            index,
            pairs,
            dino,
            probes,
            lpips_model,
            device,
        )
        all_metrics.append(frame)
        previews[model_name] = model_previews
        feature_bundles[model_name] = bundle
        np.savez_compressed(
            feature_dir / f"{model_name}_dinov3.npz",
            generated=bundle["generated"],
            real=bundle["real"],
        )
    metrics = pd.concat(all_metrics, ignore_index=True)
    mmd = _compute_mmd(feature_bundles, config["seed"])
    if config["milestone"] == "m1_factorial":
        decision, bootstraps, means = _decide(metrics, mmd, config)
    else:
        decision, bootstraps, means = _decide_m2(metrics, mmd, config)
    atomic_json_dump(decision, out / "GO_NOGO.json")

    tables = out / "tables"
    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")]
    controlled = metrics[metrics.protocol.eq("controlled_scanner_transfer")]
    metrics.to_csv(tables / "per_sample_metrics.csv", index=False)
    unseen.to_csv(tables / "unseen_combo_metrics.csv", index=False)
    controlled.to_csv(tables / "controlled_scanner_metrics.csv", index=False)
    summaries = []
    reported_metrics = [
        "l1",
        "psnr",
        "ssim",
        "lpips",
        "dino_source_generated_cosine",
        "dino_target_generated_cosine",
        "edge_similarity",
        "color_delta_e",
        "sharpness_difference",
        "he_statistics_distance",
        *SAMPLE_PRIMARY,
    ]
    for (method, protocol), group in metrics.groupby(["method", "protocol"]):
        for metric in dict.fromkeys(reported_metrics):
            summaries.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "metric": metric,
                    **summarize_samples(group[metric]),
                }
            )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(tables / "overall_metrics.csv", index=False)
    per_combination = (
        unseen.groupby(["method", "target_stain_id", "target_scanner_id"])[
            list(SAMPLE_PRIMARY)
        ]
        .mean()
        .reset_index()
        .merge(mmd, on=["method", "target_stain_id", "target_scanner_id"], how="outer")
    )
    per_combination.to_csv(tables / "per_combination_metrics.csv", index=False)
    pd.DataFrame(bootstraps).to_csv(tables / "bootstrap_CI.csv", index=False)

    figures = out / "figures"
    _comparison_grid(
        previews,
        "unseen_combo_seen_morphology",
        figures / "generated_vs_real_unseen.png",
        "Unseen combination / seen morphology",
    )
    _comparison_grid(
        previews,
        "controlled_scanner_transfer",
        figures / "controlled_scanner_examples.png",
        "Track S — same stained tissue, target scanner transfer",
    )
    _factorial_figure(previews, figures / "factorial_2x2_examples.png")
    _metric_figures(metrics, mmd, figures)
    _factor_isolation_examples(previews, metrics, figures / "factor_isolation.png")
    factor_rows = unseen[unseen.method.eq("factorstain")].sort_values(
        "morphology_preservation", ascending=False
    )
    if len(factor_rows):
        count = min(5, len(factor_rows))
        middle = max(0, len(factor_rows) // 2 - count // 2)
        selected = pd.concat(
            [
                factor_rows.head(count),
                factor_rows.iloc[middle : middle + count],
                factor_rows.tail(count),
            ]
        ).reset_index(drop=True)
        selected_pairs = selected[
            ["source_index", "target_index", "protocol", "aligned_group_id"]
        ]
        _, audit_previews, _ = evaluate_model(
            "factorstain",
            out / "checkpoints" / "factorstain" / "best.pt",
            config,
            index,
            selected_pairs,
            dino,
            probes,
            lpips_model,
            device,
            preview_limit=15,
        )
        _failure_grid(
            audit_previews,
            selected.morphology_preservation.tolist(),
            figures / "best_median_worst_examples.png",
        )
        _failure_grid(
            audit_previews[-5:] * 3,
            selected.morphology_preservation.tail(5).tolist() * 3,
            figures / "failure_cases.png",
        )
    track_means = controlled.groupby("method")[list(TRACK_S)].mean().to_dict("index")
    composite_scores = {}
    for method, values in means.items():
        mmd_score = (
            1 / (1 + values["dino_mmd2"]) if np.isfinite(values["dino_mmd2"]) else 0.0
        )
        composite_scores[method] = float(
            np.mean(
                [
                    mmd_score,
                    values["stain_target_accuracy"],
                    values["scanner_target_accuracy"],
                    values["morphology_preservation"],
                    values["factor_isolation_score"],
                ]
            )
        )
    _summary_dashboard(figures, decision, composite_scores)
    metrics_payload = {
        "milestone": config["milestone"],
        "fast_dev_run": config["fast_dev_run"],
        "decision_valid": decision["decision_valid"],
        "combination_split": split,
        "episode_statistics": episode_stats,
        "model_parameter_counts": parameter_counts,
        "acquisition_classifier_training_source": probes.training_source,
        "acquisition_classifier_trained_on_generated": probes.trained_on_generated,
        "acquisition_classifier_real_samples": probes.training_samples,
        "unseen_primary_means": means,
        "track_s_means": track_means,
        "paired_bootstrap": bootstraps,
        "factor_isolation_formula": "mean((delta_target_probability+1)/2, 1-TV(non_target_probabilities), DINO_source_intervention_cosine), averaged across stain-only and scanner-only interventions",
        "elapsed_seconds": time.monotonic() - started,
        "provenance": collect_provenance(
            config["seed"], {"evaluation_pairs": len(pairs), "plism_images": len(index)}
        ),
    }
    atomic_json_dump(metrics_payload, out / "metrics.json")
    if config["milestone"] == "m1_factorial":
        _write_report(
            out,
            split,
            episode_stats,
            means,
            track_means,
            mmd,
            decision,
            parameter_counts,
        )
    else:
        _write_m2_report(out, means, decision, parameter_counts)
    update_master(config["paths"]["outputs_root"])
    print(
        f"{config['milestone']} complete: {decision['status']} "
        f"(decision_valid={decision['decision_valid']})"
    )
    print(f"Dashboard: {figures / 'SUMMARY_DASHBOARD.png'}")


if __name__ == "__main__":
    main()
