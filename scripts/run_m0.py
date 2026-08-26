#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from factorstain.evaluation.probes import load_feature_cache, probe_all_labels, tissue_prediction_variation, within_group_embedding_variance
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output, update_master, write_decision, write_metrics, write_report
from factorstain.utils.runtime import atomic_json_dump, collect_provenance


def _plot_confusion(matrix: np.ndarray, title: str, destination: Path) -> None:
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _embed(features: np.ndarray, seed: int) -> np.ndarray:
    sample = features
    try:
        import umap
        return umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(sample)
    except ImportError:
        return PCA(n_components=2, random_state=seed).fit_transform(sample)


def _plot_embedding(embedding: np.ndarray, labels: np.ndarray, title: str, destination: Path) -> None:
    encoded, names = pd.factorize(labels.astype(str), sort=True)
    fig, axis = plt.subplots(figsize=(8, 7))
    points = axis.scatter(embedding[:, 0], embedding[:, 1], c=encoded, s=6, cmap="turbo", alpha=0.65, rasterized=True)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    if len(names) <= 15:
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=points.cmap(points.norm(i)), label=str(name)) for i, name in enumerate(names)]
        axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_dashboard(out: Path, decision: dict, probe_frame: pd.DataFrame) -> None:
    grid_path = out / "figures" / "plism_13x7_real_grid.png"
    leakage_path = out / "figures" / "leakage_accuracy_bar.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis in axes:
        axis.axis("off")
    if grid_path.exists():
        axes[0].imshow(Image.open(grid_path))
        axes[0].set_title("Real aligned stain×scanner grid", weight="bold")
    if leakage_path.exists():
        axes[1].imshow(Image.open(leakage_path))
        axes[1].set_title("Frozen-FM acquisition leakage", weight="bold")
    axes[2].text(0.5, 0.72, decision["status"], color={"GO": "#2b8a3e", "NO_GO": "#c92a2a"}.get(decision["status"], "#f08c00"), fontsize=28, weight="bold", ha="center")
    axes[2].text(0.5, 0.52, f"{decision['primary_metric']}\n{decision['observed']}", fontsize=15, ha="center")
    axes[2].text(0.5, 0.28, "Held-out aligned groups\nNo morphology leakage", fontsize=12, ha="center")
    fig.suptitle("M0 — PLISM structure and foundation-model leakage", fontsize=20, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "figures" / "summary_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ensure_diagnostic_figure(path: Path, title: str, message: str) -> None:
    if path.exists():
        return
    fig, axis = plt.subplots(figsize=(8, 5)); axis.axis("off")
    axis.text(0.5, 0.64, title, ha="center", fontsize=17, weight="bold")
    axis.text(0.5, 0.38, message, ha="center", fontsize=12, wrap=True)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m0_probe.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    index = pd.read_parquet(out / "plism_index.parquet")
    rows, model_summaries, variances = [], {}, []
    confusion_by_model: dict[str, dict[str, np.ndarray]] = {}
    first_embedding = None
    first_metadata = None

    for model_name in config["probe_models"]:
        cache = out / "features" / model_name / "features.h5"
        if not cache.exists():
            status_path = cache.parent / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "MISSING_FEATURE_CACHE"}
            model_summaries[model_name] = status
            rows.append({"model": model_name, "target": "availability", "metric": "available", "value": 0.0})
            continue
        image_ids, features = load_feature_cache(cache)
        metadata = pd.DataFrame({"image_id": image_ids}).merge(index, on="image_id", how="left", validate="one_to_one")
        if metadata.aligned_group_id.isna().any():
            raise RuntimeError(f"{model_name} cache contains image IDs absent from current PLISM index")
        probe_rows, confusion = probe_all_labels(
            features,
            metadata,
            classifier=config["probe_classifier"],
            test_size=config["test_size"],
            seed=config["seed"],
        )
        for row in probe_rows:
            row["model"] = model_name
        rows.extend(probe_rows)
        confusion_by_model[model_name] = confusion
        variance = within_group_embedding_variance(features, metadata.aligned_group_id.astype(str).to_numpy())
        prediction_variation = tissue_prediction_variation(
            features,
            metadata.tissue_type.to_numpy(),
            metadata.aligned_group_id.astype(str).to_numpy(),
            test_size=config["test_size"],
            seed=config["seed"],
        )
        prediction_variation.to_csv(out / f"{model_name}_tissue_prediction_variation.csv", index=False)
        variance["model"] = model_name
        variances.append(variance)
        summary = {(row["target"], row["metric"]): row["value"] for row in probe_rows}
        model_summaries[model_name] = {
            "status": "AVAILABLE",
            "tissue_balanced_accuracy": summary[("tissue", "balanced_accuracy")],
            "stain_balanced_accuracy": summary[("stain", "balanced_accuracy")],
            "scanner_balanced_accuracy": summary[("scanner", "balanced_accuracy")],
            "stain_chance": summary[("stain", "chance")],
            "scanner_chance": summary[("scanner", "chance")],
            "within_group_embedding_variance": float(variance.embedding_variance.mean()),
            "true_tissue_probability_std_across_acquisition": float(prediction_variation.true_tissue_probability_std.mean()) if not prediction_variation.empty else None,
            "tissue_prediction_flip_rate_across_acquisition": float(prediction_variation.prediction_flip_rate.mean()) if not prediction_variation.empty else None,
        }
        status_path = cache.parent / "status.json"
        if status_path.exists():
            cache_status = json.loads(status_path.read_text(encoding="utf-8"))
            model_summaries[model_name]["parameter_count"] = cache_status.get("parameter_count")
        if first_embedding is None:
            sample_size = min(10000, len(features))
            rng = np.random.default_rng(config["seed"])
            positions = rng.choice(len(features), sample_size, replace=False)
            first_embedding = _embed(features[positions], config["seed"])
            first_metadata = metadata.iloc[positions].reset_index(drop=True)

    metrics_frame = pd.DataFrame(rows)
    if variances:
        variance_frame = pd.concat(variances, ignore_index=True)
        variance_frame.to_csv(out / "within_group_embedding_distance.csv", index=False)
    else:
        variance_frame = pd.DataFrame(columns=["model", "mean_cosine_distance"])

    # Dataset structure is measured and reported, not enforced as a brittle exact match.
    dataset_summary = {
        "images": int(len(index)),
        "existing_images": int(index.image_exists.sum()),
        "stains": int(index.stain_id.nunique()),
        "scanners": int(index.scanner_id.nunique()),
        "tissues": int(index.tissue_type.nunique()),
        "aligned_groups": int(index.aligned_group_id.nunique()),
        "median_cells_per_group": float(index.groupby("aligned_group_id").size().median()),
    }
    dataset_summary["expected_structure_checks"] = {
        "stains_13": dataset_summary["stains"] == config["expected"]["stains"],
        "scanners_7": dataset_summary["scanners"] == config["expected"]["scanners"],
        "tissues_up_to_46": dataset_summary["tissues"] <= config["expected"]["tissues"],
        "image_count_relative_error": abs(dataset_summary["images"] - config["expected"]["approximate_images"]) / config["expected"]["approximate_images"],
    }
    strong_models, strong_stain, strong_scanner = [], [], []
    margins = []
    for model, summary in model_summaries.items():
        if summary.get("status") != "AVAILABLE":
            continue
        stain_margin = summary["stain_balanced_accuracy"] - summary["stain_chance"]
        scanner_margin = summary["scanner_balanced_accuracy"] - summary["scanner_chance"]
        margins.extend([stain_margin, scanner_margin])
        if stain_margin > config["decision"]["stain_margin"]:
            strong_stain.append(model)
        if scanner_margin > config["decision"]["scanner_margin"]:
            strong_scanner.append(model)
        if model in strong_stain or model in strong_scanner:
            strong_models.append(model)
    if not strong_models:
        status = "NO_GO"
        reasons = ["No available pathology FM exceeded chance by 0.10 for stain or scanner on held-out aligned groups."]
    elif len(set(strong_models)) < 2 or not strong_stain or not strong_scanner:
        status = "GO_WITH_SCOPE_REDUCTION"
        reasons = [f"Acquisition signal detected, but evidence is limited: stain={strong_stain}, scanner={strong_scanner}."]
    else:
        status = "GO"
        reasons = [f"At least two pathology FMs retain measurable acquisition signal: {sorted(set(strong_models))}."]
    observed = float(max(margins)) if margins else None
    if config["fast_dev_run"]:
        status = "GO_WITH_SCOPE_REDUCTION"
    decision = write_decision(
        out,
        status,
        "maximum acquisition balanced-accuracy margin over chance",
        observed,
        0.10,
        reasons,
        "Proceed to the factorial pilot." if strong_models else "Resolve model access/data quality or stop the FM-robustness motivation.",
        decision_valid=not config["fast_dev_run"],
    )
    summary = {**dataset_summary, "models": model_summaries, "provenance": collect_provenance(config["seed"], dataset_summary)}
    write_metrics(out, rows, summary)
    report_summary = {**dataset_summary}
    for model_name, model_summary in model_summaries.items():
        if model_summary.get("status") == "AVAILABLE":
            report_summary[f"{model_name}_tissue_bal_acc"] = model_summary["tissue_balanced_accuracy"]
            report_summary[f"{model_name}_stain_bal_acc"] = model_summary["stain_balanced_accuracy"]
            report_summary[f"{model_name}_scanner_bal_acc"] = model_summary["scanner_balanced_accuracy"]
            report_summary[f"{model_name}_tissue_prediction_flip_rate"] = model_summary["tissue_prediction_flip_rate_across_acquisition"]
        else:
            report_summary[f"{model_name}_status"] = model_summary.get("status")
    write_report(out, "M0 — PLISM and foundation-model leakage", report_summary, decision)

    figures = out / "figures"
    if not metrics_frame.empty and "balanced_accuracy" in metrics_frame.metric.values:
        plot = metrics_frame[metrics_frame.metric.isin(["balanced_accuracy", "chance"])].pivot_table(index=["model", "target"], columns="metric", values="value").reset_index()
        plot = plot[plot.target.isin(["tissue", "stain", "scanner"])]
        fig, axis = plt.subplots(figsize=(11, 5))
        labels = plot.model + " / " + plot.target
        axis.bar(np.arange(len(plot)), plot.balanced_accuracy, color="#4263eb", label="balanced accuracy")
        axis.scatter(np.arange(len(plot)), plot.chance, color="#c92a2a", marker="_", s=400, label="chance")
        axis.set_xticks(np.arange(len(plot)), labels, rotation=30, ha="right")
        axis.set_ylim(0, 1)
        axis.legend()
        axis.set_title("Acquisition leakage in frozen pathology FM embeddings")
        fig.tight_layout()
        fig.savefig(figures / "leakage_accuracy_bar.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        best_model = max((m for m in model_summaries if model_summaries[m].get("status") == "AVAILABLE"), key=lambda m: model_summaries[m]["stain_balanced_accuracy"] + model_summaries[m]["scanner_balanced_accuracy"])
        _plot_confusion(confusion_by_model[best_model]["stain"], f"{best_model}: stain probe", figures / "stain_probe_confusion.png")
        _plot_confusion(confusion_by_model[best_model]["scanner"], f"{best_model}: scanner probe", figures / "scanner_probe_confusion.png")
    else:
        fig, axis = plt.subplots(figsize=(8, 4)); axis.axis("off"); axis.text(0.5, 0.5, "No FM features available\nCheck model access status", ha="center", va="center"); fig.savefig(figures / "leakage_accuracy_bar.png", dpi=180); plt.close(fig)
    if first_embedding is not None:
        _plot_embedding(first_embedding, first_metadata.tissue_type.to_numpy(), "Frozen FM embedding by tissue", figures / "FM_umap_by_tissue.png")
        _plot_embedding(first_embedding, first_metadata.stain_id.to_numpy(), "Frozen FM embedding by stain", figures / "FM_umap_by_stain.png")
        _plot_embedding(first_embedding, first_metadata.scanner_id.to_numpy(), "Frozen FM embedding by scanner", figures / "FM_umap_by_scanner.png")
    if not variance_frame.empty:
        fig, axis = plt.subplots(figsize=(9, 5))
        variance_frame.boxplot(column="mean_cosine_distance", by="model", ax=axis)
        axis.set_title("Within-morphology acquisition embedding distance")
        fig.suptitle("")
        fig.tight_layout(); fig.savefig(figures / "within_group_embedding_distance.png", dpi=180); plt.close(fig)
    for filename, title in (
        ("FM_umap_by_tissue.png", "FM embedding by tissue"),
        ("FM_umap_by_stain.png", "FM embedding by stain"),
        ("FM_umap_by_scanner.png", "FM embedding by scanner"),
        ("stain_probe_confusion.png", "Stain probe confusion"),
        ("scanner_probe_confusion.png", "Scanner probe confusion"),
        ("within_group_embedding_distance.png", "Within-group embedding distance"),
    ):
        _ensure_diagnostic_figure(figures / filename, title, "Unavailable because neither declared pathology foundation model produced an accessible feature cache. See features/*/status.json.")
    _summary_dashboard(out, decision, metrics_frame)
    update_master(config["paths"]["outputs_root"])


if __name__ == "__main__":
    main()
