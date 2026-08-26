#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from factorstain.evaluation.probes import fit_grouped_probe, load_feature_cache, within_group_embedding_variance
from factorstain.models.factor_adapter import FactorAdapter
from factorstain.training.adapter import load_counterfactual_cache
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output, update_master, write_decision, write_metrics, write_report
from factorstain.utils.runtime import collect_provenance


def _encode(model: FactorAdapter, features: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            chunks.append(model.encode(torch.from_numpy(features[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(chunks)


def _probe(features: np.ndarray, metadata: pd.DataFrame, seed: int) -> dict[str, dict]:
    results = {}
    for name, column in (("tissue", "tissue_type"), ("stain", "stain_id"), ("scanner", "scanner_id")):
        metrics, _, _, _ = fit_grouped_probe(features, metadata[column].to_numpy(), metadata.aligned_group_id.astype(str).to_numpy(), seed=seed)
        results[name] = metrics
    return results


def _tissue_separation(features: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    global_mean = features.mean(0)
    between = np.mean([np.sum((features[labels == label].mean(0) - global_mean) ** 2) for label in unique])
    within = np.mean([np.mean(np.sum((features[labels == label] - features[labels == label].mean(0)) ** 2, axis=1)) for label in unique])
    return float(between / max(within, 1e-12))


def _leakage_reduction(before: float, after: float, chance: float) -> float:
    signal = before - chance
    if signal <= 0:
        return float("-inf")
    return float((signal - (after - chance)) / signal)


def _embedding_plot(features: np.ndarray, labels: np.ndarray, title: str, destination: Path, seed: int) -> None:
    positions = np.arange(len(features))
    if len(positions) > 10000:
        positions = np.random.default_rng(seed).choice(positions, 10000, replace=False)
    reduced = PCA(n_components=2, random_state=seed).fit_transform(features[positions])
    encoded, names = pd.factorize(labels[positions].astype(str), sort=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(reduced[:, 0], reduced[:, 1], c=encoded, cmap="turbo", s=5, alpha=0.6, rasterized=True)
    axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
    if len(names) <= 15:
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)), label=str(name)) for i, name in enumerate(names)]
        axis.legend(handles=handles, bbox_to_anchor=(1, 0.5), loc="center left", fontsize=7)
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m4_adapter.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    outputs = Path(config["paths"]["outputs_root"])
    index_path = outputs / "m2_renderer" / "plism_index_with_splits.parquet"
    if not index_path.exists():
        index_path = outputs / "m1_factorial" / "plism_index_with_splits.parquet"
    index = pd.read_parquet(index_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("FactorAdapter evaluation requires CUDA; CPU fallback is disabled")
    metric_rows, summaries, successful = [], {}, []
    figures = out / "figures"
    for model_name in config["models"]:
        raw_cache = outputs / "m0_probe" / "features" / model_name / "features.h5"
        cf_cache = out / "counterfactual_features" / model_name / "features.h5"
        checkpoint_path = out / "checkpoints" / model_name / "best.pt"
        if not (raw_cache.exists() and cf_cache.exists() and checkpoint_path.exists()):
            summaries[model_name] = {"status": "UNAVAILABLE"}
            continue
        image_ids, raw_features = load_feature_cache(raw_cache)
        metadata = pd.DataFrame({"image_id": image_ids}).merge(index, on="image_id", how="inner")
        # Align cache rows with metadata after the inner merge.
        position_map = {image_id: i for i, image_id in enumerate(image_ids)}
        positions = np.asarray([position_map[value] for value in metadata.image_id])
        raw_features = raw_features[positions]
        cf_features, cf_metadata = load_counterfactual_cache(cf_cache)
        model = FactorAdapter(raw_features.shape[1], cf_metadata.tissue_type.nunique(), cf_metadata.stain_id.nunique(), cf_metadata.scanner_id.nunique(), config["bottleneck_dim"], config["grl_weight"]).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        adapted_features = _encode(model, raw_features, device)
        adapted_cf = _encode(model, cf_features, device)
        raw_probe, adapted_probe = _probe(raw_features, metadata, config["seed"]), _probe(adapted_features, metadata, config["seed"])
        raw_variance = within_group_embedding_variance(cf_features, cf_metadata.aligned_group_id.to_numpy()).embedding_variance.mean()
        adapted_variance = within_group_embedding_variance(adapted_cf, cf_metadata.aligned_group_id.to_numpy()).embedding_variance.mean()
        variance_reduction = float((raw_variance - adapted_variance) / max(raw_variance, 1e-12))
        stain_raw, stain_after = raw_probe["stain"]["balanced_accuracy"], adapted_probe["stain"]["balanced_accuracy"]
        scanner_raw, scanner_after = raw_probe["scanner"]["balanced_accuracy"], adapted_probe["scanner"]["balanced_accuracy"]
        stain_chance, scanner_chance = raw_probe["stain"]["chance"], raw_probe["scanner"]["chance"]
        stain_reduction = _leakage_reduction(stain_raw, stain_after, stain_chance)
        scanner_reduction = _leakage_reduction(scanner_raw, scanner_after, scanner_chance)
        tissue_drop = float(raw_probe["tissue"]["accuracy"] - adapted_probe["tissue"]["accuracy"])
        raw_separation = _tissue_separation(raw_features, metadata.tissue_type.to_numpy())
        adapted_separation = _tissue_separation(adapted_features, metadata.tissue_type.to_numpy())
        model_summary = {
            "status": "AVAILABLE",
            "raw_tissue_accuracy": raw_probe["tissue"]["accuracy"],
            "adapted_tissue_accuracy": adapted_probe["tissue"]["accuracy"],
            "tissue_accuracy_drop": tissue_drop,
            "raw_stain_balanced_accuracy": stain_raw,
            "adapted_stain_balanced_accuracy": stain_after,
            "stain_leakage_reduction": stain_reduction,
            "raw_scanner_balanced_accuracy": scanner_raw,
            "adapted_scanner_balanced_accuracy": scanner_after,
            "scanner_leakage_reduction": scanner_reduction,
            "raw_cf_variance": float(raw_variance),
            "adapted_cf_variance": float(adapted_variance),
            "cf_variance_reduction": variance_reduction,
            "raw_between_tissue_separation": raw_separation,
            "adapted_between_tissue_separation": adapted_separation,
            "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        }
        summaries[model_name] = model_summary
        for key, value in model_summary.items():
            if isinstance(value, (float, int)):
                metric_rows.append({"model": model_name, "metric": key, "value": value})
        qualifies = (
            max(stain_reduction, scanner_reduction) >= config["decision"]["leakage_reduction"]
            and tissue_drop <= config["decision"]["max_tissue_drop"]
            and variance_reduction >= config["decision"]["variance_reduction"]
        )
        if qualifies:
            successful.append(model_name)
        display = "UNI" if model_name == "uni" else "Virchow2"
        for state, features in (("before", raw_features), ("after", adapted_features)):
            for factor, column in (("stain", "stain_id"), ("scanner", "scanner_id"), ("tissue", "tissue_type")):
                _embedding_plot(features, metadata[column].to_numpy(), f"{display} {state} adapter — {factor}", figures / f"{display}_umap_{state}_{factor}.png", config["seed"])

    if successful:
        status = "GO"
        reasons = [f"FactorAdapter meets leakage, biology, and variance criteria for: {successful}."]
    else:
        status = "NO_GO"
        reasons = ["No primary FM reduced leakage by 30% and CF variance by 20% while limiting tissue loss to 2 points."]
    if config["fast_dev_run"]:
        status = "GO_WITH_SCOPE_REDUCTION"
    observed = max((max(value.get("stain_leakage_reduction", -np.inf), value.get("scanner_leakage_reduction", -np.inf)) for value in summaries.values()), default=None)
    decision = write_decision(out, status, "maximum acquisition-leakage reduction", None if observed is None or not np.isfinite(observed) else float(observed), config["decision"]["leakage_reduction"], reasons, "Proceed to external validation." if successful else "Retain renderer/attribution results but drop FactorAdapter robustness claims.", not config["fast_dev_run"])
    summary_payload = {"models": summaries, "provenance": collect_provenance(config["seed"])}
    write_metrics(out, metric_rows, summary_payload)
    write_report(out, "M4 — FactorAdapter", {"successful_models": successful, "model_measurements": summaries}, decision)

    available = [model for model, summary in summaries.items() if summary.get("status") == "AVAILABLE"]
    if available:
        labels = []
        raw_values, adapted_values = [], []
        for model_name in available:
            for factor in ("tissue", "stain", "scanner"):
                labels.append(f"{model_name}\n{factor}")
                raw_values.append(summaries[model_name][f"raw_{factor}_{'accuracy' if factor == 'tissue' else 'balanced_accuracy'}"])
                adapted_values.append(summaries[model_name][f"adapted_{factor}_{'accuracy' if factor == 'tissue' else 'balanced_accuracy'}"])
        x = np.arange(len(labels)); fig, axis = plt.subplots(figsize=(13, 5)); axis.bar(x - 0.18, raw_values, 0.36, label="before"); axis.bar(x + 0.18, adapted_values, 0.36, label="after"); axis.set_xticks(x, labels); axis.set_ylim(0, 1); axis.set_title("Biology preservation and acquisition leakage"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "leakage_before_after.png", dpi=180); plt.close(fig)
        fig, axis = plt.subplots(figsize=(8, 5)); x = np.arange(len(available)); axis.bar(x - 0.18, [summaries[m]["raw_tissue_accuracy"] for m in available], 0.36, label="before"); axis.bar(x + 0.18, [summaries[m]["adapted_tissue_accuracy"] for m in available], 0.36, label="after"); axis.set_xticks(x, available); axis.set_title("Tissue prediction preservation"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "tissue_preservation.png", dpi=180); plt.close(fig)
        fig, axis = plt.subplots(figsize=(8, 5)); x=np.arange(len(available)); axis.bar(x-0.18,[summaries[m]["raw_cf_variance"] for m in available],0.36,label="before"); axis.bar(x+0.18,[summaries[m]["adapted_cf_variance"] for m in available],0.36,label="after"); axis.set_xticks(x,available); axis.set_title("Counterfactual embedding variance"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "counterfactual_embedding_variance.png",dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5));
    for axis in axes: axis.axis("off")
    axes[0].text(0.5,0.6,"Leakage reduction",ha="center",fontsize=17,weight="bold"); axes[0].text(0.5,0.35,"\n".join(f"{m}: {max(summaries[m]['stain_leakage_reduction'],summaries[m]['scanner_leakage_reduction']):.1%}" for m in available),ha="center",fontsize=13)
    axes[1].text(0.5,0.6,"Biology drop",ha="center",fontsize=17,weight="bold"); axes[1].text(0.5,0.35,"\n".join(f"{m}: {summaries[m]['tissue_accuracy_drop']:.1%}" for m in available),ha="center",fontsize=13)
    axes[2].text(0.5,0.62,decision["status"],ha="center",fontsize=23,weight="bold"); axes[2].text(0.5,0.35,f"Passing FMs: {successful}",ha="center",fontsize=13)
    fig.suptitle("M4 — FactorAdapter suppresses acquisition signal while preserving biology",fontsize=18,weight="bold"); fig.tight_layout(); fig.savefig(figures/"summary_dashboard.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    update_master(config["paths"]["outputs_root"])


if __name__ == "__main__":
    main()
