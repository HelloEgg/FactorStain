#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from factorstain.evaluation.attribution import factorial_sensitivity
from factorstain.models.fm_registry import encode_batch, load_foundation_model, preprocess_tensor_batch
from factorstain.training.renderer import _load_image, build_renderer
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output, update_master, write_decision, write_metrics, write_report
from factorstain.utils.runtime import collect_provenance


def _load_cache(path: Path) -> tuple[list[str], np.ndarray]:
    with h5py.File(path, "r") as handle:
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["image_id"][:]]
        features = handle["features"][:].astype(np.float32)
    return ids, features


def _scatter(frame: pd.DataFrame, factor: str, destination: Path) -> None:
    x, y = frame[f"real_{factor}"], frame[f"cf_{factor}"]
    rho = spearmanr(x, y).statistic if len(frame) > 1 else np.nan
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(x, y, s=22, alpha=0.65, color="#4263eb")
    maximum = max(float(x.max()), float(y.max()), 1e-9)
    axis.plot([0, maximum], [0, maximum], "--", color="#868e96")
    axis.set_xlabel("Real-grid sensitivity")
    axis.set_ylabel("Counterfactual-grid sensitivity")
    axis.set_title(f"{factor.replace('_', ' ').title()}\nSpearman ρ={rho:.3f}")
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def _heatmap(matrix: np.ndarray, stains: list[str], scanners: list[str], title: str, destination: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(scanners)), scanners, rotation=30, ha="right")
    axis.set_yticks(range(len(stains)), stains)
    axis.set_xlabel("Scanner")
    axis.set_ylabel("Stain")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="True tissue probability")
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def _sensitivity_examples(frame: pd.DataFrame, factor: str, destination: Path) -> None:
    score = frame[f"cf_{factor}"]
    selected = pd.concat([frame.loc[[score.idxmax()]], frame.loc[[score.sub(score.median()).abs().idxmin()]], frame.loc[[score.idxmin()]]])
    fig, axis = plt.subplots(figsize=(10, 4))
    labels = ["high", "mixed", "low"]
    x = np.arange(3)
    axis.bar(x - 0.18, selected[f"real_{factor}"], width=0.36, label="real", color="#364fc7")
    axis.bar(x + 0.18, selected[f"cf_{factor}"], width=0.36, label="counterfactual", color="#f76707")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Sensitivity variance")
    axis.set_title(f"Representative {factor.replace('_sensitivity', '')}-sensitivity cases")
    axis.legend()
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m3_attribution.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    outputs_root = Path(config["paths"]["outputs_root"])
    renderer_out = outputs_root / "m2_renderer"
    if not (renderer_out / "checkpoints" / "factorstain" / "best.pt").exists():
        renderer_out = outputs_root / "m1_factorial"
    index_path = renderer_out / "plism_index_with_splits.parquet"
    if not index_path.exists():
        raise FileNotFoundError("M1/M2 split index is required before counterfactual attribution")
    index = pd.read_parquet(index_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Counterfactual attribution requires CUDA; CPU fallback is disabled")
    foundation = cache = model_name = None
    for candidate in ("uni", "virchow2"):
        candidate_cache = outputs_root / "m0_probe" / "features" / candidate / "features.h5"
        if candidate_cache.exists():
            foundation = load_foundation_model(candidate, device)
            cache, model_name = candidate_cache, candidate
            break
    if foundation is None:
        raise RuntimeError("No accessible pathology FM cache from M0 is available")
    image_ids, features = _load_cache(cache)
    metadata = pd.DataFrame({"image_id": image_ids, "feature_position": np.arange(len(image_ids))}).merge(index, on="image_id", how="inner")
    tissue_encoder = LabelEncoder().fit(index.tissue_type.astype(str))
    train = metadata[metadata.morphology_split.eq("train")]
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1))
    classifier.fit(features[train.feature_position], tissue_encoder.transform(train.tissue_type.astype(str)))
    trained_tissue_classes = set(classifier.classes_.tolist())

    stains, scanners = sorted(index.stain_id.unique()), sorted(index.scanner_id.unique())
    stain_map, scanner_map = {v: i for i, v in enumerate(stains)}, {v: i for i, v in enumerate(scanners)}
    model = build_renderer("factorstain", len(stains), len(scanners)).to(device)
    checkpoint = torch.load(renderer_out / "checkpoints" / "factorstain" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.eval()
    feature_lookup = {row.image_id: int(row.feature_position) for _, row in metadata.iterrows()}
    groups = []
    for group_id, group in index[index.morphology_split.eq("test") & index.image_exists].groupby("aligned_group_id"):
        group_tissue_class = int(tissue_encoder.transform([str(group.tissue_type.mode().iloc[0])])[0])
        if group_tissue_class not in trained_tissue_classes:
            continue
        lookup = {(row.stain_id, row.scanner_id): row for _, row in group.iterrows() if row.image_id in feature_lookup}
        if len(lookup) >= config["minimum_complete_cells"] and all((s, q) in lookup for s in stains for q in scanners):
            groups.append((group_id, group, lookup))
    max_groups = 2 if config["fast_dev_run"] else config.get("max_groups", 200)
    groups = groups[:max_groups]
    if len(groups) < 2:
        raise RuntimeError("At least two complete held-out aligned groups with cached real FM features are required for attribution")

    rows, matrices = [], []
    batch_size = config["batch_size"]
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for group_id, group, lookup in groups:
        tissue = str(group.tissue_type.mode().iloc[0])
        tissue_class = int(tissue_encoder.transform([tissue])[0])
        real_matrix = np.zeros((len(stains), len(scanners)), dtype=float)
        for i, stain in enumerate(stains):
            for j, scanner in enumerate(scanners):
                row = lookup[(stain, scanner)]
                probabilities = classifier.predict_proba(features[[feature_lookup[row.image_id]]])[0]
                class_position = list(classifier.classes_).index(tissue_class)
                real_matrix[i, j] = probabilities[class_position]
        source_row = group.iloc[0]
        source = _load_image(source_row.image_path, 512).unsqueeze(0).to(device)
        combinations = [(i, j) for i in range(len(stains)) for j in range(len(scanners))]
        cf_probabilities = []
        for start in range(0, len(combinations), batch_size):
            subset = combinations[start : start + batch_size]
            sources = source.expand(len(subset), -1, -1, -1)
            stain_ids = torch.tensor([i for i, _ in subset], device=device)
            scanner_ids = torch.tensor([j for _, j in subset], device=device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype):
                generated = model(sources, stain_ids, scanner_ids)
                generated_features = encode_batch(foundation, preprocess_tensor_batch(foundation, generated)).cpu().numpy()
            probabilities = classifier.predict_proba(generated_features)
            class_position = list(classifier.classes_).index(tissue_class)
            cf_probabilities.extend(probabilities[:, class_position].tolist())
        cf_matrix = np.asarray(cf_probabilities).reshape(len(stains), len(scanners))
        real_sensitivity, cf_sensitivity = factorial_sensitivity(real_matrix), factorial_sensitivity(cf_matrix)
        row = {"aligned_group_id": group_id, "tissue_type": tissue}
        for key in ("stain_sensitivity", "scanner_sensitivity", "interaction_sensitivity"):
            row[f"real_{key}"] = real_sensitivity[key]
            row[f"cf_{key}"] = cf_sensitivity[key]
        rows.append(row)
        matrices.append((real_matrix, cf_matrix))
    results = pd.DataFrame(rows)
    correlations = {}
    metric_rows = results.to_dict("records")
    for factor in ("stain_sensitivity", "scanner_sensitivity", "interaction_sensitivity"):
        x, y = results[f"real_{factor}"], results[f"cf_{factor}"]
        correlations[factor] = {"spearman_rho": float(spearmanr(x, y).statistic), "pearson_r": float(pearsonr(x, y).statistic), "n": len(results)}
    stain_rho = correlations["stain_sensitivity"]["spearman_rho"]
    scanner_rho = correlations["scanner_sensitivity"]["spearman_rho"]
    if stain_rho >= config["decision"]["go_rho"] and scanner_rho >= config["decision"]["go_rho"]:
        status = "GO"; reasons = ["Both factor sensitivities reproduce real-grid rankings at ρ≥0.60."]
    else:
        status = "GO_WITH_SCOPE_REDUCTION"
        reasons = [f"Sensitivity correlations are stain ρ={stain_rho:.3f}, scanner ρ={scanner_rho:.3f}; retain only factors meeting threshold."]
        if stain_rho < config["decision"]["drop_rho"] and scanner_rho < config["decision"]["drop_rho"]:
            reasons.append("Both correlations are below 0.40; drop attribution as a paper contribution, while retaining renderer results.")
    if config["fast_dev_run"]:
        status = "GO_WITH_SCOPE_REDUCTION"
    observed = float(min(stain_rho, scanner_rho)) if np.isfinite(stain_rho) and np.isfinite(scanner_rho) else None
    decision = write_decision(out, status, "minimum stain/scanner Spearman rho", observed, config["decision"]["go_rho"], reasons, "Proceed to FactorAdapter; scope attribution according to factor-specific evidence.", not config["fast_dev_run"])
    summary = {
        "foundation_model": model_name,
        "correlations": correlations,
        "n_groups": len(results),
        "model_parameter_counts": {
            "frozen_foundation_model": int(sum(parameter.numel() for parameter in foundation.model.parameters())),
            "factorstain_renderer": int(sum(parameter.numel() for parameter in model.parameters())),
            "real_only_tissue_probe": int(sum(np.size(value) for value in classifier[-1].coef_)) + int(np.size(classifier[-1].intercept_)),
        },
        "provenance": collect_provenance(config["seed"], {"complete_groups": len(results)}),
    }
    write_metrics(out, metric_rows, summary)
    write_report(out, "M3 — Counterfactual sensitivity attribution", {"foundation_model": model_name, "complete_groups": len(results), **{f"{k}_rho": v["spearman_rho"] for k, v in correlations.items()}}, decision)
    results.to_csv(out / "sensitivity_per_group.csv", index=False)
    figures = out / "figures"
    _heatmap(matrices[0][0], stains, scanners, "Real acquisition sensitivity grid", figures / "real_sensitivity_heatmap.png")
    _heatmap(matrices[0][1], stains, scanners, "Counterfactual sensitivity grid", figures / "counterfactual_sensitivity_heatmap.png")
    for factor, filename in (("stain_sensitivity", "stain_sensitivity_scatter.png"), ("scanner_sensitivity", "scanner_sensitivity_scatter.png"), ("interaction_sensitivity", "interaction_sensitivity_scatter.png")):
        _scatter(results, factor, figures / filename)
    _sensitivity_examples(results, "scanner_sensitivity", figures / "scanner_sensitive_examples.png")
    _sensitivity_examples(results, "stain_sensitivity", figures / "stain_sensitive_examples.png")
    _sensitivity_examples(results, "interaction_sensitivity", figures / "mixed_sensitive_examples.png")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].imshow(matrices[0][0], cmap="viridis", vmin=0, vmax=1); axes[0].set_title("Real probabilities")
    axes[1].imshow(matrices[0][1], cmap="viridis", vmin=0, vmax=1); axes[1].set_title("Counterfactual probabilities")
    axes[2].axis("off"); axes[2].text(0.5, 0.65, decision["status"], ha="center", fontsize=24, weight="bold"); axes[2].text(0.5, 0.38, f"Stain ρ={stain_rho:.3f}\nScanner ρ={scanner_rho:.3f}", ha="center", fontsize=16)
    fig.suptitle("M3 — Does counterfactual sensitivity match the real factorial grid?", fontsize=18, weight="bold"); fig.tight_layout(); fig.savefig(figures / "summary_dashboard.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    update_master(config["paths"]["outputs_root"])


if __name__ == "__main__":
    main()
