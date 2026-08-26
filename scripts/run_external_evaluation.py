#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from factorstain.evaluation.probes import fit_grouped_probe
from factorstain.training.adapter import load_counterfactual_cache
from factorstain.training.external import _adapter_for, classification_metrics
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output, update_master, write_decision, write_metrics, write_report
from factorstain.utils.runtime import collect_provenance


def _camelyon(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((out / "predictions").glob("camelyon_*.csv")) if (out / "predictions").exists() else []
    if not files:
        return pd.DataFrame(), pd.DataFrame()
    predictions = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    rows = []
    for (model, representation, center), group in predictions.groupby(["model", "representation", "center"]):
        rows.append({"dataset": "camelyon", "model": model, "representation": representation, "center": int(center), **classification_metrics(group.label.to_numpy(), group.probability.to_numpy()), "n": len(group), "mil_parameter_count": int(group.mil_parameter_count.iloc[0]) if "mil_parameter_count" in group else None})
    return predictions, pd.DataFrame(rows)


def _midog(out: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records = []
    summaries = {}
    for model_name in config["models"]:
        bag_dir = out / "external_features" / "midog" / model_name
        paths = sorted(bag_dir.glob("*.h5")) if bag_dir.exists() else []
        if not paths:
            continue
        features, labels, scanners, cases = [], [], [], []
        for path in paths:
            with h5py.File(path, "r") as handle:
                values = handle["features"][:].astype(np.float32)
                item_labels = np.asarray(json.loads(handle.attrs["labels_json"]), dtype=int)
                scanner = str(handle.attrs["scanner_id"])
                case = str(handle.attrs["case_id"])
            features.append(values); labels.extend(item_labels); scanners.extend([scanner] * len(values)); cases.extend([case] * len(values))
        raw = np.concatenate(features); labels = np.asarray(labels); scanners = np.asarray(scanners); cases = np.asarray(cases)
        representations = {"raw": raw}
        try:
            adapter = _adapter_for(model_name, config, raw.shape[1])
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(raw), 2048): chunks.append(adapter.encode(torch.from_numpy(raw[start : start + 2048])).numpy())
            representations["adapter"] = np.concatenate(chunks)
        except FileNotFoundError:
            pass
        for representation, values in representations.items():
            for heldout in sorted(np.unique(scanners)):
                train, test = scanners != heldout, scanners == heldout
                if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
                    continue
                classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
                classifier.fit(values[train], labels[train])
                probabilities = classifier.predict_proba(values[test])[:, 1]
                metric = classification_metrics(labels[test], probabilities)
                records.append({"dataset": "midog", "model": model_name, "representation": representation, "scanner": heldout, **metric, "n": int(test.sum())})
            leakage, _, _, _ = fit_grouped_probe(values, scanners, cases, seed=config["seed"])
            summaries[(model_name, representation)] = {"scanner_leakage_balanced_accuracy": leakage["balanced_accuracy"], "scanner_leakage_chance": leakage["chance"]}
        # PCA source for the requested scanner view.
        positions = np.arange(len(raw)); positions = positions[: min(10000, len(positions))]
        reduced = PCA(2, random_state=config["seed"]).fit_transform(raw[positions]); encoded, names = pd.factorize(scanners[positions], sort=True)
        fig, axis = plt.subplots(figsize=(7, 6)); axis.scatter(reduced[:, 0], reduced[:, 1], c=encoded, cmap="tab10", s=6, alpha=0.6); axis.set_title(f"MIDOG {model_name}: raw FM by scanner"); axis.set_xticks([]); axis.set_yticks([]); fig.tight_layout(); fig.savefig(out / "figures" / "midog_umap_by_scanner.png", dpi=180); plt.close(fig)
    metrics = pd.DataFrame(records)
    if not metrics.empty:
        for (model_name, representation), group in metrics.groupby(["model", "representation"]):
            summaries[(model_name, representation)].update(
                {
                    "worst_scanner_auroc": float(group.auroc.min()),
                    "mean_scanner_auroc": float(group.auroc.mean()),
                    "scanner_performance_gap": float(group.auroc.max() - group.auroc.min()),
                }
            )
    return metrics, metrics, summaries


def _plot_external(cam_predictions: pd.DataFrame, cam_metrics: pd.DataFrame, midog_metrics: pd.DataFrame, midog_summary: dict, out: Path) -> None:
    figures = out / "figures"
    if not cam_metrics.empty:
        pivot = cam_metrics.pivot_table(index=["model", "center"], columns="representation", values="auroc").reset_index()
        fig, axis = plt.subplots(figsize=(12, 5));
        for representation, color in (("raw", "#4263eb"), ("adapter", "#f76707")):
            if representation in pivot:
                axis.plot(np.arange(len(pivot)), pivot[representation], "o-", label=representation, color=color)
        axis.set_xticks(np.arange(len(pivot)), pivot.model + " / C" + pivot.center.astype(str), rotation=30, ha="right"); axis.set_ylabel("AUROC"); axis.set_title("Leave-one-center-out CAMELYON17 performance"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "center_auroc_bar.png", dpi=180); plt.close(fig)
        aggregates = cam_metrics.groupby(["model", "representation"]).auroc.agg(["mean", "min", "max"]).reset_index(); aggregates["gap"] = aggregates["max"] - aggregates["min"]
        fig, axis = plt.subplots(figsize=(9, 5));
        for representation, group in aggregates.groupby("representation"):
            axis.bar(np.arange(len(group)) + (0 if representation == "raw" else 0.35), group["gap"], 0.35, label=representation)
        axis.set_xticks(np.arange(len(aggregates.model.unique())) + 0.175, sorted(aggregates.model.unique())); axis.set_ylabel("Best − worst center AUROC"); axis.set_title("Center performance gap"); axis.legend(); fig.tight_layout(); fig.savefig(figures / "center_gap.png", dpi=180); plt.close(fig)
        worst = cam_metrics.loc[cam_metrics.groupby(["model", "representation"]).auroc.idxmin()]
        fig, axis = plt.subplots(figsize=(9, 5)); axis.bar(np.arange(len(worst)), worst.auroc, color=["#4263eb" if value == "raw" else "#f76707" for value in worst.representation]); axis.set_xticks(np.arange(len(worst)), worst.model + "\n" + worst.representation + " C" + worst.center.astype(str)); axis.set_title("Worst-center AUROC"); fig.tight_layout(); fig.savefig(figures / "worst_center_comparison.png", dpi=180); plt.close(fig)
        fig, axis = plt.subplots(figsize=(8, 7))
        for (model, representation, center), group in cam_predictions.groupby(["model", "representation", "center"]):
            if len(np.unique(group.label)) == 2:
                RocCurveDisplay.from_predictions(group.label, group.probability, name=f"{model}/{representation}/C{center}", ax=axis, plot_chance_level=False)
        axis.set_title("Per-center ROC curves"); axis.legend(fontsize=6); fig.tight_layout(); fig.savefig(figures / "per_center_roc.png", dpi=180); plt.close(fig)
        fig, axis = plt.subplots(figsize=(9, 5)); table = cam_metrics.groupby(["model", "representation"]).auroc.mean().unstack(); table.plot.bar(ax=axis, color=["#4263eb", "#f76707"]); axis.set_title("Raw FM vs FactorAdapter: mean-center AUROC"); axis.tick_params(axis="x", rotation=0); fig.tight_layout(); fig.savefig(figures / "raw_vs_adapter.png", dpi=180); plt.close(fig)
        shutil.copy2(figures / "raw_vs_adapter.png", figures / "summary_dashboard_camelyon.png")
    if not midog_metrics.empty:
        table = midog_metrics.pivot_table(index=["model", "scanner"], columns="representation", values="auroc")
        fig, axis = plt.subplots(figsize=(12, 5)); table.plot.bar(ax=axis); axis.set_title("MIDOG leave-one-scanner-out AUROC"); axis.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(figures / "midog_scanner_performance.png", dpi=180); plt.close(fig)
        shutil.copy2(figures / "midog_scanner_performance.png", figures / "midog_raw_vs_adapter.png")
        leakage_frame = pd.DataFrame([{"model": key[0], "representation": key[1], **value} for key, value in midog_summary.items()])
        fig, axis = plt.subplots(figsize=(8, 5)); leakage_frame.pivot(index="model", columns="representation", values="scanner_leakage_balanced_accuracy").plot.bar(ax=axis); axis.set_title("MIDOG scanner leakage"); axis.tick_params(axis="x", rotation=0); fig.tight_layout(); fig.savefig(figures / "midog_scanner_leakage.png", dpi=180); plt.close(fig)


def _final_outputs(outputs: Path) -> None:
    decisions = {}
    for milestone in ("m0_probe", "m1_factorial", "m2_renderer", "m3_attribution", "m4_adapter", "m5_external"):
        path = outputs / milestone / "GO_NOGO.json"
        decisions[milestone] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "MISSING", "primary_metric": "-", "observed": None}
    statuses = {key: value["status"] for key, value in decisions.items()}
    if statuses["m1_factorial"] == "NO_GO" or statuses["m0_probe"] == "NO_GO": recommendation = "STOP"
    elif all(statuses.get(key) == "GO" for key in ("m0_probe", "m1_factorial", "m2_renderer", "m3_attribution", "m4_adapter", "m5_external")): recommendation = "STRONG_GO_FOR_CVPR"
    elif statuses["m1_factorial"] == "GO": recommendation = "GO_BUT_REDUCE_SCOPE"
    else: recommendation = "PIVOT"
    questions = [
        ("Was stain/scanner leakage measurable?", "m0_probe"),
        ("Did factorization improve unseen-combination generation?", "m1_factorial"),
        ("Did S→Q ordering matter?", "m2_renderer"),
        ("Did counterfactual sensitivity match real sensitivity?", "m3_attribution"),
        ("Did FactorAdapter remove acquisition information without losing biology?", "m4_adapter"),
        ("Did external center/scanner robustness improve?", "m5_external"),
    ]
    lines = ["# FactorStain Final Report", "", f"## Recommendation: {recommendation}", ""]
    for question, milestone in questions:
        decision = decisions[milestone]; lines.extend([f"## {question}", "", f"**{decision['status']}** — {decision['primary_metric']}: {decision['observed']}", ""])
    (outputs / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    table_dir, figure_dir = outputs / "FINAL_TABLES", outputs / "FINAL_FIGURES"; table_dir.mkdir(exist_ok=True); figure_dir.mkdir(exist_ok=True)
    for milestone in decisions:
        source = outputs / milestone / "metrics.csv"
        if source.exists(): shutil.copy2(source, table_dir / f"{milestone}_metrics.csv")
        for table_name in ("statistical_comparisons.csv", "paired_bootstrap_comparisons.csv", "sensitivity_per_group.csv"):
            table_source = outputs / milestone / table_name
            if table_source.exists(): shutil.copy2(table_source, table_dir / f"{milestone}_{table_name}")
    master = outputs / "MASTER_DASHBOARD.png"
    if master.exists(): shutil.copy2(master, outputs / "FINAL_DASHBOARD.png")
    for name in ("plism_13x7_real_grid.png", "factorial_2x2_examples.png", "seen_vs_unseen_metrics.png", "summary_dashboard.png", "leakage_before_after.png", "center_auroc_bar.png"):
        matches = list(outputs.glob(f"m*/figures/{name}"))
        for source in matches: shutil.copy2(source, figure_dir / f"{source.parents[1].name}_{name}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/m5_external.yaml"); args = parser.parse_args()
    config = load_config(args.config); out = prepare_output(config)
    cam_predictions, cam_metrics = _camelyon(out)
    midog_predictions, midog_metrics, midog_summary = _midog(out, config)
    rows = [*cam_metrics.to_dict("records"), *midog_metrics.to_dict("records")]
    comparisons, passes = {}, []
    if not cam_metrics.empty:
        aggregates = cam_metrics.groupby(["model", "representation"]).auroc.agg(["mean", "min", "max"]).reset_index(); aggregates["gap"] = aggregates["max"] - aggregates["min"]
        for model in aggregates.model.unique():
            raw = aggregates[(aggregates.model == model) & (aggregates.representation == "raw")]
            adapted = aggregates[(aggregates.model == model) & (aggregates.representation == "adapter")]
            if raw.empty or adapted.empty: continue
            raw, adapted = raw.iloc[0], adapted.iloc[0]
            result = {"worst_center_delta": float(adapted["min"] - raw["min"]), "center_gap_reduction": float((raw.gap - adapted.gap) / max(raw.gap, 1e-12)), "mean_auroc_delta": float(adapted["mean"] - raw["mean"])}
            comparisons[model] = result
            if result["worst_center_delta"] >= config["decision"]["worst_center_delta"] and result["center_gap_reduction"] >= config["decision"]["center_gap_reduction"] and result["mean_auroc_delta"] >= -config["decision"]["max_mean_drop"]: passes.append(model)
    if passes: status = "GO"; reasons = [f"External robustness criteria met for {passes}."]
    elif comparisons and any(value["mean_auroc_delta"] > 0 for value in comparisons.values()): status = "GO_WITH_SCOPE_REDUCTION"; reasons = ["Mean AUROC improved, but worst-center/gap criteria were not both met."]
    else: status = "NO_GO"; reasons = ["No FactorAdapter configuration met the external center-robustness criterion."]
    if config["fast_dev_run"]: status = "GO_WITH_SCOPE_REDUCTION"
    observed = max((value["worst_center_delta"] for value in comparisons.values()), default=None)
    decision = write_decision(out, status, "best worst-center AUROC improvement", observed, config["decision"]["worst_center_delta"], reasons, "Use MIDOG as corroboration and scope the clinical robustness claim to measured center results.", not config["fast_dev_run"])
    summary = {"camelyon_comparisons": comparisons, "midog_scanner_leakage": {f"{k[0]}:{k[1]}": v for k, v in midog_summary.items()}, "provenance": collect_provenance(config["seed"])}
    write_metrics(out, rows, summary); write_report(out, "M5 — External clinical generalization", {"camelyon_comparisons": comparisons, "passing_models": passes, "midog_experiments": len(midog_metrics), "midog_summary": summary["midog_scanner_leakage"]}, decision)
    _plot_external(cam_predictions, cam_metrics, midog_metrics, midog_summary, out)
    fig, axis = plt.subplots(figsize=(12, 5)); axis.axis("off"); axis.text(0.5,0.7,decision["status"],ha="center",fontsize=28,weight="bold"); axis.text(0.5,0.45,f"Best worst-center ΔAUROC: {observed}\nPassing models: {passes}",ha="center",fontsize=16); axis.text(0.5,0.2,"CAMELYON17: leave-one-center-out\nMIDOG21: leave-one-scanner-out",ha="center",fontsize=13); fig.tight_layout(); fig.savefig(out/"figures"/"summary_dashboard.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    update_master(config["paths"]["outputs_root"]); _final_outputs(Path(config["paths"]["outputs_root"]))


if __name__ == "__main__": main()
