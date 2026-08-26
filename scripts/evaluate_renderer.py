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
import torch
from PIL import Image

from factorstain.evaluation.renderer import RealAcquisitionProbes, build_evaluation_pairs, evaluate_model
from factorstain.metrics.statistics import paired_bootstrap, summarize_samples
from factorstain.models.fm_registry import BlockedModelAccess, load_foundation_model
from factorstain.utils.config import load_config
from factorstain.utils.outputs import prepare_output, update_master, write_decision, write_metrics, write_report
from factorstain.utils.runtime import collect_provenance


PRIMARY = {"lpips": -1, "ssim": 1, "morphology_cosine": 1, "factor_isolation_score": 1}


def _tensor_image(axis, tensor, title: str) -> None:
    axis.imshow(tensor.permute(1, 2, 0).numpy().clip(0, 1))
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _method_grid(previews: dict, destination: Path, title: str) -> None:
    methods = list(previews)
    if not methods:
        return
    rows = min(3, min(len(values) for values in previews.values()))
    fig, axes = plt.subplots(rows, 2 + len(methods), figsize=(3 * (2 + len(methods)), 3 * rows), squeeze=False)
    for i in range(rows):
        first = previews[methods[0]][i]
        _tensor_image(axes[i, 0], first["source"], "Source")
        for j, method in enumerate(methods):
            _tensor_image(axes[i, j + 1], previews[method][i]["generated"], method)
        _tensor_image(axes[i, -1], first["target"], "Real held-out target")
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def _factorial_example(preview: dict, destination: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    _tensor_image(axes[0, 0], preview["source"], "Real source A+X")
    _tensor_image(axes[0, 1], preview["scanner_cf"], "Generated A+Y")
    axes[0, 2].axis("off")
    _tensor_image(axes[1, 0], preview["stain_cf"], "Generated B+X")
    _tensor_image(axes[1, 1], preview["generated"], "FactorStain B+Y\nnever jointly trained")
    _tensor_image(axes[1, 2], preview["target"], "Real B+Y target")
    fig.suptitle("Factorial swap consistency", fontsize=16, weight="bold")
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def _metric_plot(metrics: pd.DataFrame, destination: Path, title: str) -> None:
    protocols = [p for p in ["seen_combo_unseen_morphology", "unseen_combo_seen_morphology", "unseen_combo_unseen_morphology"] if p in metrics.protocol.unique()]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for axis, (metric, direction) in zip(axes, PRIMARY.items()):
        grouped = metrics.groupby(["method", "protocol"])[metric].mean().unstack()
        grouped.reindex(columns=protocols).plot.bar(ax=axis)
        axis.set_title(metric + (" ↓" if direction < 0 else " ↑"))
        axis.tick_params(axis="x", rotation=30)
        axis.set_xlabel("")
        if axis is not axes[0] and axis.get_legend():
            axis.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)))
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 0.95]); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def _metric_ci_plot(summaries: pd.DataFrame, destination: Path) -> None:
    subset = summaries[summaries.protocol.str.startswith("unseen_combo")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for axis, (metric, direction) in zip(axes, PRIMARY.items()):
        values = subset[subset.metric.eq(metric)].groupby("method").agg(mean=("mean", "mean"), low=("ci_low", "mean"), high=("ci_high", "mean")).reset_index()
        errors = np.vstack([values["mean"] - values["low"], values["high"] - values["mean"]])
        axis.bar(values.method, values["mean"], yerr=errors, color="#4263eb", capsize=3)
        axis.set_title(metric + (" ↓" if direction < 0 else " ↑"))
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle("Unseen-combination metrics with 95% confidence intervals", fontsize=16, weight="bold")
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def _failure_grid(previews: list[dict[str, torch.Tensor]], scores: list[float], destination: Path) -> None:
    if not previews:
        return
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), squeeze=False)
    categories = ("Best 5", "Median 5", "Worst 5")
    for position, (preview, score) in enumerate(zip(previews, scores)):
        row, column = divmod(position, 5)
        generated = preview["generated"].permute(1, 2, 0).numpy().clip(0, 1)
        target = preview["target"].permute(1, 2, 0).numpy().clip(0, 1)
        axes[row, column].imshow(np.concatenate([generated, target], axis=1))
        axes[row, column].set_title(f"{categories[row]}\nSSIM={score:.3f}\ngenerated | real", fontsize=8)
        axes[row, column].axis("off")
    fig.suptitle("FactorStain failure audit: best, median, and worst held-out combinations", fontsize=16, weight="bold")
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def _summary_dashboard(figures: Path, decision: dict, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for axis in axes:
        axis.axis("off")
    method_grid = figures / "method_comparison_grid.png"
    metric_plot = figures / "seen_vs_unseen_metrics.png"
    if method_grid.exists():
        axes[0].imshow(Image.open(method_grid)); axes[0].set_title("Held-out combination examples", weight="bold")
    if metric_plot.exists():
        axes[1].imshow(Image.open(metric_plot)); axes[1].set_title("Seen vs unseen performance", weight="bold")
    color = {"GO": "#2b8a3e", "NO_GO": "#c92a2a"}.get(decision["status"], "#f08c00")
    axes[2].text(0.5, 0.68, decision["status"], ha="center", color=color, fontsize=25, weight="bold")
    axes[2].text(0.5, 0.46, f"{decision['primary_metric']}\n{decision['observed']} (threshold {decision['threshold']})", ha="center", fontsize=13)
    axes[2].text(0.5, 0.22, "Decision uses held-out acquisition cells\nand aligned-group bootstrap CIs", ha="center", fontsize=11)
    fig.suptitle(title, fontsize=19, weight="bold"); fig.tight_layout(); fig.savefig(figures / "summary_dashboard.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def _decide_m1(metrics: pd.DataFrame, config: dict) -> tuple[str, float, list[str], list[dict]]:
    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")].copy()
    means = unseen.groupby("method")[list(PRIMARY)].mean()
    baselines = [m for m in ("joint", "parallel") if m in means.index]
    if "factorstain" not in means.index or not baselines:
        return "NO_GO", 0.0, ["Required FactorStain/non-factorized comparison is incomplete."], []
    strongest = {
        metric: (min(baselines, key=lambda method: means.loc[method, metric]) if direction < 0 else max(baselines, key=lambda method: means.loc[method, metric]))
        for metric, direction in PRIMARY.items()
    }
    wins, bootstraps = [], []
    consistent = False
    for metric, direction in PRIMARY.items():
        baseline = strongest[metric]
        if direction * (means.loc["factorstain", metric] - means.loc[baseline, metric]) > 0:
            wins.append(metric)
        comparison_frame = unseen.copy()
        comparison_frame["score"] = direction * comparison_frame[metric]
        result = paired_bootstrap(comparison_frame, "score", "factorstain", baseline, n_bootstrap=config["bootstrap_samples"], seed=config["seed"])
        result["source_metric"] = metric
        bootstraps.append(result)
        if result["ci_low"] > 0:
            consistent = True
    lpips_baseline, fis_baseline = strongest["lpips"], strongest["factor_isolation_score"]
    lpips_relative = (means.loc[lpips_baseline, "lpips"] - means.loc["factorstain", "lpips"]) / max(abs(means.loc[lpips_baseline, "lpips"]), 1e-8)
    fis_relative = (means.loc["factorstain", "factor_isolation_score"] - means.loc[fis_baseline, "factor_isolation_score"]) / max(abs(means.loc[fis_baseline, "factor_isolation_score"]), 0.05)
    important_improvement = max(lpips_relative, fis_relative)
    conditions = len(wins) >= config["decision"]["win_metrics"] and important_improvement >= config["decision"]["relative_improvement"] and consistent
    status = "GO" if conditions else "NO_GO"
    reasons = [f"FactorStain wins {len(wins)}/4 metrics ({wins}) against the strongest non-factorized method for each metric ({strongest}).", f"Best LPIPS/FIS relative improvement is {important_improvement:.1%}.", f"A paired aligned-group bootstrap CI excludes zero: {consistent}."]
    return status, float(important_improvement), reasons, bootstraps


def _decide_m2(metrics: pd.DataFrame, config: dict) -> tuple[str, float, list[str], list[dict]]:
    unseen = metrics[metrics.protocol.str.startswith("unseen_combo")].copy()
    means = unseen.groupby("method")[list(PRIMARY)].mean()
    required = {"factorstain", "reverse", "parallel", "joint"}
    if not required <= set(means.index):
        return "NO_GO", 0.0, [f"Required architecture comparison is incomplete: missing {required - set(means.index)}."], []
    order_wins = [metric for metric, direction in PRIMARY.items() if all(direction * (means.loc["factorstain", metric] - means.loc[other, metric]) > 0 for other in ("reverse", "parallel"))]
    joint_wins = {method: [metric for metric, direction in PRIMARY.items() if direction * (means.loc[method, metric] - means.loc["joint", metric]) > 0] for method in ("factorstain", "reverse")}
    factorized_beats_joint = any(len(wins) >= 3 for wins in joint_wins.values())
    bootstraps = []
    for other in ("reverse", "parallel", "joint"):
        for metric, direction in PRIMARY.items():
            comparison_frame = unseen.copy(); comparison_frame["score"] = direction * comparison_frame[metric]
            result = paired_bootstrap(comparison_frame, "score", "factorstain", other, n_bootstrap=config["bootstrap_samples"], seed=config["seed"])
            result["source_metric"] = metric; bootstraps.append(result)
    if not factorized_beats_joint:
        status = "NO_GO"
    elif len(order_wins) >= config["decision"]["win_metrics"]:
        status = "GO"
    else:
        status = "GO_WITH_SCOPE_REDUCTION"
    reasons = [f"S→Q beats both Q→S and parallel on {len(order_wins)}/4 metrics ({order_wins}).", f"Factorized metric wins versus joint are {joint_wins}; substantive superiority: {factorized_beats_joint}."]
    return status, float(len(order_wins)), reasons, bootstraps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    out = prepare_output(config)
    index = pd.read_parquet(out / "plism_index_with_splits.parquet")
    split = json.loads((out / "splits" / f"combination_split_seed{config['seed']}.json").read_text(encoding="utf-8"))
    limit = 12 if config["fast_dev_run"] else config.get("max_eval_per_protocol", 5000)
    pairs = build_evaluation_pairs(index, split, config["seed"], limit)
    if pairs.empty:
        raise RuntimeError("No evaluation pairs could be formed for the held-out combination protocols")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Renderer evaluation requires CUDA; CPU fallback is disabled")
    # Use the first genuinely available requested pathology FM, never an unrelated substitute.
    m0 = Path(config["paths"]["outputs_root"]) / "m0_probe"
    foundation = probes = None
    model_used = None
    train_cells = {(item["stain_id"], item["scanner_id"]) for item in split["train_cells"]}
    for candidate in ("uni", "virchow2"):
        cache = m0 / "features" / candidate / "features.h5"
        if not cache.exists():
            continue
        try:
            foundation = load_foundation_model(candidate, device)
            probes = RealAcquisitionProbes(cache, index, train_cells)
            model_used = candidate
            break
        except BlockedModelAccess:
            continue
    if foundation is None or probes is None:
        raise RuntimeError("No accessible UNI/Virchow2 feature cache is available for independent evaluation. Complete M0 model access first.")
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
    except Exception as exc:
        raise RuntimeError(f"LPIPS is mandatory for renderer decisions and could not be initialized: {exc}") from exc

    all_metrics, previews, parameter_counts = [], {}, {}
    for model_name in config["models"]:
        checkpoint = out / "checkpoints" / model_name / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Required best checkpoint is missing: {checkpoint}")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parameter_counts[model_name] = int(sum(value.numel() for value in checkpoint_payload["model"].values()))
        frame, preview = evaluate_model(model_name, checkpoint, config, index, pairs, foundation, probes, lpips_model, device)
        all_metrics.append(frame)
        previews[model_name] = preview
    metrics = pd.concat(all_metrics, ignore_index=True)
    summaries = []
    for (method, protocol), group in metrics.groupby(["method", "protocol"]):
        for metric in [*PRIMARY, "psnr", "he_distance", "stain_target_correct", "scanner_target_correct"]:
            summary = summarize_samples(group[metric])
            summaries.append({"method": method, "protocol": protocol, "metric": metric, **summary})
    summaries_frame = pd.DataFrame(summaries)
    summaries_frame.to_csv(out / "statistical_comparisons.csv", index=False)

    if config["milestone"] == "m1_factorial":
        status, observed, reasons, bootstraps = _decide_m1(metrics, config)
        primary_metric, threshold = "best relative unseen-combination improvement", config["decision"]["relative_improvement"]
        title = "M1 — Factorial compositional pilot"
    else:
        status, observed, reasons, bootstraps = _decide_m2(metrics, config)
        primary_metric, threshold = "principal metrics beating reverse and parallel", float(config["decision"]["win_metrics"])
        title = "M2 — Full ordered renderer ablation"
    pd.DataFrame(bootstraps).to_csv(out / "paired_bootstrap_comparisons.csv", index=False)
    if config["fast_dev_run"]:
        status = "GO_WITH_SCOPE_REDUCTION"
    decision = write_decision(out, status, primary_metric, observed, threshold, reasons, "Proceed to the next milestone." if status != "NO_GO" else "Stop and inspect held-out-combination failures before expanding the model.", not config["fast_dev_run"])
    summary_payload = {
        "evaluation_foundation_model": model_used,
        "n_pairs": int(len(pairs)),
        "aggregate": summaries,
        "paired_bootstrap": bootstraps,
        "model_parameter_counts": parameter_counts,
        "provenance": collect_provenance(config["seed"], {"evaluation_pairs": len(pairs)}),
    }
    write_metrics(out, metrics.to_dict("records"), summary_payload)
    unseen_means = metrics[metrics.protocol.str.startswith("unseen_combo")].groupby("method")[list(PRIMARY)].mean().round(5).to_dict("index")
    write_report(out, title, {"evaluation_foundation_model": model_used, "evaluation_pairs": len(pairs), "unseen_combination_means": unseen_means, "model_parameter_counts": parameter_counts}, decision)

    figures = out / "figures"
    _metric_plot(metrics, figures / "seen_vs_unseen_metrics.png", "Seen versus unseen acquisition performance")
    _metric_ci_plot(summaries_frame, figures / "metric_bar_with_CI.png")
    _method_grid(previews, figures / "method_comparison_grid.png", "Never-jointly-observed acquisition combinations")
    if "factorstain" in previews and previews["factorstain"]:
        _factorial_example(previews["factorstain"][0], figures / "factorial_2x2_examples.png")
        _factorial_example(previews["factorstain"][0], figures / "factor_isolation_examples.png")
        _factorial_example(previews["factorstain"][0], figures / "morphology_preservation.png")
    if config["milestone"] == "m2_renderer":
        _method_grid({key: previews[key] for key in ("factorstain", "reverse")}, figures / "ordered_vs_reverse_grid.png", "Physical order versus reverse order")
        _method_grid({key: previews[key] for key in ("factorstain", "parallel")}, figures / "ordered_vs_parallel_grid.png", "Ordered versus parallel factor injection")
        _method_grid(previews, figures / "unseen_combo_large_examples.png", "Unseen-combination examples")
        factor_rows = metrics[(metrics.method == "factorstain") & metrics.protocol.str.startswith("unseen_combo")].sort_values("ssim", ascending=False)
        count = min(5, len(factor_rows) // 3)
        if count:
            middle = max(0, len(factor_rows) // 2 - count // 2)
            selected_metrics = pd.concat([factor_rows.head(count), factor_rows.iloc[middle : middle + count], factor_rows.tail(count)]).reset_index(drop=True)
            selected_pairs = selected_metrics[["source_index", "target_index", "protocol", "aligned_group_id"]]
            _, failure_previews = evaluate_model("factorstain", out / "checkpoints" / "factorstain" / "best.pt", config, index, selected_pairs, foundation, probes, lpips_model, device, preview_limit=len(selected_pairs))
            # Always render a fixed 3×5 audit; duplicate only in tiny invalid development runs.
            while len(failure_previews) < 15:
                failure_previews.extend(failure_previews[: 15 - len(failure_previews)])
            scores = selected_metrics.ssim.tolist()
            while len(scores) < 15:
                scores.extend(scores[: 15 - len(scores)])
            _failure_grid(failure_previews[:15], scores[:15], figures / "failure_cases.png")
        fis = metrics[metrics.protocol.str.startswith("unseen")].groupby("method").factor_isolation_score.agg(["mean", "sem"])
        fig, axis = plt.subplots(figsize=(8, 5)); axis.bar(fis.index, fis["mean"], yerr=1.96 * fis["sem"], color="#4263eb"); axis.set_title("Factor Isolation Score on unseen combinations"); axis.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(figures / "factor_isolation_bar.png", dpi=180); plt.close(fig)
    _summary_dashboard(figures, decision, title)
    update_master(config["paths"]["outputs_root"])


if __name__ == "__main__":
    main()
