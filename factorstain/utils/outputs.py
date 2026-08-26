from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import save_resolved_config
from .runtime import atomic_json_dump


MILESTONES = [
    ("m0_probe", "M0"),
    ("m1_factorial", "M1"),
    ("m2_renderer", "M2"),
    ("m3_attribution", "M3"),
    ("m4_adapter", "M4"),
    ("m5_external", "M5"),
]


def prepare_output(config: dict) -> Path:
    out = Path(config["paths"]["outputs_root"]) / config["milestone"]
    for child in ("logs", "figures", "checkpoints", "splits"):
        (out / child).mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, out / "config_resolved.yaml")
    return out


def write_metrics(out: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    atomic_json_dump(summary, out / "metrics.json")


def write_decision(
    out: Path,
    status: str,
    primary_metric: str,
    observed: float | None,
    threshold: float | None,
    reasons: list[str],
    recommended_next_step: str,
    decision_valid: bool,
) -> dict[str, Any]:
    if status not in {"GO", "NO_GO", "GO_WITH_SCOPE_REDUCTION"}:
        raise ValueError(f"Invalid decision status: {status}")
    if not decision_valid:
        reasons = ["FAST_DEV_RUN is a pipeline check, not a scientific experiment.", *reasons]
    payload = {
        "status": status,
        "primary_metric": primary_metric,
        "observed": observed,
        "threshold": threshold,
        "reasons": reasons,
        "recommended_next_step": recommended_next_step,
        "decision_valid": decision_valid,
    }
    atomic_json_dump(payload, out / "GO_NOGO.json")
    return payload


def write_report(out: Path, title: str, summary: dict[str, Any], decision: dict[str, Any]) -> None:
    metrics = "\n".join(f"- **{key}**: {value}" for key, value in summary.items())
    reasons = "\n".join(f"- {reason}" for reason in decision["reasons"]) or "- None"
    text = f"""# {title}

## Decision

**{decision['status']}** (decision valid: `{str(decision['decision_valid']).lower()}`)

Primary metric: `{decision['primary_metric']}` = `{decision['observed']}`; threshold = `{decision['threshold']}`.

## Measured summary

{metrics}

## Reasons

{reasons}

## Recommended next step

{decision['recommended_next_step']}
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def update_master(outputs_root: str | Path) -> None:
    root = Path(outputs_root)
    records = []
    for directory, label in MILESTONES:
        decision_file = root / directory / "GO_NOGO.json"
        if decision_file.exists():
            decision = json.loads(decision_file.read_text(encoding="utf-8"))
            records.append((label, decision["status"], decision["primary_metric"], decision["observed"], decision.get("decision_valid", True)))
        else:
            records.append((label, "PENDING", "-", None, False))

    colors = {"GO": "#2b8a3e", "GO_WITH_SCOPE_REDUCTION": "#f08c00", "NO_GO": "#c92a2a", "PENDING": "#868e96"}
    fig, axis = plt.subplots(figsize=(14, 5.5))
    axis.axis("off")
    for i, (label, status, metric, value, valid) in enumerate(records):
        x = (i + 0.5) / len(records)
        axis.add_patch(plt.Rectangle((x - 0.075, 0.38), 0.15, 0.32, color=colors[status], alpha=0.92))
        axis.text(x, 0.61, label, color="white", weight="bold", size=17, ha="center")
        short = status.replace("GO_WITH_SCOPE_REDUCTION", "SCOPE\nREDUCED")
        axis.text(x, 0.49, short, color="white", weight="bold", size=10, ha="center")
        value_text = "—" if value is None else f"{value:.3f}" if isinstance(value, (float, int)) else str(value)
        axis.text(x, 0.27, f"{metric}\n{value_text}", size=9, ha="center", va="top")
        if not valid and status != "PENDING":
            axis.text(x, 0.13, "DEV RUN", color="#c92a2a", size=9, weight="bold", ha="center")
    axis.set_title("FactorStain — Master Experimental Dashboard", fontsize=20, weight="bold", pad=18)
    fig.savefig(root / "MASTER_DASHBOARD.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = ["# FactorStain Master Report", "", "| Milestone | Status | Primary metric | Observed | Valid |", "|---|---|---|---:|---|"]
    for label, status, metric, value, valid in records:
        lines.append(f"| {label} | {status} | {metric} | {value if value is not None else '—'} | {valid} |")
    (root / "MASTER_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
