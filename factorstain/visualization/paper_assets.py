from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


class PaperAssetInputError(FileNotFoundError):
    """Raised when a completed experiment artifact is unavailable."""


@dataclass(frozen=True)
class PaperAssetPaths:
    project_root: Path
    outputs_root: Path
    motivation_audit: Path
    motivation_mmd: Path
    motivation_output: Path
    method_m1: Path
    method_output: Path


def _resolve_under(base: Path, value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else base / path


def load_paper_asset_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], PaperAssetPaths]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    root_value = (
        project_root
        or os.getenv("FACTORSTAIN_ROOT")
        or config.get("project_root")
        or Path.cwd()
    )
    root = Path(root_value).expanduser().resolve()
    outputs_value = os.getenv("OUTPUTS_ROOT") or config.get("outputs_root", "outputs")
    outputs = _resolve_under(root, outputs_value).resolve()
    motivation = config.get("motivation", {})
    method = config.get("method", {})
    paths = PaperAssetPaths(
        project_root=root,
        outputs_root=outputs,
        motivation_audit=_resolve_under(
            outputs, motivation.get("domain_audit", "m_minus1_domain_audit")
        ).resolve(),
        motivation_mmd=_resolve_under(
            outputs, motivation.get("mmd", "m_minus1_mmd")
        ).resolve(),
        motivation_output=_resolve_under(
            outputs,
            motivation.get("output", "paper_assets/motivation_sources"),
        ).resolve(),
        method_m1=_resolve_under(outputs, method.get("m1", "m1_factorial")).resolve(),
        method_output=_resolve_under(
            outputs, method.get("output", "paper_assets/method_sources")
        ).resolve(),
    )
    return config, paths


def motivation_required_inputs(paths: PaperAssetPaths) -> list[Path]:
    audit, mmd = paths.motivation_audit, paths.motivation_mmd
    return [
        audit / "metadata" / "plism_samples.parquet",
        audit / "metadata" / "feature_projections.parquet",
        audit / "metadata" / "plism_controlled_feature_distances.csv",
        audit / "features" / "plism_dinov3.npz",
        audit / "metrics.json",
        mmd / "metrics.json",
        mmd / "tables" / "scanner_mmd_controlled.csv",
        mmd / "tables" / "stain_mmd_balanced.csv",
    ]


def method_required_inputs(paths: PaperAssetPaths, seed: int = 42) -> list[Path]:
    m1 = paths.method_m1
    return [
        m1 / "config_resolved.yaml",
        m1 / "metadata" / "plism_index_with_splits.parquet",
        m1 / "splits" / f"combination_split_seed{seed}.json",
        m1 / "tables" / "per_sample_metrics.csv",
        m1 / "tables" / "overall_metrics.csv",
        m1 / "metrics.json",
        m1 / "checkpoints" / "joint" / "best.pt",
        m1 / "checkpoints" / "parallel" / "best.pt",
        m1 / "checkpoints" / "factorstain" / "best.pt",
    ]


def require_inputs(paths: Iterable[Path], run_command: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return
    listing = "\n".join(f"  {path}" for path in missing)
    raise PaperAssetInputError(
        "ERROR:\nRequired experiment output not found:\n"
        f"{listing}\n\nRun:\n{run_command}"
    )


def print_dry_run(
    label: str,
    config_path: Path,
    inputs: Iterable[Path],
    output: Path,
    expected_files: Iterable[str],
) -> None:
    print(f"DRY RUN - {label}")
    print(f"config: {config_path.resolve()}")
    print("expected inputs (existence not required in dry run):")
    for path in inputs:
        print(f"  {path}")
    print(f"expected output directory: {output}")
    print("expected generated files:")
    for name in expected_files:
        print(f"  {output / name}")
    print("No datasets, checkpoints, cached features, or scientific outputs were read.")
    print("No paper assets were generated.")


def configure_paper_style(dpi: int = 300) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": dpi,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(
    fig: plt.Figure,
    destination: Path,
    dpi: int = 300,
    vector_suffixes: Iterable[str] = (),
) -> list[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    outputs = [destination]
    for suffix in vector_suffixes:
        companion = destination.with_suffix(suffix)
        fig.savefig(companion, bbox_inches="tight", pad_inches=0.03)
        outputs.append(companion)
    plt.close(fig)
    return outputs


def natural_key(value: object) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)
