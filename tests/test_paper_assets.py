from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image

from factorstain.models.dinov3 import save_feature_cache
from factorstain.training.renderer import build_renderer
from factorstain.visualization.paper_assets import (
    PaperAssetInputError,
    load_paper_asset_config,
    require_inputs,
)
from scripts.build_method_sources import _find_factorial_example, _select_ranked_rows

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs" / "paper_assets.yaml"


def test_paper_asset_config_honors_factorstain_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FACTORSTAIN_ROOT", str(tmp_path))
    monkeypatch.delenv("OUTPUTS_ROOT", raising=False)
    _, paths = load_paper_asset_config(CONFIG)
    assert paths.project_root == tmp_path.resolve()
    assert (
        paths.motivation_output
        == (tmp_path / "outputs" / "paper_assets" / "motivation_sources").resolve()
    )
    assert (
        paths.method_output
        == (tmp_path / "outputs" / "paper_assets" / "method_sources").resolve()
    )


def test_missing_inputs_report_actionable_command(tmp_path: Path) -> None:
    missing = tmp_path / "outputs" / "m1_factorial" / "metrics.json"
    try:
        require_inputs([missing], "bash shell/m1_factorial.sh")
    except PaperAssetInputError as error:
        message = str(error)
    else:
        raise AssertionError("Missing scientific input did not stop generation")
    assert str(missing) in message
    assert "bash shell/m1_factorial.sh" in message


def test_dry_run_reads_no_data_and_writes_no_assets(tmp_path: Path) -> None:
    for script in ("build_motivation_sources.py", "build_method_sources.py"):
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "scripts" / script),
                "--config",
                str(CONFIG),
                "--project-root",
                str(tmp_path),
                "--dry-run",
            ],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "No paper assets were generated." in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_factorial_example_uses_saved_split_and_real_complete_grid(
    tmp_path: Path,
) -> None:
    rows = []
    for stain, scanner, shade in (
        ("A", "X", 70),
        ("A", "Y", 100),
        ("B", "X", 130),
        ("B", "Y", 160),
    ):
        path = tmp_path / f"{stain}_{scanner}.png"
        Image.new("RGB", (64, 64), (shade, 40, 120)).save(path)
        rows.append(
            {
                "sample_id": f"{stain}_{scanner}",
                "image_path": str(path),
                "aligned_group_id": "group-1",
                "tissue_type": "fixture",
                "stain_id": stain,
                "scanner_id": scanner,
                "morphology_split": "test",
            }
        )
    split = {
        "train_cells": [
            {"stain_id": "A", "scanner_id": "X"},
            {"stain_id": "A", "scanner_id": "Y"},
            {"stain_id": "B", "scanner_id": "X"},
        ],
        "validation_cells": [],
        "test_cells": [{"stain_id": "B", "scanner_id": "Y"}],
    }
    selected = _find_factorial_example(pd.DataFrame(rows), split)
    assert selected["aligned_group_id"] == "group-1"
    assert (selected["stain_a"], selected["stain_b"]) == ("A", "B")
    assert (selected["scanner_x"], selected["scanner_y"]) == ("X", "Y")


def test_best_median_hard_selection_uses_saved_metric() -> None:
    metrics = pd.DataFrame(
        [
            {
                "method": "factorstain",
                "protocol": "unseen_combo_seen_morphology",
                "source_index": index,
                "target_index": index + 10,
                "aligned_group_id": f"g{index}",
                "morphology_preservation": value,
            }
            for index, value in enumerate((0.9, 0.7, 0.4, 0.1))
        ]
    )
    selected = _select_ranked_rows(metrics, "morphology_preservation")
    assert selected["best"].morphology_preservation == 0.9
    assert selected["median"].morphology_preservation in {0.7, 0.4}
    assert selected["hard"].morphology_preservation == 0.1


def test_method_generator_runs_on_temporary_fixture(tmp_path: Path) -> None:
    m1 = tmp_path / "m1"
    for directory in (
        m1 / "metadata",
        m1 / "splits",
        m1 / "tables",
        m1 / "checkpoints" / "joint",
        m1 / "checkpoints" / "parallel",
        m1 / "checkpoints" / "factorstain",
    ):
        directory.mkdir(parents=True)
    (m1 / "config_resolved.yaml").write_text(
        yaml.safe_dump({"image_size": 32, "model_width": 8}), encoding="utf-8"
    )
    split = {
        "seed": 42,
        "stains": ["A", "B"],
        "scanners": ["X", "Y"],
        "train_cells": [
            {"stain_id": "A", "scanner_id": "X"},
            {"stain_id": "A", "scanner_id": "Y"},
            {"stain_id": "B", "scanner_id": "X"},
        ],
        "validation_cells": [],
        "test_cells": [{"stain_id": "B", "scanner_id": "Y"}],
    }
    (m1 / "splits" / "combination_split_seed42.json").write_text(
        json.dumps(split), encoding="utf-8"
    )
    index_rows = []
    cell_order = (("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y"))
    for group in range(3):
        for cell_position, (stain, scanner) in enumerate(cell_order):
            path = tmp_path / f"g{group}_{stain}_{scanner}.png"
            Image.new(
                "RGB",
                (64, 64),
                (70 + group * 25, 45 + cell_position * 30, 135),
            ).save(path)
            index_rows.append(
                {
                    "sample_id": f"g{group}_{stain}_{scanner}",
                    "image_path": str(path),
                    "aligned_group_id": f"g{group}",
                    "tissue_type": f"fixture-{group}",
                    "stain_id": stain,
                    "scanner_id": scanner,
                    "morphology_split": "test",
                }
            )
    pd.DataFrame(index_rows).to_parquet(
        m1 / "metadata" / "plism_index_with_splits.parquet", index=False
    )
    metric_rows = []
    for method_number, method in enumerate(("joint", "parallel", "factorstain")):
        for group in range(3):
            offset = group * 4
            for protocol, target_offset, adjustment in (
                ("unseen_combo_unseen_morphology", 3, 0.0),
                ("seen_combo_unseen_morphology", 1, 0.05),
            ):
                value = 0.9 - group * 0.25 - method_number * 0.03 + adjustment
                metric_rows.append(
                    {
                        "method": method,
                        "protocol": protocol,
                        "source_index": offset,
                        "target_index": offset + target_offset,
                        "aligned_group_id": f"g{group}",
                        "scanner_target_accuracy": max(0.0, value - 0.02),
                        "stain_target_accuracy": max(0.0, value - 0.04),
                        "morphology_preservation": value,
                        "factor_isolation_score": max(0.0, value - 0.06),
                    }
                )
    pd.DataFrame(metric_rows).to_csv(
        m1 / "tables" / "per_sample_metrics.csv", index=False
    )
    pd.DataFrame({"fixture": [1]}).to_csv(
        m1 / "tables" / "overall_metrics.csv", index=False
    )
    (m1 / "metrics.json").write_text(
        json.dumps({"fast_dev_run": False, "decision_valid": True}),
        encoding="utf-8",
    )
    for method in ("joint", "parallel", "factorstain"):
        model = build_renderer(method, 2, 2, width=8)
        torch.save(
            {"model": model.state_dict()},
            m1 / "checkpoints" / method / "best.pt",
        )

    output = tmp_path / "generated"
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "build_method_sources.py"),
            "--config",
            str(CONFIG),
            "--project-root",
            str(tmp_path),
            "--m1-dir",
            str(m1),
            "--output-dir",
            str(output),
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "manifest.json").is_file()
    assert (output / "06_joint_parallel_ordered_comparison.svg").is_file()
    assert (output / "10_method_key_equations.png").is_file()
    assert not (tmp_path / "outputs" / "paper_assets").exists()


def test_motivation_generator_runs_on_temporary_fixture(tmp_path: Path) -> None:
    audit, mmd, images = tmp_path / "audit", tmp_path / "mmd", tmp_path / "images"
    for directory in (audit / "metadata", audit / "features", mmd / "tables", images):
        directory.mkdir(parents=True)
    scanners = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]
    stains = [f"H&E-{index}" for index in range(1, 6)]
    samples = []
    for group, tissue in enumerate(("breast", "colon", "lymph")):
        for stain_position, stain in enumerate(stains):
            for scanner_position, scanner in enumerate(scanners):
                sample_id = f"g{group}_{stain_position}_{scanner_position}"
                path = images / f"{sample_id}.png"
                Image.new(
                    "RGB",
                    (64, 64),
                    (
                        120 + stain_position * 10,
                        45 + scanner_position * 8,
                        135 + group * 10,
                    ),
                ).save(path)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "image_path": str(path),
                        "aligned_group_id": f"g{group}",
                        "stain_id": stain,
                        "scanner_id": scanner,
                        "tissue_type": tissue,
                        "tissue_fraction": 0.8 - group * 0.05,
                        "dataset": "plism",
                    }
                )
    sample_frame = pd.DataFrame(samples)
    sample_frame.to_parquet(audit / "metadata" / "plism_samples.parquet", index=False)
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(len(sample_frame), 24, generator=generator).numpy()
    for group in range(3):
        features[group * 35 : (group + 1) * 35, group * 3 : group * 3 + 3] += 4
    save_feature_cache(
        audit / "features" / "plism_dinov3.npz",
        features,
        sample_frame.sample_id.to_numpy(),
        {
            "model_name": "fixture-dinov3",
            "feature_dimension": 24,
            "pooling_strategy": "fixture",
        },
    )
    coordinates = torch.randn(len(sample_frame), 2, generator=generator).numpy()
    pd.DataFrame(
        {
            "sample_id": sample_frame.sample_id,
            "dataset": "plism",
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
        }
    ).to_parquet(audit / "metadata" / "feature_projections.parquet", index=False)
    categories = (
        "same morphology / same stain / different scanner",
        "different morphology / same stain / same scanner",
        "different morphology / same stain / different scanner",
    )
    distance_rows = []
    for category_position, category in enumerate(categories):
        for position in range(15):
            distance_rows.append(
                {
                    "category": category,
                    "distance": 0.08 + category_position * 0.12 + position * 0.002,
                    "left_sample_id": sample_frame.sample_id.iloc[position],
                    "right_sample_id": sample_frame.sample_id.iloc[position + 35],
                }
            )
    pd.DataFrame(distance_rows).to_csv(
        audit / "metadata" / "plism_controlled_feature_distances.csv", index=False
    )
    probes = [
        {
            "dataset": "plism",
            "target": target,
            "balanced_accuracy": value,
            "chance": chance,
            "group_overlap_count": 0,
        }
        for target, value, chance in (
            ("scanner", 0.876, 1 / 7),
            ("stain", 0.563, 1 / 5),
            ("tissue", 0.635, 1 / 3),
        )
    ]
    (audit / "metrics.json").write_text(
        json.dumps({"audit_valid": True, "fast_dev_run": False, "probes": probes}),
        encoding="utf-8",
    )

    def matrix(labels: list[str], scale: float) -> pd.DataFrame:
        values = torch.zeros(len(labels), len(labels)).numpy()
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                values[left, right] = values[right, left] = (
                    scale * (1 + left + right) / 100
                )
        return pd.DataFrame(values, index=labels, columns=labels)

    scanner_matrix = matrix(scanners, 1.0)
    stain_matrix = matrix(stains, 1.4)
    scanner_matrix.to_csv(mmd / "tables" / "scanner_mmd_controlled.csv")
    stain_matrix.to_csv(mmd / "tables" / "stain_mmd_balanced.csv")
    (mmd / "metrics.json").write_text(
        json.dumps(
            {
                "audit_valid": True,
                "fast_dev_run": False,
                "domains": {"scanners": scanners, "stains": stains},
                "extrema": {
                    "controlled_scanner_largest": {
                        "domain_a": "S60",
                        "domain_b": "SQ",
                        "mmd2": float(scanner_matrix.to_numpy().max()),
                    },
                    "balanced_stain_largest": {
                        "domain_a": "H&E-4",
                        "domain_b": "H&E-5",
                        "mmd2": float(stain_matrix.to_numpy().max()),
                    },
                },
                "correlations": {
                    "scanner_centroid_vs_controlled_mmd_spearman": {"rho": 0.742}
                },
                "stain_balance_fallback_pairs": [],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "motivation_generated"
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "build_motivation_sources.py"),
            "--config",
            str(CONFIG),
            "--project-root",
            str(tmp_path),
            "--audit-dir",
            str(audit),
            "--mmd-dir",
            str(mmd),
            "--output-dir",
            str(output),
        ],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "02_aligned_tissue_across_stains.png").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["assets"]) == 12
    assert all(asset["dataset"] == "PLISM" for asset in manifest["assets"])
    assert not (tmp_path / "outputs" / "paper_assets").exists()
