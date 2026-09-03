from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from factorstain.baselines.classical import ScannerTransformBank
from factorstain.baselines.features import FEATMAPHarmonizer, ScanGenHarmonizer
from factorstain.baselines.registry import BASELINES, resolve_methods
from factorstain.data.splits import build_combination_split, group_train_val_test_split
from factorstain.utils.config import load_config
from scripts import (
    aggregate_sota_benchmark,
    prepare_sota_benchmark,
    run_sota_method,
)
from third_party import fetch_baselines


def _frame(groups: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "aligned_group_id": f"g{group:02d}",
                "tissue_type": f"t{group % 3}",
                "stain_id": stain,
                "scanner_id": scanner,
                "sample_id": f"g{group:02d}-{stain}-{scanner}",
                "image_id": f"g{group:02d}-{stain}-{scanner}",
                "image_exists": True,
            }
            for group in range(groups)
            for stain in ("A", "B", "C")
            for scanner in ("X", "Y", "Z")
        ]
    )


def test_registry_scientific_capabilities_and_tiers():
    assert BASELINES["histofs"].primary_track == "D"
    assert not BASELINES["histofs"].supports_unseen_composition
    assert BASELINES["pix2pix"].supports_scanner
    assert not BASELINES["pix2pix"].supports_stain
    assert BASELINES["igan"].availability == "NOT_REPRODUCIBLE_FROM_AVAILABLE_RESOURCES"
    tier_one = resolve_methods(tier="1")
    assert {"macenko", "stainnet", "histaugan", "factorstain"} <= set(tier_one)
    assert resolve_methods(methods="macenko,reinhard,macenko") == [
        "macenko",
        "reinhard",
    ]


def test_single_seed_runtime_override_preserves_declared_protocol(
    tmp_path, monkeypatch
):
    paths = tmp_path / "paths.yaml"
    config = tmp_path / "benchmark.yaml"
    paths.write_text(
        f"project_root: '{tmp_path.as_posix()}'\n"
        f"outputs_root: '{(tmp_path / 'outputs').as_posix()}'\n"
        f"hf_home: '{(tmp_path / 'hf').as_posix()}'\n",
        encoding="utf-8",
    )
    config.write_text("seed: 42\nseeds: [42, 43, 44]\n", encoding="utf-8")
    monkeypatch.setenv("SOTA_SEEDS", "42")
    loaded = load_config(config, paths)
    assert loaded["seeds"] == [42, 43, 44]
    assert loaded["run_seeds"] == [42]

    monkeypatch.setenv("SOTA_SEEDS", "99")
    with pytest.raises(ValueError, match="not declared"):
        load_config(config, paths)


def test_scanner_lut_uses_paired_color_response(tmp_path):
    source_path = tmp_path / "source.png"
    target_path = tmp_path / "target.png"
    source = np.full((24, 24, 3), [80, 110, 140], dtype=np.uint8)
    target = np.clip(source.astype(int) + [18, -8, 12], 0, 255).astype(np.uint8)
    Image.fromarray(source).save(source_path)
    Image.fromarray(target).save(target_path)
    pairs = pd.DataFrame(
        [
            {
                "source_scanner": "X",
                "target_scanner": "Y",
                "source_path": str(source_path),
                "target_path": str(target_path),
            }
        ]
    )
    bank = ScannerTransformBank("lut", grid_size=9)
    bank.fit(pairs, image_size=24, seed=42)
    corrected = bank.apply(source, "X", "Y")
    before = np.abs(source.astype(float) - target).mean()
    after = np.abs(corrected.astype(float) - target).mean()
    assert after < before


def test_feature_harmonizers_fit_training_pairs_only():
    metadata = pd.DataFrame(
        [
            {
                "aligned_group_id": f"g{group}",
                "stain_id": "H&E",
                "scanner_id": scanner,
                "tissue_type": f"t{group % 2}",
                "is_training": True,
            }
            for group in range(4)
            for scanner in ("X", "Y")
        ]
    )
    base = np.asarray([[group, group % 2, 0.5, -0.5] for group in range(4)])
    features = np.vstack(
        [
            base[group] + (np.asarray([1.0, -0.5, 0.25, 0.0]) if scanner == "Y" else 0)
            for group in range(4)
            for scanner in ("X", "Y")
        ]
    ).astype(np.float32)
    before = np.mean(np.linalg.norm(features[::2] - features[1::2], axis=1))
    featmap = FEATMAPHarmonizer(ridge=1e-4)
    featmap.fit(features, metadata)
    corrected = featmap.transform(features, metadata)
    after = np.mean(np.linalg.norm(corrected[::2] - corrected[1::2], axis=1))
    assert after < before

    scangen = ScanGenHarmonizer(hidden=4, epochs=1, seed=42)
    scangen.fit(features, metadata)
    projected = scangen.transform(features, metadata)
    assert projected.shape == features.shape
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0, atol=1e-5)


def test_macro_combination_bootstrap_clusters_groups_and_balances_cells():
    frame = pd.DataFrame(
        [
            {
                "method": method,
                "combination_id": combination,
                "aligned_group_id": group,
                "utility": base + (0.2 if method == "ours" else 0.0),
            }
            for combination, base in (("AxY", 0.2), ("BxZ", 0.8))
            for group in ("g1", "g2", "g3")
            for method in ("ours", "external")
        ]
    )
    result = aggregate_sota_benchmark._paired_macro_combination_bootstrap(
        frame,
        "utility",
        "ours",
        "external",
        n_bootstrap=100,
        seed=42,
        confidence=0.95,
    )
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci_low"] > 0
    assert result["n_combinations"] == 2


def test_fetcher_verifies_non_git_vendored_snapshot(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
    subprocess.run(
        ["git", "config", "user.email", "benchmark@example.invalid"],
        cwd=upstream,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "FactorStain test"],
        cwd=upstream,
        check=True,
    )
    (upstream / "model.py").write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "pinned"], cwd=upstream, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    vendor_root = tmp_path / "vendor"
    destination = vendor_root / "Method"
    destination.mkdir(parents=True)
    (destination / "model.py").write_text("PINNED = True\n", encoding="utf-8")
    spec = SimpleNamespace(official_repository=str(upstream), source_commit=commit)
    fetch_baselines._verify_vendored_snapshot(destination, spec, vendor_root)
    assert fetch_baselines._valid_marker(destination, spec)

    (destination / "model.py").write_text("PINNED = False\n", encoding="utf-8")
    assert not fetch_baselines._valid_marker(destination, spec)
    with pytest.raises(RuntimeError, match="does not exactly match"):
        fetch_baselines._verify_vendored_snapshot(destination, spec, vendor_root)


def test_fetcher_initializes_existing_empty_gitlink(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=upstream, check=True)
    subprocess.run(
        ["git", "config", "user.email", "benchmark@example.invalid"],
        cwd=upstream,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "FactorStain test"],
        cwd=upstream,
        check=True,
    )
    (upstream / "model.py").write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "pinned"], cwd=upstream, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    vendor_root = tmp_path / "vendor"
    destination = vendor_root / "Method"
    destination.mkdir(parents=True)
    spec = SimpleNamespace(official_repository=str(upstream), source_commit=commit)
    monkeypatch.setitem(fetch_baselines.DESTINATIONS, "fake", destination.name)
    monkeypatch.setitem(
        sys.modules["factorstain.baselines.registry"].BASELINES, "fake", spec
    )

    fetch_baselines.fetch("fake", vendor_root)

    assert (destination / "model.py").read_text(encoding="utf-8") == "PINNED = True\n"
    assert fetch_baselines._valid_marker(destination, spec)


def test_fast_sota_output_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: self.to_pickle(path),
    )
    monkeypatch.setattr(pd, "read_parquet", pd.read_pickle)
    frame = _frame()
    frame["morphology_split"] = group_train_val_test_split(frame, seed=42)
    image_root = tmp_path / "images"
    image_root.mkdir()
    paths = []
    for position, row in frame.iterrows():
        path = image_root / f"{position}.png"
        color = (
            60 + 25 * (ord(row.stain_id) - ord("A")),
            90 + 15 * (ord(row.scanner_id) - ord("X")),
            130 + position % 20,
        )
        Image.new("RGB", (20, 20), color).save(path)
        paths.append(str(path))
    frame["image_path"] = paths
    split = build_combination_split(frame, seed=42)
    split_path = tmp_path / "combination.json"
    morphology_path = tmp_path / "morphology.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    morphology_path.write_text(
        json.dumps(
            {
                "seed": 42,
                "groups": frame.groupby("aligned_group_id")
                .morphology_split.first()
                .to_dict(),
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.parquet"
    frame.to_parquet(index_path, index=False)
    outputs = tmp_path / "outputs"
    config = {
        "milestone": "m1_sota_benchmark",
        "seed": 42,
        "seeds": [42, 43, 44],
        "fast_dev_run": True,
        "paths": {
            "project_root": str(tmp_path),
            "outputs_root": str(outputs),
        },
        "combination_split": str(split_path),
        "morphology_split": str(morphology_path),
        "m1_index": str(index_path),
        "m1_milestone": "m1_factorial",
        "m1_training_config": "configs/m1_factorial.yaml",
        "dinov3_cache": str(tmp_path / "missing_features.npz"),
        "dinov3_model_name": "fake",
        "dinov3_dtype": "float32",
        "image_size": 16,
        "batch_size": 2,
        "num_workers": 0,
        "max_eval_per_protocol": 50,
        "fast_dev_eval_per_protocol": 2,
        "prototype_pool_limit": 8,
        "scanner_fit_pairs_per_mapping": 8,
        "lut_grid_size": 7,
        "bootstrap_samples": 1000,
        "fast_dev_bootstrap_samples": 10,
        "model_width": 8,
        "feature_methods": {
            "featmap_ridge": 1.0,
            "scangen_hidden": 4,
            "scangen_alpha": 0.16,
            "scangen_radius": 1.0,
            "scangen_epochs": 2,
            "fast_dev_scangen_epochs": 1,
            "scangen_max_train_rows": 64,
        },
        "composite": {
            "weights": {
                "dino_mmd2": 1.0,
                "target_stain_ba": 1.0,
                "target_scanner_ba": 1.0,
                "morphology_preservation": 1.0,
                "od_distance": 1.0,
                "factor_isolation": 1.0,
                "tissue_consistency": 1.0,
            },
            "sensitivity_weights": {
                "domain_fidelity": {
                    "dino_mmd2": 2.0,
                    "target_stain_ba": 2.0,
                    "target_scanner_ba": 2.0,
                    "morphology_preservation": 1.0,
                    "od_distance": 1.0,
                    "factor_isolation": 1.0,
                    "tissue_consistency": 1.0,
                },
                "biology_preservation": {
                    "dino_mmd2": 1.0,
                    "target_stain_ba": 1.0,
                    "target_scanner_ba": 1.0,
                    "morphology_preservation": 2.0,
                    "od_distance": 1.0,
                    "factor_isolation": 1.0,
                    "tissue_consistency": 2.0,
                },
            },
        },
        "compute_budget": {"noadapt": {"gpu_hours": 0.0}},
        "decision": {"win_metrics": 4},
    }
    monkeypatch.setattr(prepare_sota_benchmark, "load_config", lambda _: config)
    monkeypatch.setattr(
        sys, "argv", ["prepare_sota_benchmark.py", "--config", "unused"]
    )
    prepare_sota_benchmark.main()

    monkeypatch.setattr(run_sota_method, "load_config", lambda _: config.copy())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_sota_method.py", "--config", "unused", "--method", "noadapt"],
    )
    run_sota_method.main()

    monkeypatch.setattr(aggregate_sota_benchmark, "load_config", lambda _: config)
    monkeypatch.setattr(
        aggregate_sota_benchmark,
        "_lpips_values",
        lambda *_args: ({}, "LPIPS mocked unavailable"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_sota_benchmark.py",
            "--config",
            "unused",
            "--method",
            "noadapt",
        ],
    )
    aggregate_sota_benchmark.main()
    out = outputs / "m1_sota_benchmark_fast_dev"
    expected = [
        "REPORT.md",
        "FINAL_DECISION.json",
        "LITERATURE_BASELINES.md",
        "baseline_capability_matrix.csv",
        "config_resolved.yaml",
        "metadata/evaluation_manifest.parquet",
        "metadata/reference_policy.json",
        "metadata/method_provenance.json",
        "metadata/leakage_audit.json",
        "tables/PRIMARY_STRICT_COMPOSITION.csv",
        "tables/STAIN_TRACK.csv",
        "tables/SCANNER_TRACK.csv",
        "tables/FEATURE_ROBUSTNESS_TRACK.csv",
        "tables/ORACLE_DIAGNOSTIC.csv",
        "tables/per_combination.csv",
        "tables/statistical_comparisons.csv",
        "tables/compute_budget.csv",
        "figures/SOTA_SUMMARY_DASHBOARD.png",
        "figures/primary_composition_ranking.png",
        "figures/generated_vs_real_grid.png",
        "figures/per_combination_wins.png",
        "figures/real_vs_generated_mmd.png",
        "figures/scanner_track.png",
        "figures/stain_track.png",
        "figures/feature_robustness.png",
        "figures/morphology_vs_domain_tradeoff.png",
        "figures/best_median_worst.png",
    ]
    assert all(
        (out / path).exists() and (out / path).stat().st_size for path in expected
    )
    decision = json.loads((out / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    assert decision["decision_valid"] is False
    audit = json.loads((out / "metadata" / "leakage_audit.json").read_text())
    assert audit["reference_and_scanner_fit_disjoint_from_heldout_targets"] is True
