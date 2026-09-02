import numpy as np
import pandas as pd

from factorstain.evaluation.mmd import (
    balanced_stain_indices,
    bootstrap_mmd2,
    compute_mmd2,
    controlled_scanner_indices,
    matrix_from_pair_records,
    median_bandwidth,
    mmd2_with_bandwidth,
)
from factorstain.models.dinov3 import save_feature_cache


def test_mmd_identical_is_reported_as_zero_and_is_symmetric():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 12)).astype(np.float32)
    sigma = median_bandwidth(features, features, seed=42)
    identical = mmd2_with_bandwidth(features, features, sigma)
    forward = mmd2_with_bandwidth(features[:40], features[40:], sigma)
    reverse = mmd2_with_bandwidth(features[40:], features[:40], sigma)
    assert identical["reported_mmd2"] < 1e-10
    assert np.isclose(forward["raw_mmd2"], reverse["raw_mmd2"], atol=1e-12)
    assert np.isclose(
        forward["single_raw_mmd2"], reverse["single_raw_mmd2"], atol=1e-12
    )


def test_shifted_normal_has_larger_mmd_than_same_distribution():
    rng = np.random.default_rng(7)
    left = rng.normal(size=(160, 10)).astype(np.float32)
    same = rng.normal(size=(160, 10)).astype(np.float32)
    shifted = rng.normal(loc=1.25, size=(160, 10)).astype(np.float32)
    same_result = compute_mmd2(left, same, seed=11)
    shifted_result = compute_mmd2(left, shifted, seed=11)
    assert same_result["reported_mmd2"] < shifted_result["reported_mmd2"]
    assert shifted_result["reported_mmd2"] > 0.05


def test_mmd_is_deterministic_with_fixed_seed():
    rng = np.random.default_rng(13)
    left = rng.normal(size=(70, 8)).astype(np.float32)
    right = rng.normal(loc=0.4, size=(70, 8)).astype(np.float32)
    first = compute_mmd2(left, right, max_distance_pairs=300, seed=99)
    second = compute_mmd2(left, right, max_distance_pairs=300, seed=99)
    assert first == second
    boot_first = bootstrap_mmd2(
        left, right, first["sigma"], (0.5, 1.0, 2.0), 12, seed=123
    )
    boot_second = bootstrap_mmd2(
        left, right, first["sigma"], (0.5, 1.0, 2.0), 12, seed=123
    )
    assert boot_first == boot_second


def test_pairwise_matrix_is_symmetric_with_zero_diagonal():
    records = pd.DataFrame(
        [
            {"domain_a": "A", "domain_b": "B", "reported_mmd2": 0.1},
            {"domain_a": "A", "domain_b": "C", "reported_mmd2": 0.2},
            {"domain_a": "B", "domain_b": "C", "reported_mmd2": 0.3},
        ]
    )
    matrix = matrix_from_pair_records(records, ["A", "B", "C"], "reported_mmd2")
    np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)
    np.testing.assert_array_equal(np.diag(matrix), np.zeros(3))


def test_controlled_scanner_and_balanced_stain_selection():
    metadata = pd.DataFrame(
        [
            {
                "aligned_group_id": f"g{group}",
                "tissue_type": f"t{group % 2}",
                "stain_id": stain,
                "scanner_id": scanner,
            }
            for group in range(6)
            for stain in ("s1", "s2")
            for scanner in ("q1", "q2")
        ]
    )
    left, right, clusters, info = controlled_scanner_indices(
        metadata, "q1", "q2", max_samples=100, seed=42
    )
    assert len(left) == len(right) == 12
    assert info["n_aligned_groups"] == 6
    assert set(clusters) == {f"g{group}" for group in range(6)}
    left, right, strata, info = balanced_stain_indices(
        metadata, "s1", "s2", max_samples=100, seed=42
    )
    assert len(left) == len(right) == 12
    assert info["n_strata"] == 4
    assert len(strata) == 12


def test_mmd_runner_reuses_cache_and_writes_requested_outputs(tmp_path, monkeypatch):
    from scripts import run_mmd_analysis

    output_root = tmp_path / "outputs"
    source = output_root / "m_minus1_domain_audit"
    (source / "features").mkdir(parents=True)
    (source / "metadata").mkdir(parents=True)
    rng = np.random.default_rng(123)
    scanner_names, stain_names = ("q1", "q2", "q3"), ("s1", "s2", "s3")
    rows, feature_rows = [], []
    for group in range(8):
        for stain_number, stain in enumerate(stain_names):
            for scanner_number, scanner in enumerate(scanner_names):
                sample_id = f"g{group}-{stain}-{scanner}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "aligned_group_id": f"g{group}",
                        "tissue_type": f"t{group % 2}",
                        "stain_id": stain,
                        "scanner_id": scanner,
                    }
                )
                value = rng.normal(size=12)
                value[scanner_number] += 0.7
                value[3 + stain_number] += 0.5
                feature_rows.append(value)
    metadata = pd.DataFrame(rows)
    metadata_path = source / "metadata" / "plism_samples.parquet"
    metadata_path.write_text("synthetic parquet placeholder", encoding="utf-8")
    monkeypatch.setattr(run_mmd_analysis.pd, "read_parquet", lambda _: metadata.copy())
    save_feature_cache(
        source / "features" / "plism_dinov3.npz",
        np.asarray(feature_rows, dtype=np.float32),
        metadata.sample_id.to_numpy(),
        {
            "model_name": "synthetic-dinov3",
            "feature_dimension": 12,
            "pooling_strategy": "pooler_output",
        },
    )
    (source / "metrics.json").write_text(
        '{"probes":[{"dataset":"plism","target":"scanner","balanced_accuracy":0.8}]}',
        encoding="utf-8",
    )
    monkeypatch.setenv("OUTPUTS_ROOT", str(output_root))
    monkeypatch.setenv("FAST_DEV_RUN", "1")
    monkeypatch.setattr(
        "sys.argv", ["run_mmd_analysis.py", "--config", "configs/m_minus1_mmd.yaml"]
    )
    run_mmd_analysis.main()
    result = output_root / "m_minus1_mmd"
    expected = [
        result / "REPORT.md",
        result / "metrics.json",
        result / "tables" / "scanner_mmd.csv",
        result / "tables" / "scanner_mmd_controlled.csv",
        result / "tables" / "stain_mmd.csv",
        result / "tables" / "stain_mmd_balanced.csv",
        result / "tables" / "scanner_mmd_ci_lower.csv",
        result / "tables" / "stain_mmd_ci_upper.csv",
        result / "figures" / "scanner_mmd_controlled_heatmap.png",
        result / "figures" / "stain_mmd_balanced_heatmap.png",
        result / "figures" / "scanner_centroid_vs_mmd.png",
        result / "figures" / "MMD_SUMMARY.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
    scanner_matrix = pd.read_csv(result / "tables" / "scanner_mmd.csv", index_col=0)
    np.testing.assert_array_equal(np.diag(scanner_matrix), np.zeros(3))
