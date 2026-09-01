import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from factorstain.data.domain_audit import (
    MIDOG_VERBOSE_SCANNERS,
    build_midog_case_index,
    sample_midog_case_patches,
)
from factorstain.evaluation.domain_audit import (
    domain_centroids,
    grouped_probe,
    paired_cluster_bootstrap,
    plism_controlled_distances,
    sampled_within_between_distances,
)
from factorstain.models.dinov3 import (
    load_feature_cache,
    save_feature_cache,
    select_image_embedding,
)


def test_midog_scanner_assignment_and_metadata_crosscheck(tmp_path: Path):
    records = []
    for case, short in (
        ("001", "Hamamatsu XR"),
        ("051", "Hamamatsu S360"),
        ("101", "Aperio CS2"),
        ("151", "Leica GT450"),
    ):
        filename = f"{case}.tiff"
        Image.new("RGB", (64, 64), (220, 80, 140)).save(tmp_path / filename)
        records.append({"id": int(case), "file_name": filename, "scanner": short})
    (tmp_path / "MIDOG.json").write_text(
        json.dumps({"images": records}), encoding="utf-8"
    )
    cases, verification = build_midog_case_index(tmp_path)
    assert cases.scanner_id.tolist() == list(MIDOG_VERBOSE_SCANNERS.values())
    assert verification["metadata_mismatches"] == []
    assert verification["metadata_scanner_fields_found"] == 4


def test_midog_tissue_patch_sampling_records_coordinates(tmp_path: Path):
    path = tmp_path / "001.tiff"
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    image[16:240, 16:240] = (215, 70, 135)
    Image.fromarray(image).save(path)
    settings = {
        "thumbnail_max_size": 256,
        "assumed_mpp": 0.5,
        "target_mpp": 0.5,
        "patch_size": 64,
        "minimum_tissue_fraction": 0.1,
        "maximum_white_fraction": 0.9,
        "max_patches_per_case": 4,
    }
    patches = sample_midog_case_patches(
        {
            "case_id": "001",
            "image_path": str(path),
            "scanner_id": "Hamamatsu NanoZoomer XR",
        },
        settings,
        seed=42,
    )
    assert len(patches) == 4
    assert {
        "x",
        "y",
        "read_size",
        "patch_size",
        "source_mpp",
        "target_mpp",
        "sample_id",
    }.issubset(patches.columns)
    assert patches.sample_id.nunique() == len(patches)
    assert (patches.tissue_fraction >= 0.1).all()


def test_dinov3_output_selection_has_image_level_dimension():
    pooled = torch.randn(3, 768)
    selected, strategy = select_image_embedding(
        SimpleNamespace(pooler_output=pooled), SimpleNamespace(num_register_tokens=4)
    )
    assert selected.shape == (3, 768)
    assert strategy == "pooler_output"
    hidden = torch.arange(2 * 10 * 4, dtype=torch.float32).reshape(2, 10, 4)
    selected, strategy = select_image_embedding(
        SimpleNamespace(pooler_output=None, last_hidden_state=hidden),
        SimpleNamespace(num_register_tokens=2),
    )
    assert selected.shape == (2, 4)
    assert torch.equal(selected, hidden[:, 3:, :].mean(dim=1))
    assert "excluding_2_register_tokens" in strategy


def test_dinov3_feature_cache_roundtrip(tmp_path: Path):
    destination = tmp_path / "features.npz"
    features = np.arange(15, dtype=np.float32).reshape(3, 5)
    ids = np.asarray(["a", "b", "c"])
    metadata = {
        "model_name": "test",
        "pooling_strategy": "pooler_output",
        "feature_dimension": 5,
    }
    save_feature_cache(destination, features, ids, metadata)
    loaded, loaded_ids, loaded_metadata = load_feature_cache(destination)
    np.testing.assert_array_equal(loaded, features)
    np.testing.assert_array_equal(loaded_ids, ids)
    assert loaded_metadata == metadata


def test_every_audit_probe_split_has_empty_group_intersection():
    rng = np.random.default_rng(42)
    groups = np.repeat([f"g{index:02d}" for index in range(30)], 4)
    labels = np.tile(["a", "b", "c", "d"], 30)
    features = rng.normal(size=(len(groups), 16)).astype(np.float32)
    metrics, _, split = grouped_probe(
        features, labels, groups, "scanner", "plism", 0.2, 42, 200
    )
    train_groups = set(split.loc[split.partition.eq("train"), "group_id"])
    test_groups = set(split.loc[split.partition.eq("test"), "group_id"])
    assert train_groups.intersection(test_groups) == set()
    assert metrics["group_overlap_count"] == 0


def test_audit_statistics_and_projection_execute_on_balanced_fixture():
    rng = np.random.default_rng(7)
    metadata = pd.DataFrame(
        [
            {
                "sample_id": f"g{group}-s{stain}-q{scanner}",
                "aligned_group_id": f"g{group}",
                "stain_id": f"s{stain}",
                "scanner_id": f"q{scanner}",
                "tissue_type": f"t{group % 2}",
            }
            for group in range(8)
            for stain in range(2)
            for scanner in range(2)
        ]
    )
    features = rng.normal(size=(len(metadata), 12)).astype(np.float32)
    matrix, _, dispersion = domain_centroids(features, metadata.scanner_id)
    assert matrix.shape == (2, 2)
    assert len(dispersion) == 2
    controlled = plism_controlled_distances(features, metadata, seed=42)
    reference = "same morphology / same stain / different scanner"
    comparison = "different morphology / same stain / same scanner"
    summary = paired_cluster_bootstrap(
        controlled, reference, comparison, n_bootstrap=50, confidence=0.95, seed=42
    )
    assert summary["bootstrap_unit"] == "aligned_group_id"
    assert summary["n_groups"] > 0
    pairs = sampled_within_between_distances(
        features, metadata.scanner_id, metadata.sample_id, max_pairs=50, seed=42
    )
    assert set(pairs.category) == {"within scanner", "between scanners"}
