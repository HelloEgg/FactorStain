from __future__ import annotations

import hashlib
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from factorstain.baselines import official_neural
from factorstain.baselines.base import FitContext
from factorstain.baselines.external import (
    OfficialAdapterExecutionError,
    OfficialSubprocessMethod,
)


def test_external_settings_have_bounded_fast_dev_overrides():
    _, settings = official_neural._read_settings("sastaindiff", fast_dev=True)
    assert settings["train_steps"] == 1
    assert settings["batch_size"] == 2
    assert settings["model_channels"] == 32
    assert settings["max_train_images_per_domain"] == 8


def test_fit_request_rejects_forbidden_training_sample(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (80, 40, 120)).save(image)
    frame = pd.DataFrame(
        [
            {
                "sample_id": "forbidden",
                "image_id": "forbidden",
                "aligned_group_id": "g1",
                "stain_id": "A",
                "scanner_id": "X",
                "image_path": str(image),
            }
        ]
    )
    train = tmp_path / "train.parquet"
    validation = tmp_path / "validation.parquet"
    scanner_pairs = tmp_path / "scanner_pairs.parquet"
    frame.to_parquet(train, index=False)
    frame.iloc[0:0].to_parquet(validation, index=False)
    pd.DataFrame().to_parquet(scanner_pairs, index=False)
    policy = tmp_path / "policy.json"
    forbidden_hash = hashlib.sha256(b"forbidden").hexdigest()
    policy.write_text(
        json.dumps(
            {
                "forbidden_target_sample_ids": ["forbidden"],
                "forbidden_target_ids_sha256": forbidden_hash,
            }
        ),
        encoding="utf-8",
    )
    request = {
        "train_manifest": str(train),
        "validation_manifest": str(validation),
        "reference_policy": str(policy),
        "scanner_fit_pairs": str(scanner_pairs),
        "checkpoint_dir": str(tmp_path / "checkpoint"),
        "forbidden_target_ids_sha256": forbidden_hash,
    }
    with pytest.raises(official_neural.AdapterError, match="leakage"):
        official_neural._load_fit_request("stainnet", request)


def test_default_external_command_uses_bundled_adapter(monkeypatch):
    monkeypatch.delenv("FACTORSTAIN_STAINNET_COMMAND", raising=False)
    method = OfficialSubprocessMethod("stainnet")
    command = shlex.split(method.command)
    assert command[0] == sys.executable
    assert command[-2:] == ["--method", "stainnet"]
    assert Path(command[-3]).name == "run_official_baseline_adapter.py"


def test_installed_adapter_crash_is_execution_failure(tmp_path):
    adapter = tmp_path / "fail.py"
    adapter.write_text("raise RuntimeError('training failed')\n", encoding="utf-8")
    method = OfficialSubprocessMethod(
        "histaugan", command=shlex.join([sys.executable, str(adapter)])
    )
    request = tmp_path / "request.json"
    request.write_text("{}\n", encoding="utf-8")
    with pytest.raises(OfficialAdapterExecutionError, match="subprocess failed"):
        method._run("fit", request)


def test_persistent_adapter_transport(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """
import argparse
import json
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--method')
parser.add_argument('mode')
parser.add_argument('--request')
args = parser.parse_args()
if args.mode == 'serve':
    print(json.dumps({'status': 'READY'}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        shutil.copyfile(request['source_path'], request['output_path'])
        print(json.dumps({'status': 'COMPLETE'}), flush=True)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    command = shlex.join([sys.executable, str(adapter), "--method", "pix2pix"])
    monkeypatch.setenv("FACTORSTAIN_ADAPTER_PERSISTENT", "1")
    benchmark = tmp_path / "benchmark"
    (benchmark / "metadata").mkdir(parents=True)
    context = FitContext(
        train_index=pd.DataFrame(),
        reference_policy={"forbidden_target_ids_sha256": "hash"},
        scanner_pairs=pd.DataFrame(),
        output_dir=tmp_path / "checkpoints",
        image_size=8,
        seed=42,
        metadata={"benchmark_output": str(benchmark)},
    )
    method = OfficialSubprocessMethod(
        "pix2pix", command=command, append_scanner_lut=False
    )
    method.fit(context)
    source = np.full((8, 8, 3), (12, 34, 56), dtype=np.uint8)
    generated = method.translate(source, "A", "X", "A", "Y", track="B")
    method.close()
    np.testing.assert_array_equal(generated, source)


def test_stain_matrix_and_normalization_preserve_shape():
    y, x = np.mgrid[:32, :32]
    image = (
        np.stack([80 + x * 2, 45 + y * 2, 100 + (x + y)], axis=-1)
        .clip(1, 220)
        .astype(np.uint8)
    )
    matrix = official_neural._stain_matrix(image)
    normalized = official_neural._normalize_to_stain(image, matrix)
    assert matrix.shape == (3, 2)
    assert normalized.shape == image.shape
    assert normalized.dtype == np.uint8


def test_external_checkpoint_path_is_fit_fingerprint_scoped(tmp_path):
    first = {
        "checkpoint_dir": tmp_path,
        "fit_fingerprint": "a" * 64,
    }
    second = {
        "checkpoint_dir": tmp_path,
        "fit_fingerprint": "b" * 64,
    }
    first_path = official_neural._checkpoint_path(first, "target")
    second_path = official_neural._checkpoint_path(second, "target")
    assert first_path != second_path
    assert "aaaaaaaaaaaa" in first_path.name


def test_checkpoint_migration_requires_matching_training_protocol():
    hashes = {
        "train_manifest": "train",
        "validation_manifest": "validation",
        "reference_policy": "policy",
        "scanner_fit_pairs": "pairs",
    }
    previous = {
        "method": "stainnet",
        "official_source_commit": "commit",
        "seed": 42,
        "image_size": 256,
        "fast_dev_run": True,
        "settings": {"train_steps": 1},
        "training_manifest_sha256": "train",
        "validation_manifest_sha256": "validation",
        "reference_policy_sha256": "policy",
        "scanner_fit_pairs_sha256": "pairs",
    }
    arguments = {
        "method": "stainnet",
        "source_commit": "commit",
        "seed": 42,
        "image_size": 256,
        "fast_dev_run": True,
        "settings": {"train_steps": 1},
        "input_hashes": hashes,
    }
    assert official_neural._previous_fit_is_compatible(
        previous, protocol_revision=1, **arguments
    )
    assert not official_neural._previous_fit_is_compatible(
        previous, protocol_revision=2, **arguments
    )


def test_external_limit_sorts_string_hashes_deterministically():
    frame = pd.DataFrame(
        {
            "sample_id": [f"sample-{number}" for number in range(10)],
            "value": range(10),
        }
    )
    expected = sorted(
        frame.sample_id,
        key=lambda value: hashlib.sha256(f"42:{value}".encode()).hexdigest(),
    )[:3]
    first = official_neural._limit(frame, limit=3, seed=42)
    second = official_neural._limit(frame, limit=3, seed=42)
    assert first.sample_id.tolist() == expected
    pd.testing.assert_frame_equal(first, second)
    assert "_selection_key" not in first


def test_histaugan_training_batch_uses_official_crop_size(tmp_path):
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    path = tmp_path / "tile.png"
    Image.fromarray(image).save(path)
    batch = official_neural._load_random_crop_batch(
        [str(path)],
        resize_size=256,
        crop_size=216,
        device=torch.device("cpu"),
        rng=np.random.default_rng(42),
    )
    assert batch.shape == (1, 3, 216, 216)
    assert float(batch.min()) >= -1 and float(batch.max()) <= 1


def test_sastaindiff_train_only_stain_augmentation(tmp_path):
    y, x = np.mgrid[:32, :32]
    image = (
        np.stack([80 + x * 2, 45 + y * 2, 100 + (x + y)], axis=-1)
        .clip(1, 220)
        .astype(np.uint8)
    )
    path = tmp_path / "target.png"
    Image.fromarray(image).save(path)
    matrix = official_neural._stain_matrix(image)
    database = np.stack([matrix, matrix])
    tree = official_neural.cKDTree(database.reshape(2, 6))
    settings = {
        "nearest_neighbours": 2,
        "sigma_perturb": 0.1,
        "sigma1": 0.0,
        "sigma2": 0.0,
        "shift_value": 0,
        "color_augmentation_probability": 0.0,
        "color_attempts": 1,
        "stain_attempts": 1,
        "gaussian_blur": False,
    }
    clean, augmented = official_neural._augment_stain(
        str(path), 32, database, tree, settings, np.random.default_rng(42)
    )
    assert clean.shape == augmented.shape == (3, 32, 32)
    assert torch.isfinite(clean).all() and torch.isfinite(augmented).all()
