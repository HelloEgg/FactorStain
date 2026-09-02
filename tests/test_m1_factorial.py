from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from factorstain.data.samplers import enumerate_rendering_episodes
from factorstain.data.splits import build_combination_split, group_train_val_test_split
from factorstain.evaluation.renderer import (
    RealAcquisitionProbes,
    build_evaluation_pairs,
)
from factorstain.losses.factorial import FactorStainLoss
from factorstain.models.dinov3 import FrozenDINOv3Encoder
from factorstain.training.renderer import train_renderer
from factorstain.utils.checkpoint import resume_if_available, save_training_checkpoint
from scripts import evaluate_renderer, prepare_factorial_split


def _factorial_frame(groups: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "aligned_group_id": f"g{group}",
                "tissue_type": f"t{group % 3}",
                "stain_id": stain,
                "scanner_id": scanner,
                "image_exists": True,
            }
            for group in range(groups)
            for stain in ("A", "B", "C")
            for scanner in ("X", "Y", "Z")
        ]
    )


def test_scanner_episodes_are_exact_same_group_and_stain():
    frame = _factorial_frame(3)
    cells = {
        (str(stain), str(scanner))
        for stain, scanner in zip(frame.stain_id, frame.scanner_id)
    }
    episodes = enumerate_rendering_episodes(frame, cells, max_per_group=1000, seed=42)
    scanner_pairs = [
        episode for episode in episodes if episode.pair_type == "scanner_pair"
    ]
    assert scanner_pairs
    for episode in scanner_pairs:
        source, target = frame.loc[episode.source], frame.loc[episode.target]
        assert source.aligned_group_id == target.aligned_group_id
        assert source.stain_id == target.stain_id
        assert source.scanner_id != target.scanner_id


def test_protocol_b_groups_never_overlap_training_groups():
    frame = _factorial_frame()
    frame["morphology_split"] = group_train_val_test_split(frame, seed=42)
    split = build_combination_split(frame, seed=42)
    pairs = build_evaluation_pairs(frame, split, seed=42)
    train_groups = set(
        frame.loc[frame.morphology_split.eq("train"), "aligned_group_id"]
    )
    protocol_b_targets = frame.loc[
        pairs.loc[pairs.protocol.eq("unseen_combo_unseen_morphology"), "target_index"]
    ]
    assert set(protocol_b_targets.aligned_group_id).isdisjoint(train_groups)
    controlled = pairs[pairs.protocol.eq("controlled_scanner_transfer")]
    for pair in controlled.itertuples():
        source, target = frame.loc[pair.source_index], frame.loc[pair.target_index]
        assert source.aligned_group_id == target.aligned_group_id
        assert source.stain_id == target.stain_id


class _TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)
        self.config = SimpleNamespace(image_size=8, num_register_tokens=0)

    def forward(self, pixel_values):
        return SimpleNamespace(
            pooler_output=self.projection(pixel_values).mean((-2, -1))
        )


def test_frozen_dinov3_has_no_parameter_gradients_but_preserves_image_gradient():
    processor = SimpleNamespace(
        image_mean=(0.5, 0.5, 0.5), image_std=(0.25, 0.25, 0.25), size=8
    )
    encoder = FrozenDINOv3Encoder(processor, _TinyDINO())
    image = torch.rand(2, 3, 8, 8, requires_grad=True)
    encoder(image).sum().backward()
    assert image.grad is not None
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_cross_stain_loss_never_uses_scanner_pixel_supervision():
    loss = FactorStainLoss(
        {
            "scanner_pixel": 1.0,
            "cross_stain_statistics": 1.0,
            "cross_stain_edge": 1.0,
        }
    )
    generated, target, source = (
        torch.rand(2, 3, 16, 16),
        torch.rand(2, 3, 16, 16),
        torch.rand(2, 3, 16, 16),
    )
    values = loss(
        generated,
        target,
        source,
        scanner_pair_mask=torch.tensor([False, False]),
    )
    assert values["scanner_pixel"] == 0
    assert values["cross_stain_statistics"] > 0


def test_checkpoint_resume_restores_epoch_and_parameters(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    save_training_checkpoint(tmp_path, model, optimizer, scheduler, 2, 0.4, True)
    restored = nn.Linear(4, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, 1)
    epoch, best = resume_if_available(
        tmp_path / "latest.pt",
        restored,
        restored_optimizer,
        restored_scheduler,
    )
    assert epoch == 3
    assert best == 0.4
    for expected, observed in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, observed)


def test_acquisition_classifier_contract_is_real_only():
    assert RealAcquisitionProbes.trained_on_generated is False


class _FakeDINO(nn.Module):
    def forward(self, images):
        mean = images.mean((-2, -1))
        std = images.std((-2, -1))
        return torch.cat([mean, std], dim=1)


class _FakeProbes:
    trained_on_generated = False
    training_source = "real_training_images_only"

    def __init__(self, _cache, metadata, _train_cells):
        self.stains = metadata.stain_id.nunique()
        self.scanners = metadata.scanner_id.nunique()
        self.training_samples = len(metadata)

    def probabilities(self, features):
        stain = np.full((len(features), self.stains), 1 / self.stains)
        scanner = np.full((len(features), self.scanners), 1 / self.scanners)
        return stain, scanner


class _FakeLPIPS(nn.Module):
    def forward(self, generated, target):
        return (generated - target).abs().mean((1, 2, 3), keepdim=True)


def test_fast_dev_run_creates_complete_m1_output_contract(tmp_path, monkeypatch):
    # The minimal local test runtime omits pyarrow; exercise the identical
    # DataFrame contract through pickle-backed test files.
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=False: self.to_pickle(path),
    )
    monkeypatch.setattr(pd, "read_parquet", pd.read_pickle)
    outputs = tmp_path / "outputs"
    image_root = tmp_path / "images"
    image_root.mkdir()
    frame = _factorial_frame()
    paths = []
    for position in range(len(frame)):
        path = image_root / f"{position}.png"
        Image.new("RGB", (20, 20), (position % 255, 80, 140)).save(path)
        paths.append(str(path))
    frame["image_path"] = paths
    frame["sample_id"] = [f"sample-{position}" for position in range(len(frame))]
    frame["image_id"] = frame.sample_id
    index_path = tmp_path / "plism.parquet"
    frame.to_parquet(index_path, index=False)
    config = {
        "milestone": "m1_factorial",
        "seed": 42,
        "fast_dev_run": True,
        "paths": {
            "outputs_root": str(outputs),
            "project_root": str(tmp_path),
            "plism_root": str(image_root),
        },
        "image_size": 16,
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 1,
        "fast_dev_epochs": 1,
        "fast_dev_batches": 1,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "gradient_accumulation_steps": 1,
        "early_stopping_patience": 2,
        "model_width": 8,
        "holdout_fraction": 0.18,
        "validation_fraction": 0.10,
        "morphology_split": [0.70, 0.15, 0.15],
        "models": ["joint", "parallel", "factorstain"],
        "episodes_per_group": 12,
        "validation_episodes_per_group": 6,
        "fast_dev_eval_per_protocol": 3,
        "fast_dev_bootstrap_samples": 50,
        "bootstrap_samples": 1000,
        "loss_weights": {
            "scanner_pixel": 1.0,
            "cross_stain_statistics": 1.0,
            "cross_stain_edge": 0.25,
        },
        "auxiliary": {
            "dinov3_model_name": "fake",
            "dinov3_dtype": "float32",
            "use_dinov3_training_loss": False,
            "use_lpips_training_loss": False,
        },
        "decision": {
            "win_metrics": 4,
            "relative_improvement": 0.05,
            "relative_improvement_metrics": 2,
            "consistent_metrics": 1,
        },
    }
    monkeypatch.setenv("PLISM_INDEX", str(index_path))
    monkeypatch.setenv("ALLOW_CPU_FAST_DEV", "1")
    monkeypatch.setattr(prepare_factorial_split, "load_config", lambda _: config)
    monkeypatch.setattr(
        sys, "argv", ["prepare_factorial_split.py", "--config", "unused"]
    )
    prepare_factorial_split.main()
    out = outputs / "m1_factorial"
    indexed = pd.read_parquet(out / "metadata" / "plism_index_with_splits.parquet")
    split = json.loads((out / "splits" / "combination_split_seed42.json").read_text())
    for model_name in config["models"]:
        train_renderer(config, model_name, indexed, split, out)

    fake_lpips_module = types.ModuleType("lpips")
    fake_lpips_module.LPIPS = lambda net="alex": _FakeLPIPS()
    monkeypatch.setitem(sys.modules, "lpips", fake_lpips_module)
    monkeypatch.setattr(evaluate_renderer, "load_config", lambda _: config)
    monkeypatch.setattr(
        evaluate_renderer,
        "load_frozen_dinov3_encoder",
        lambda *_args, **_kwargs: _FakeDINO(),
    )
    monkeypatch.setattr(evaluate_renderer, "RealAcquisitionProbes", _FakeProbes)
    monkeypatch.setattr(
        evaluate_renderer, "_locate_dino_cache", lambda _config: tmp_path / "unused.npz"
    )
    monkeypatch.setattr(sys, "argv", ["evaluate_renderer.py", "--config", "unused"])
    evaluate_renderer.main()
    expected = [
        "REPORT.md",
        "GO_NOGO.json",
        "metrics.json",
        "config_resolved.yaml",
        "splits/combination_split_seed42.json",
        "splits/morphology_split_seed42.json",
        "metadata/factorial_episode_statistics.json",
        "tables/overall_metrics.csv",
        "tables/unseen_combo_metrics.csv",
        "tables/controlled_scanner_metrics.csv",
        "tables/per_combination_metrics.csv",
        "tables/bootstrap_CI.csv",
        "figures/SUMMARY_DASHBOARD.png",
        "figures/heldout_combination_matrix.png",
        "figures/factorial_2x2_examples.png",
        "figures/generated_vs_real_unseen.png",
        "figures/controlled_scanner_examples.png",
        "figures/seen_vs_unseen_performance.png",
        "figures/model_comparison_metrics.png",
        "figures/unseen_combo_mmd.png",
        "figures/factor_isolation.png",
        "figures/morphology_preservation.png",
        "figures/best_median_worst_examples.png",
        "figures/failure_cases.png",
    ]
    assert all(
        (out / path).exists() and (out / path).stat().st_size > 0 for path in expected
    )
    decision = json.loads((out / "GO_NOGO.json").read_text())
    assert decision["decision_valid"] is False
