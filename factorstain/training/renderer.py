from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from factorstain.data.samplers import RenderingEpisode, enumerate_rendering_episodes
from factorstain.losses.factorial import FactorStainLoss
from factorstain.models.dinov3 import load_frozen_dinov3_encoder
from factorstain.models.factorstain import FactorStain
from factorstain.models.joint_baseline import (
    JointConditionalGenerator,
    ParallelFactorGenerator,
)
from factorstain.models.reverse_baseline import ReverseFactorStain
from factorstain.utils.checkpoint import resume_if_available, save_training_checkpoint
from factorstain.utils.runtime import (
    barrier,
    init_distributed,
    is_main_process,
    seed_everything,
)

MODEL_BUILDERS = {
    "factorstain": FactorStain,
    "joint": JointConditionalGenerator,
    "parallel": ParallelFactorGenerator,
    "reverse": ReverseFactorStain,
}


def build_renderer(
    name: str,
    num_stains: int,
    num_scanners: int,
    width: int = 64,
) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown renderer {name}; available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](num_stains, num_scanners, width=width)


def _load_image(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize(
            (size, size), resample=Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class FactorialImageDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        episodes: list[RenderingEpisode],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.index = index
        self.episodes = episodes
        self.image_size = image_size
        self.augment = augment
        self.stain_to_idx = {
            value: position
            for position, value in enumerate(sorted(index.stain_id.unique()))
        }
        self.scanner_to_idx = {
            value: position
            for position, value in enumerate(sorted(index.scanner_id.unique()))
        }

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, position: int) -> dict:
        episode = self.episodes[position]
        source_row = self.index.loc[episode.source]
        target_row = self.index.loc[episode.target]
        source = _load_image(source_row.image_path, self.image_size)
        target = _load_image(target_row.image_path, self.image_size)
        if self.augment:
            if torch.rand(()) < 0.5:
                source, target = source.flip(-1), target.flip(-1)
            if torch.rand(()) < 0.5:
                source, target = source.flip(-2), target.flip(-2)
            rotations = int(torch.randint(0, 4, ()).item())
            source, target = (
                torch.rot90(source, rotations, (-2, -1)),
                torch.rot90(target, rotations, (-2, -1)),
            )
        return {
            "source": source,
            "target": target,
            "source_stain": self.stain_to_idx[source_row.stain_id],
            "source_scanner": self.scanner_to_idx[source_row.scanner_id],
            "target_stain": self.stain_to_idx[target_row.stain_id],
            "target_scanner": self.scanner_to_idx[target_row.scanner_id],
            "scanner_pair": episode.pair_type == "scanner_pair",
            "pair_type": episode.pair_type,
            "group": str(target_row.aligned_group_id),
        }


def _cell_set(items: list[dict]) -> set[tuple[str, str]]:
    return {(str(item["stain_id"]), str(item["scanner_id"])) for item in items}


def _autocast(device: torch.device, dtype: torch.dtype):
    return (
        torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
    )


def _load_auxiliaries(config: dict, device: torch.device):
    settings = config.get("auxiliary", {})
    if not settings:
        return None, None
    dino = None
    if settings.get("use_dinov3_training_loss", True):
        dino = load_frozen_dinov3_encoder(
            settings["dinov3_model_name"],
            device,
            settings.get("dinov3_dtype", "bfloat16"),
        )
    lpips_model = None
    if settings.get("use_lpips_training_loss", True):
        try:
            import lpips

            lpips_model = (
                lpips.LPIPS(net="alex").to(device).eval().requires_grad_(False)
            )
        except Exception as exc:
            raise RuntimeError(
                f"LPIPS training loss could not be initialized: {exc}"
            ) from exc
    return dino, lpips_model


def _loss_weights(config: dict) -> dict[str, float]:
    weights = config["loss_weights"]
    if "scanner_pixel" in weights:
        return weights
    # Preserve older M2 configs while sharing the scientifically corrected
    # pair-aware trainer introduced for M1.
    return {
        "scanner_pixel": weights.get("reconstruction", 1.0),
        "scanner_ssim": 0.0,
        "scanner_lpips": weights.get("perceptual", 0.0),
        "scanner_dino": weights.get("morphology", 0.0),
        "scanner_edge": weights.get("morphology", 0.0),
        "cross_stain_statistics": weights.get("factor_correctness", 1.0),
        "cross_stain_dino_morphology": weights.get("morphology", 0.5),
        "cross_stain_edge": weights.get("isolation", 0.2),
        "cross_stain_grayscale": weights.get("factorial", 0.2),
    }


def _semantic_features(dino, source, target, generated):
    if dino is None:
        return None, None, None
    with torch.no_grad():
        source_features = dino(source)
        target_features = dino(target)
    generated_features = dino(generated)
    return source_features, target_features, generated_features


def train_renderer(
    config: dict,
    model_name: str,
    index: pd.DataFrame,
    split: dict,
    out: Path,
) -> None:
    allow_cpu = config["fast_dev_run"] and os.getenv("ALLOW_CPU_FAST_DEV", "0") == "1"
    device, rank, local_rank, world_size = init_distributed(require_cuda=not allow_cpu)
    seed_everything(config["seed"] + rank)
    train_cells = _cell_set(split["train_cells"])
    validation_cells = _cell_set(split.get("validation_cells", [])) or train_cells
    train_episodes = enumerate_rendering_episodes(
        index,
        train_cells,
        morphology_split="train",
        max_per_group=config.get("episodes_per_group", 64),
        seed=config["seed"],
    )
    validation_episodes = enumerate_rendering_episodes(
        index,
        train_cells,
        validation_cells,
        morphology_split="val",
        max_per_group=config.get("validation_episodes_per_group", 16),
        seed=config["seed"] + 1,
    )
    if not validation_episodes:
        validation_episodes = enumerate_rendering_episodes(
            index,
            train_cells,
            morphology_split="val",
            max_per_group=config.get("validation_episodes_per_group", 16),
            seed=config["seed"] + 1,
        )
    if not train_episodes or not validation_episodes:
        raise RuntimeError(
            "No M1 source-target episodes remain after acquisition and morphology splits"
        )
    if config["fast_dev_run"]:
        maximum = config.get("fast_dev_batches", 2) * config["batch_size"] * world_size
        train_episodes = train_episodes[:maximum]
        validation_episodes = validation_episodes[
            : max(config["batch_size"], maximum // 2)
        ]

    train_dataset = FactorialImageDataset(
        index, train_episodes, config["image_size"], augment=True
    )
    val_dataset = FactorialImageDataset(
        index, validation_episodes, config["image_size"], augment=False
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config["seed"],
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= config["batch_size"] * world_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = build_renderer(
        model_name,
        index.stain_id.nunique(),
        index.scanner_id.nunique(),
        config.get("model_width", 64),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    epochs = (
        config.get("fast_dev_epochs", 1) if config["fast_dev_run"] else config["epochs"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    loss_fn = FactorStainLoss(_loss_weights(config))
    checkpoint_dir = out / "checkpoints" / model_name
    start_epoch, best_metric = resume_if_available(
        checkpoint_dir / "latest.pt", model, optimizer, scheduler, map_location=device
    )
    if math.isinf(best_metric):
        best_metric = float("inf")
    dino, lpips_model = _load_auxiliaries(config, device)
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    history_path = out / "logs" / f"train_{model_name}.csv"
    history = (
        pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    )
    no_improvement = 0
    accumulation = max(1, config.get("gradient_accumulation_steps", 1))
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, batches = 0.0, 0
        for batch_number, batch in enumerate(loader):
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            stain = batch["target_stain"].to(device, non_blocking=True)
            scanner = batch["target_scanner"].to(device, non_blocking=True)
            scanner_pair = batch["scanner_pair"].to(device, non_blocking=True)
            with _autocast(device, autocast_dtype):
                generated = model(source, stain, scanner)
                source_features, target_features, generated_features = (
                    _semantic_features(dino, source, target, generated)
                )
                lpips_values = (
                    lpips_model(generated * 2 - 1, target * 2 - 1).flatten()
                    if lpips_model is not None
                    else None
                )
                losses = loss_fn(
                    generated,
                    target,
                    source,
                    scanner_pair,
                    generated_features,
                    target_features,
                    source_features,
                    lpips_values,
                )
                scaled_loss = losses["total"] / accumulation
            scaled_loss.backward()
            should_step = (
                batch_number + 1
            ) % accumulation == 0 or batch_number + 1 == len(loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += float(losses["total"].detach())
            batches += 1
            if config["fast_dev_run"] and batch_number + 1 >= config.get(
                "fast_dev_batches", 2
            ):
                break

        model.eval()
        validation_sum = torch.zeros(2, device=device)
        with torch.inference_mode():
            for batch in val_loader:
                source = batch["source"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                stain = batch["target_stain"].to(device, non_blocking=True)
                scanner = batch["target_scanner"].to(device, non_blocking=True)
                scanner_pair = batch["scanner_pair"].to(device, non_blocking=True)
                with _autocast(device, autocast_dtype):
                    generated = model(source, stain, scanner)
                    validation_loss = loss_fn(generated, target, source, scanner_pair)[
                        "total"
                    ]
                validation_sum[0] += validation_loss * len(source)
                validation_sum[1] += len(source)
        if world_size > 1:
            torch.distributed.all_reduce(validation_sum)
        val_loss = float((validation_sum[0] / validation_sum[1].clamp_min(1)).cpu())
        scheduler.step()
        improved = val_loss < best_metric
        if improved:
            best_metric = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if is_main_process():
            peak_memory = (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": running / max(1, batches),
                    "validation_pair_aware_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "parameter_count": parameter_count,
                    "epoch_seconds": time.monotonic() - epoch_started,
                    "peak_gpu_memory_gb": peak_memory,
                    "training_episodes": len(train_episodes),
                    "validation_episodes": len(validation_episodes),
                }
            )
            save_training_checkpoint(
                checkpoint_dir,
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                improved,
            )
            pd.DataFrame(history).to_csv(history_path, index=False)
        if no_improvement >= config.get("early_stopping_patience", 10):
            break
    barrier()
