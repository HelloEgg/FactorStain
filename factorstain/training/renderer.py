from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import functional as TF

from factorstain.data.samplers import FactorialEpisode
from factorstain.losses.factorial import FactorStainLoss
from factorstain.models.factorstain import FactorStain
from factorstain.models.joint_baseline import JointConditionalGenerator, ParallelFactorGenerator
from factorstain.models.reverse_baseline import ReverseFactorStain
from factorstain.utils.checkpoint import resume_if_available, save_training_checkpoint
from factorstain.utils.runtime import barrier, init_distributed, is_main_process, seed_everything


MODEL_BUILDERS = {
    "factorstain": FactorStain,
    "joint": JointConditionalGenerator,
    "parallel": ParallelFactorGenerator,
    "reverse": ReverseFactorStain,
}


def build_renderer(name: str, num_stains: int, num_scanners: int) -> nn.Module:
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown renderer {name}; available: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](num_stains, num_scanners)


def _load_image(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image = TF.resize(image, [size, size], antialias=True)
        return TF.to_tensor(image)


class FactorialImageDataset(Dataset):
    def __init__(
        self,
        index: pd.DataFrame,
        episodes: list[FactorialEpisode],
        image_size: int,
        seed: int = 42,
    ) -> None:
        self.index = index
        self.episodes = episodes
        self.image_size = image_size
        self.seed = seed
        self.stain_to_idx = {value: i for i, value in enumerate(sorted(index.stain_id.unique()))}
        self.scanner_to_idx = {value: i for i, value in enumerate(sorted(index.scanner_id.unique()))}

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, position: int) -> dict:
        episode = self.episodes[position]
        indices = [episode.ax, episode.ay, episode.bx, episode.by]
        # Worker RNG is deterministically reseeded by DataLoader each epoch, so the masked corner changes over time.
        masked = int(torch.randint(0, 4, (1,)).item())
        target_index = indices[masked]
        source_index = indices[3 - masked]  # opposite diagonal contains neither target acquisition factor jointly
        context_indices = [value for value in indices if value not in (target_index, source_index)]
        source_row = self.index.loc[source_index]
        target_row = self.index.loc[target_index]
        return {
            "source": _load_image(source_row.image_path, self.image_size),
            "contexts": torch.stack([_load_image(self.index.loc[item].image_path, self.image_size) for item in context_indices]),
            "target": _load_image(target_row.image_path, self.image_size),
            "stain": self.stain_to_idx[target_row.stain_id],
            "scanner": self.scanner_to_idx[target_row.scanner_id],
            "group": str(target_row.aligned_group_id),
        }


def _episodes_from_cells(index: pd.DataFrame, max_per_group: int = 64, seed: int = 42) -> list[FactorialEpisode]:
    episodes: list[FactorialEpisode] = []
    rng = np.random.default_rng(seed)
    for _, group in index.groupby("aligned_group_id"):
        group_episodes: list[FactorialEpisode] = []
        lookup = {(str(row.stain_id), str(row.scanner_id)): int(i) for i, row in group.iterrows()}
        stains = sorted({key[0] for key in lookup})
        scanners = sorted({key[1] for key in lookup})
        for ai in range(len(stains) - 1):
            for bi in range(ai + 1, len(stains)):
                for xi in range(len(scanners) - 1):
                    for yi in range(xi + 1, len(scanners)):
                        keys = [(stains[ai], scanners[xi]), (stains[ai], scanners[yi]), (stains[bi], scanners[xi]), (stains[bi], scanners[yi])]
                        if all(key in lookup for key in keys):
                            group_episodes.append(FactorialEpisode(*(lookup[key] for key in keys)))
        if len(group_episodes) > max_per_group:
            selected = rng.choice(len(group_episodes), max_per_group, replace=False)
            group_episodes = [group_episodes[int(position)] for position in selected]
        episodes.extend(group_episodes)
    return episodes


def train_renderer(config: dict, model_name: str, index: pd.DataFrame, split: dict, out: Path) -> None:
    device, rank, local_rank, world_size = init_distributed(require_cuda=True)
    seed_everything(config["seed"] + rank)
    train_cells = {(item["stain_id"], item["scanner_id"]) for item in split["train_cells"]}
    train_index = index[
        index.apply(lambda row: (str(row.stain_id), str(row.scanner_id)) in train_cells, axis=1)
        & index.image_exists
        & index.morphology_split.eq("train")
    ]
    episodes = _episodes_from_cells(train_index, config.get("episodes_per_group", 64), config["seed"])
    if not episodes:
        raise RuntimeError("No complete 2×2 factorial episodes remain after the combination/morphology split")
    rng = np.random.default_rng(config["seed"])
    rng.shuffle(episodes)
    validation_count = max(1, round(len(episodes) * 0.1))
    validation_episodes, training_episodes = episodes[:validation_count], episodes[validation_count:]
    if config["fast_dev_run"]:
        maximum = config.get("fast_dev_batches", 3) * config["batch_size"] * world_size
        training_episodes = training_episodes[:maximum]
        validation_episodes = validation_episodes[:max(config["batch_size"], maximum // 2)]

    train_dataset = FactorialImageDataset(index, training_episodes, config["image_size"], config["seed"])
    val_dataset = FactorialImageDataset(index, validation_episodes, config["image_size"], config["seed"] + 999)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config["seed"])
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=config["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], sampler=val_sampler, num_workers=config["num_workers"], pin_memory=True)

    model = build_renderer(model_name, index.stain_id.nunique(), index.scanner_id.nunique()).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs = config.get("fast_dev_epochs", 1) if config["fast_dev_run"] else config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = FactorStainLoss(config["loss_weights"])
    checkpoint_dir = out / "checkpoints" / model_name
    start_epoch, best_metric = resume_if_available(checkpoint_dir / "latest.pt", model, optimizer, scheduler, map_location=device)
    if math.isinf(best_metric):
        best_metric = float("inf")
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    history_path = out / "logs" / f"train_{model_name}.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running, batches = 0.0, 0
        for batch_number, batch in enumerate(loader):
            source = batch["source"].to(device, non_blocking=True)
            contexts = batch["contexts"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            stain = batch["stain"].to(device, non_blocking=True)
            scanner = batch["scanner"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=autocast_dtype):
                generated = model(source, stain, scanner)
                context_flat = contexts.flatten(0, 1)
                repeated_stain = stain[:, None].expand(-1, contexts.shape[1]).reshape(-1)
                repeated_scanner = scanner[:, None].expand(-1, contexts.shape[1]).reshape(-1)
                factorial_generated = model(context_flat, repeated_stain, repeated_scanner).view(source.shape[0], contexts.shape[1], *source.shape[1:]).mean(dim=1)
                losses = loss_fn(generated, target, factorial_generated, target)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(losses["total"].detach())
            batches += 1
            if config["fast_dev_run"] and batch_number + 1 >= config.get("fast_dev_batches", 3):
                break
        model.eval()
        validation_sum = torch.zeros(2, device=device)
        with torch.inference_mode():
            for batch in val_loader:
                source = batch["source"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                stain = batch["stain"].to(device, non_blocking=True)
                scanner = batch["scanner"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=autocast_dtype):
                    generated = model(source, stain, scanner)
                validation_sum[0] += torch.nn.functional.l1_loss(generated, target, reduction="sum")
                validation_sum[1] += target.numel()
        if world_size > 1:
            torch.distributed.all_reduce(validation_sum)
        val_l1 = float((validation_sum[0] / validation_sum[1]).cpu())
        scheduler.step()
        improved = val_l1 < best_metric
        best_metric = min(best_metric, val_l1)
        if is_main_process():
            history.append({"epoch": epoch, "train_loss": running / max(1, batches), "val_l1": val_l1, "lr": optimizer.param_groups[0]["lr"]})
            save_training_checkpoint(checkpoint_dir, model, optimizer, scheduler, epoch, best_metric, improved)
            pd.DataFrame(history).to_csv(history_path, index=False)
    barrier()
