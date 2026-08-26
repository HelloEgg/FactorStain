from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from factorstain.models.factor_adapter import FactorAdapter
from factorstain.utils.checkpoint import resume_if_available, save_training_checkpoint
from factorstain.utils.runtime import barrier, init_distributed, is_main_process, seed_everything


def load_counterfactual_cache(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    with h5py.File(path, "r") as handle:
        features = handle["features"][:].astype(np.float32)
        metadata = pd.DataFrame(
            {
                key: [v.decode() if isinstance(v, bytes) else str(v) for v in handle[key][:]]
                for key in ("aligned_group_id", "tissue_type", "stain_id", "scanner_id", "morphology_split", "source_image_id")
            }
        )
    return features, metadata


class CounterfactualPairDataset(Dataset):
    def __init__(self, features: np.ndarray, metadata: pd.DataFrame, label_maps: dict[str, dict[str, int]], seed: int = 42) -> None:
        self.features = torch.from_numpy(features)
        self.metadata = metadata.reset_index(drop=True)
        self.seed = seed
        self.group_positions = {group: positions.to_numpy() for group, positions in self.metadata.groupby("aligned_group_id").groups.items()}
        self.label_maps = label_maps

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item: int) -> dict:
        row = self.metadata.iloc[item]
        positions = self.group_positions[row.aligned_group_id]
        rng = np.random.default_rng(self.seed + item)
        other = int(rng.choice(positions[positions != item])) if np.any(positions != item) else item
        return {
            "feature": self.features[item],
            "paired_feature": self.features[other],
            "tissue": self.label_maps["tissue"][row.tissue_type],
            "stain": self.label_maps["stain"][row.stain_id],
            "scanner": self.label_maps["scanner"][row.scanner_id],
        }


def adapter_losses(model: FactorAdapter, batch: dict, device: torch.device, weights: dict) -> dict[str, torch.Tensor]:
    original = batch["feature"].to(device)
    paired = batch["paired_feature"].to(device)
    output = model(original)
    target_model = model.module if hasattr(model, "module") else model
    paired_adapted = target_model.encode(paired)
    cf_consistency = 1 - F.cosine_similarity(output["features"], paired_adapted).mean()
    leakage = F.cross_entropy(output["stain_logits"], batch["stain"].to(device)) + F.cross_entropy(output["scanner_logits"], batch["scanner"].to(device))
    tissue = F.cross_entropy(output["tissue_logits"], batch["tissue"].to(device))
    # Preserve both the individual embedding and local pairwise neighborhood geometry.
    identity = 1 - F.cosine_similarity(output["features"], original).mean()
    raw_similarity = F.normalize(original) @ F.normalize(original).T
    adapted_similarity = F.normalize(output["features"]) @ F.normalize(output["features"]).T
    biology = identity + F.mse_loss(adapted_similarity, raw_similarity)
    total = weights["cf_consistency"] * cf_consistency + weights["leakage"] * leakage + weights["biology"] * biology + weights["task"] * tissue
    return {"total": total, "cf_consistency": cf_consistency, "leakage": leakage, "biology": biology, "task": tissue}


def train_adapter(config: dict, model_name: str, cache: Path, out: Path) -> None:
    device, rank, local_rank, world_size = init_distributed(require_cuda=True)
    seed_everything(config["seed"] + rank)
    features, metadata = load_counterfactual_cache(cache)
    train_mask = metadata.morphology_split.eq("train").to_numpy()
    val_mask = metadata.morphology_split.eq("val").to_numpy()
    label_maps = {
        "tissue": {value: i for i, value in enumerate(sorted(metadata.tissue_type.unique()))},
        "stain": {value: i for i, value in enumerate(sorted(metadata.stain_id.unique()))},
        "scanner": {value: i for i, value in enumerate(sorted(metadata.scanner_id.unique()))},
    }
    train_dataset = CounterfactualPairDataset(features[train_mask], metadata[train_mask], label_maps, config["seed"])
    val_dataset = CounterfactualPairDataset(features[val_mask], metadata[val_mask], label_maps, config["seed"] + 1000)
    if not len(train_dataset) or not len(val_dataset):
        raise RuntimeError("Counterfactual cache has no train/validation groups")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config["seed"])
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    per_rank_train = max(1, len(train_dataset) // world_size)
    batch_size = min(config["batch_size"], per_rank_train)
    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=min(config["batch_size"], max(1, len(val_dataset) // world_size)), sampler=val_sampler, num_workers=4, pin_memory=True)
    model = FactorAdapter(features.shape[1], metadata.tissue_type.nunique(), metadata.stain_id.nunique(), metadata.scanner_id.nunique(), config["bottleneck_dim"], config["grl_weight"]).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs = config["fast_dev_epochs"] if config["fast_dev_run"] else config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    checkpoint_dir = out / "checkpoints" / model_name
    try:
        start, best = resume_if_available(checkpoint_dir / "latest.pt", model, optimizer, scheduler, map_location=device)
    except RuntimeError as exc:
        if is_main_process():
            print(f"Existing adapter checkpoint is incompatible with the refreshed feature schema; starting clean: {exc}")
        start, best = 0, float("inf")
    history = []
    for epoch in range(start, epochs):
        train_sampler.set_epoch(epoch); model.train(); running = 0.0; batches = 0
        for batch_no, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            losses = adapter_losses(model, batch, device, config["loss_weights"])
            losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            running += float(losses["total"].detach()); batches += 1
            if config["fast_dev_run"] and batch_no + 1 >= config["fast_dev_batches"]:
                break
        model.eval(); validation = torch.zeros(2, device=device)
        with torch.inference_mode():
            for batch in val_loader:
                losses = adapter_losses(model, batch, device, config["loss_weights"])
                validation += torch.tensor([float(losses["total"]), 1], device=device)
        if world_size > 1:
            torch.distributed.all_reduce(validation)
        val_loss = float((validation[0] / validation[1]).cpu()); scheduler.step(); improved = val_loss < best; best = min(best, val_loss)
        if is_main_process():
            history.append({"epoch": epoch, "train_loss": running / max(1, batches), "val_loss": val_loss})
            save_training_checkpoint(checkpoint_dir, model, optimizer, scheduler, epoch, best, improved)
            pd.DataFrame(history).to_csv(out / "logs" / f"train_{model_name}.csv", index=False)
    barrier()
