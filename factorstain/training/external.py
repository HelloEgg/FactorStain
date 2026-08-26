from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from factorstain.models.factor_adapter import FactorAdapter
from factorstain.models.mil import ABMIL
from factorstain.training.adapter import load_counterfactual_cache
from factorstain.utils.checkpoint import resume_if_available, save_training_checkpoint
from factorstain.utils.runtime import barrier, init_distributed, is_main_process, seed_everything


class H5BagDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, bag_dir: Path, max_tiles: int, adapter: FactorAdapter | None = None, seed: int = 42) -> None:
        self.frame = frame.reset_index(drop=True)
        self.bag_dir = bag_dir
        self.max_tiles = max_tiles
        self.adapter = adapter.cpu().eval() if adapter is not None else None
        self.seed = seed

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, item: int) -> dict:
        row = self.frame.iloc[item]
        with h5py.File(self.bag_dir / f"{row.slide_id}.h5", "r") as handle:
            features = handle["features"][:].astype(np.float32)
        if len(features) > self.max_tiles:
            positions = np.random.default_rng(self.seed + item).choice(len(features), self.max_tiles, replace=False)
            features = features[positions]
        tensor = torch.from_numpy(features)
        if self.adapter is not None:
            with torch.inference_mode():
                tensor = self.adapter.encode(tensor)
        return {"features": tensor, "label": int(row.label), "slide_id": str(row.slide_id), "patient_id": int(row.patient_id), "center": int(row.center)}


def _adapter_for(model_name: str, config: dict, feature_dim: int, device: str = "cpu") -> FactorAdapter:
    outputs = Path(config["paths"]["outputs_root"])
    cf_cache = outputs / "m4_adapter" / "counterfactual_features" / model_name / "features.h5"
    checkpoint = outputs / "m4_adapter" / "checkpoints" / model_name / "best.pt"
    if not (cf_cache.exists() and checkpoint.exists()):
        raise FileNotFoundError(f"FactorAdapter artifacts missing for {model_name}")
    _, metadata = load_counterfactual_cache(cf_cache)
    import yaml
    resolved_path = outputs / "m4_adapter" / "config_resolved.yaml"
    adapter_config = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) if resolved_path.exists() else {}
    bottleneck = int(adapter_config.get("bottleneck_dim", config.get("adapter_bottleneck_dim", 256)))
    grl_weight = float(adapter_config.get("grl_weight", 1.0))
    adapter = FactorAdapter(feature_dim, metadata.tissue_type.nunique(), metadata.stain_id.nunique(), metadata.scanner_id.nunique(), bottleneck, grl_weight)
    adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    return adapter.to(device).eval()


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
        "auprc": float(average_precision_score(labels, probabilities)) if labels.sum() else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def train_camelyon(config: dict, model_name: str, representation: str, heldout_center: int, out: Path) -> None:
    device, rank, local_rank, world_size = init_distributed(require_cuda=True)
    seed_everything(config["seed"] + rank + heldout_center * 100)
    full_index = pd.read_parquet(out / "camelyon_index.parquet")
    bag_dir = out / "external_features" / "camelyon" / model_name
    full_index = full_index[full_index.slide_id.map(lambda value: (bag_dir / f"{value}.h5").exists())]
    test = full_index[full_index.center.eq(heldout_center)]
    development = full_index[full_index.center.ne(heldout_center)]
    if test.empty or development.empty:
        raise RuntimeError(f"Center {heldout_center}: missing cached bags for train or test")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config["seed"] + heldout_center)
    train_positions, val_positions = next(splitter.split(development, groups=development.patient_id))
    train, val = development.iloc[train_positions], development.iloc[val_positions]
    first_bag = next(bag_dir.glob("*.h5"))
    with h5py.File(first_bag, "r") as handle:
        feature_dim = handle["features"].shape[1]
    adapter = _adapter_for(model_name, config, feature_dim) if representation == "adapter" else None
    train_dataset = H5BagDataset(train, bag_dir, config["tiles_per_slide"], adapter, config["seed"])
    val_dataset = H5BagDataset(val, bag_dir, config["tiles_per_slide"], adapter, config["seed"] + 1)
    test_dataset = H5BagDataset(test, bag_dir, config["tiles_per_slide"], adapter, config["seed"] + 2)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config["seed"])
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=2, pin_memory=True)
    model = ABMIL(feature_dim, config["mil_hidden_dim"]).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs = config["fast_dev_epochs"] if config["fast_dev_run"] else config["mil_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    checkpoint_dir = out / "checkpoints" / "camelyon" / model_name / representation / f"center{heldout_center}"
    start, best = resume_if_available(checkpoint_dir / "latest.pt", model, optimizer, scheduler, map_location=device)
    history = []
    for epoch in range(start, epochs):
        train_sampler.set_epoch(epoch); model.train(); total = 0.0; n = 0
        for batch in train_loader:
            features, labels = batch["features"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(features)["logits"]; loss = torch.nn.functional.cross_entropy(logits, labels); loss.backward(); optimizer.step(); total += float(loss); n += 1
        model.eval(); val_loss = torch.zeros(2, device=device)
        with torch.inference_mode():
            for batch in val_loader:
                logits = model(batch["features"].to(device))["logits"]
                val_loss += torch.tensor([float(torch.nn.functional.cross_entropy(logits, batch["label"].to(device))), 1], device=device)
        if world_size > 1: torch.distributed.all_reduce(val_loss)
        value = float((val_loss[0] / val_loss[1]).cpu()); scheduler.step(); improved = value < best; best = min(best, value)
        if is_main_process():
            history.append({"epoch": epoch, "train_loss": total / max(1, n), "val_loss": value}); save_training_checkpoint(checkpoint_dir, model, optimizer, scheduler, epoch, best, improved); pd.DataFrame(history).to_csv(out / "logs" / f"camelyon_{model_name}_{representation}_center{heldout_center}.csv", index=False)
    barrier()
    if is_main_process():
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)["model"]); target.eval()
        records = []
        mil_parameter_count = int(sum(parameter.numel() for parameter in target.parameters()))
        with torch.inference_mode():
            for item in DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0):
                probability = torch.softmax(target(item["features"].to(device))["logits"], dim=1)[0, 1].item()
                records.append({"slide_id": item["slide_id"][0], "patient_id": int(item["patient_id"]), "center": int(item["center"]), "label": int(item["label"]), "probability": probability, "model": model_name, "representation": representation, "mil_parameter_count": mil_parameter_count})
        predictions_dir = out / "predictions"; predictions_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(predictions_dir / f"camelyon_{model_name}_{representation}_center{heldout_center}.csv", index=False)
    barrier()
