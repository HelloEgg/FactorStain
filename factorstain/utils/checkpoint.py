from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    is_best: bool,
    scaler: Any = None,
) -> None:
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    directory = Path(directory)
    atomic_torch_save(payload, directory / "latest.pt")
    if is_best:
        atomic_torch_save(payload, directory / "best.pt")


def resume_if_available(
    checkpoint: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float]:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return 0, float("inf")
    state = torch.load(checkpoint, map_location=map_location, weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return int(state["epoch"]) + 1, float(state["best_metric"])

