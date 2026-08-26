from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def distributed_context() -> tuple[int, int, int]:
    return (
        int(os.getenv("RANK", "0")),
        int(os.getenv("LOCAL_RANK", "0")),
        int(os.getenv("WORLD_SIZE", "1")),
    )


def init_distributed(require_cuda: bool = True) -> tuple[torch.device, int, int, int]:
    rank, local_rank, world_size = distributed_context()
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this neural milestone; no GPU is visible. "
            "Set CUDA_VISIBLE_DEVICES=0,1,2,3 and verify the PyTorch CUDA build."
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return device, rank, local_rank, world_size


def is_main_process() -> bool:
    return int(os.getenv("RANK", "0")) == 0


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def collect_provenance(seed: int, dataset_counts: dict[str, int] | None = None) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    cuda_version = torch.version.cuda
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_names": gpu_names,
        "cuda_version": cuda_version,
        "pytorch_version": torch.__version__,
        "seed": seed,
        "dataset_counts": dataset_counts or {},
    }


def atomic_json_dump(payload: Any, destination: str | Path) -> None:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, np.ndarray):
            return sanitize(value.tolist())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2, allow_nan=False)
    os.replace(temporary, destination)


def ensure_disk_space(path: str | Path, minimum_gb: float = 5.0) -> None:
    import shutil

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB is free under {path}; at least {minimum_gb:.1f} GB is required."
        )
