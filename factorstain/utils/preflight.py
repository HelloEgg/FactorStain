from __future__ import annotations

import os
from pathlib import Path

import torch

from .runtime import ensure_disk_space


def preflight(
    config: dict,
    dataset_keys: tuple[str, ...] = (),
    require_cuda: bool = False,
    required_milestone: str | None = None,
) -> None:
    errors: list[str] = []
    for key in dataset_keys:
        path = Path(config["paths"][key])
        if not path.exists():
            errors.append(f"Dataset path does not exist: {key}={path}")
    if require_cuda and not torch.cuda.is_available():
        errors.append("No CUDA GPU is visible to PyTorch.")
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if require_cuda and not visible:
        errors.append("CUDA_VISIBLE_DEVICES is empty.")
    if required_milestone:
        decision = Path(config["paths"]["outputs_root"]) / required_milestone / "GO_NOGO.json"
        if not decision.exists():
            errors.append(f"Required prior milestone output is missing: {decision}")
    if errors:
        actions = [
            "Edit configs/paths.yaml or export the corresponding *_ROOT variable.",
            "For GPU failures, export CUDA_VISIBLE_DEVICES=0,1,2,3 and verify torch.cuda.is_available().",
        ]
        raise RuntimeError("ERROR:\n" + "\n".join(errors) + "\n\nAction:\n" + "\n".join(actions))
    ensure_disk_space(config["paths"]["outputs_root"], 0.1 if config["fast_dev_run"] else 5.0)

