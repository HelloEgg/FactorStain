from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_dtype(name: str, device: torch.device | str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in choices:
        raise ValueError(
            f"Unsupported DINOv3 dtype {name!r}; choose bfloat16, float16, or float32"
        )
    dtype = choices[normalized]
    if (
        torch.device(device).type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        return torch.float16
    return dtype


def load_dinov3(
    model_name: str, device: torch.device | str, dtype_name: str = "bfloat16"
):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "Official DINOv3 requires transformers>=4.56; run bash shell/setup.sh"
        ) from exc
    token = os.getenv("HF_TOKEN") or None
    dtype = resolve_dtype(dtype_name, device)
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, token=token)
        model = AutoModel.from_pretrained(model_name, token=token, torch_dtype=dtype)
    except Exception as exc:
        message = str(exc).lower()
        if any(
            value in message
            for value in ("401", "403", "gated", "unauthorized", "access")
        ):
            raise RuntimeError(
                f"DINOv3 model access failed for {model_name}. Accept the Meta/Hugging Face terms, then export HF_TOKEN=... and rerun. Original error: {exc}"
            ) from exc
        raise RuntimeError(
            f"Unable to load official DINOv3 model {model_name}: {exc}"
        ) from exc
    model.eval().requires_grad_(False).to(device)
    return processor, model, dtype


def select_image_embedding(outputs: Any, model_config: Any) -> tuple[torch.Tensor, str]:
    pooled = getattr(outputs, "pooler_output", None)
    if (
        pooled is not None
        and pooled.ndim == 2
        and pooled.shape[0] > 0
        and torch.isfinite(pooled).all()
    ):
        return pooled, "pooler_output"
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None or hidden.ndim != 3:
        raise RuntimeError(
            "DINOv3 output exposes neither a valid pooler_output nor 3D last_hidden_state"
        )
    register_tokens = int(getattr(model_config, "num_register_tokens", 0))
    patch_tokens = hidden[:, 1 + register_tokens :, :]
    if patch_tokens.shape[1] == 0:
        raise RuntimeError("DINOv3 fallback pooling found no patch tokens")
    return patch_tokens.mean(
        dim=1
    ), f"mean_patch_tokens_excluding_{register_tokens}_register_tokens"


def manifest_signature(frame, model_name: str) -> str:
    columns = [
        column
        for column in (
            "sample_id",
            "image_path",
            "x",
            "y",
            "level",
            "read_size",
            "patch_size",
        )
        if column in frame
    ]
    content = (
        frame[columns]
        .fillna("")
        .astype(str)
        .sort_values("sample_id")
        .to_csv(index=False)
    )
    return hashlib.sha256((model_name + "\n" + content).encode("utf-8")).hexdigest()


def save_feature_cache(
    path: str | Path,
    features: np.ndarray,
    sample_ids: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        features=np.asarray(features, dtype=np.float32),
        sample_id=np.asarray(sample_ids, dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, path)


def load_feature_cache(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        sample_ids = payload["sample_id"].astype(str)
        metadata = json.loads(str(payload["metadata_json"].item()))
    if len(features) != len(sample_ids):
        raise ValueError(
            f"Feature cache row mismatch in {path}: {len(features)} features vs {len(sample_ids)} IDs"
        )
    return features, sample_ids, metadata


def processor_description(
    processor: Any, model: torch.nn.Module, pooling: str, feature_dim: int
) -> dict[str, Any]:
    def serializable(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return list(value)
        if isinstance(value, dict):
            return value
        return str(value)

    return {
        "model_name": getattr(model.config, "_name_or_path", "unknown"),
        "feature_dimension": int(feature_dim),
        "pooling_strategy": pooling,
        "processor_class": processor.__class__.__name__,
        "image_mean": serializable(getattr(processor, "image_mean", None)),
        "image_std": serializable(getattr(processor, "image_std", None)),
        "processor_size": serializable(getattr(processor, "size", None)),
        "crop_size": serializable(getattr(processor, "crop_size", None)),
        "input_resolution": serializable(getattr(model.config, "image_size", None)),
        "patch_size": serializable(getattr(model.config, "patch_size", None)),
        "num_register_tokens": int(getattr(model.config, "num_register_tokens", 0)),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }
