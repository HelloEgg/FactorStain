from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


class BlockedModelAccess(RuntimeError):
    pass


@dataclass
class FoundationModel:
    name: str
    model: nn.Module
    transform: Callable
    feature_dim: int


REGISTRY = {
    "uni": {"repo": "hf-hub:MahmoodLab/UNI", "model": "vit_large_patch16_224"},
    "virchow2": {
        "repo": "hf-hub:paige-ai/Virchow2",
        "model": "vit_huge_patch14_224",
    },
}


def load_foundation_model(name: str, device: torch.device | str = "cuda") -> FoundationModel:
    name = name.lower()
    if name not in REGISTRY:
        raise KeyError(f"Unknown pathology foundation model {name!r}; available: {sorted(REGISTRY)}")
    try:
        import timm
        from timm.data import create_transform, resolve_data_config

        spec = REGISTRY[name]
        if name == "virchow2":
            from timm.layers import SwiGLUPacked
            model = timm.create_model(spec["repo"], pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        else:
            model = timm.create_model(spec["repo"], pretrained=True, init_values=1e-5, dynamic_img_size=True)
        model.eval().requires_grad_(False).to(device)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model), is_training=False)
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feature = normalize_model_output(model(dummy), name)
        return FoundationModel(name, model, transform, int(feature.shape[-1]))
    except Exception as exc:
        message = str(exc).lower()
        if any(token in message for token in ("401", "403", "gated", "unauthorized", "access request")):
            raise BlockedModelAccess(
                f"BLOCKED_MODEL_ACCESS: {name} is gated or unauthorized. Export HF_TOKEN after accepting access for {REGISTRY[name]['repo']}."
            ) from exc
        raise RuntimeError(f"Failed to load the declared {name} model ({REGISTRY[name]['repo']}): {exc}") from exc


def normalize_model_output(output, name: str) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("x_norm_clstoken", output.get("last_hidden_state", next(iter(output.values()))))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 3:
        if name == "virchow2" and output.shape[1] >= 5:
            cls = output[:, 0]
            patch = output[:, 5:].mean(dim=1)
            return torch.cat([cls, patch], dim=-1)
        return output[:, 0]
    if output.ndim == 4:
        return output.mean(dim=(-2, -1))
    return output


@torch.inference_mode()
def encode_batch(foundation_model: FoundationModel, images: torch.Tensor) -> torch.Tensor:
    output = foundation_model.model(images)
    return normalize_model_output(output, foundation_model.name).float()


def preprocess_tensor_batch(foundation_model: FoundationModel, images: torch.Tensor) -> torch.Tensor:
    """Apply the model's declared inference resize/normalization to a [0,1] tensor batch."""
    from torch.nn import functional as F

    config = foundation_model.model.pretrained_cfg
    input_size = config.get("input_size", (3, 224, 224))[-2:]
    resized = F.interpolate(images, size=input_size, mode="bicubic", align_corners=False, antialias=True)
    mean = torch.as_tensor(config.get("mean", (0.485, 0.456, 0.406)), device=images.device, dtype=images.dtype)[None, :, None, None]
    std = torch.as_tensor(config.get("std", (0.229, 0.224, 0.225)), device=images.device, dtype=images.dtype)[None, :, None, None]
    return (resized - mean) / std
