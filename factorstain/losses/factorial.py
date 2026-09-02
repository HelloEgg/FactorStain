from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def edge_consistency_per_sample(
    generated: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    gx, gy = image_gradients(generated.mean(1, keepdim=True))
    tx, ty = image_gradients(target.mean(1, keepdim=True))
    return (gx - tx).abs().mean(dim=(1, 2, 3)) + (gy - ty).abs().mean(dim=(1, 2, 3))


def edge_consistency_loss(
    generated: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return edge_consistency_per_sample(generated, target).mean()


def differentiable_ssim_loss(
    generated: torch.Tensor, target: torch.Tensor, window: int = 7
) -> torch.Tensor:
    """Return a stable per-sample 1-SSIM approximation."""
    padding = window // 2
    mu_x = F.avg_pool2d(generated, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(generated.square(), window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(generated * target, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-8)
    return 1 - ssim.clamp(-1, 1).mean(dim=(1, 2, 3))


def appearance_statistics_per_sample(
    generated: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    generated_od = -torch.log(generated.clamp(min=1 / 255))
    target_od = -torch.log(target.clamp(min=1 / 255))
    losses = generated.new_zeros(len(generated))
    for source_values, target_values in (
        (generated, target),
        (generated_od, target_od),
    ):
        source_mean = source_values.mean(dim=(-2, -1))
        target_mean = target_values.mean(dim=(-2, -1))
        source_std = source_values.std(dim=(-2, -1))
        target_std = target_values.std(dim=(-2, -1))
        losses = losses + (source_mean - target_mean).abs().mean(1)
        losses = losses + (source_std - target_std).abs().mean(1)
    return losses


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, anchor: torch.Tensor
) -> torch.Tensor:
    return values[mask].mean() if mask.any() else anchor.sum() * 0.0


class FactorStainLoss(nn.Module):
    """Pair-type-aware objective respecting the PLISM serial-section caveat."""

    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        scanner_pair_mask: torch.Tensor,
        semantic_generated: torch.Tensor | None = None,
        semantic_target: torch.Tensor | None = None,
        semantic_source: torch.Tensor | None = None,
        lpips_per_sample: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scanner_pair_mask = scanner_pair_mask.bool()
        cross_stain_mask = ~scanner_pair_mask
        pixel = (generated - target).abs().mean(dim=(1, 2, 3))
        scanner_ssim = differentiable_ssim_loss(generated, target)
        scanner_edge = edge_consistency_per_sample(generated, target)
        cross_edge = edge_consistency_per_sample(generated, source)
        cross_gray = differentiable_ssim_loss(
            generated.mean(1, keepdim=True), source.mean(1, keepdim=True)
        )
        losses = {
            "scanner_pixel": _masked_mean(pixel, scanner_pair_mask, generated),
            "scanner_ssim": _masked_mean(scanner_ssim, scanner_pair_mask, generated),
            "scanner_edge": _masked_mean(scanner_edge, scanner_pair_mask, generated),
            "cross_stain_statistics": _masked_mean(
                appearance_statistics_per_sample(generated, target),
                cross_stain_mask,
                generated,
            ),
            "cross_stain_edge": _masked_mean(cross_edge, cross_stain_mask, generated),
            "cross_stain_grayscale": _masked_mean(
                cross_gray, cross_stain_mask, generated
            ),
        }
        if lpips_per_sample is not None:
            losses["scanner_lpips"] = _masked_mean(
                lpips_per_sample.flatten(), scanner_pair_mask, generated
            )
        else:
            losses["scanner_lpips"] = generated.sum() * 0.0
        if semantic_generated is not None and semantic_target is not None:
            distance = 1 - F.cosine_similarity(semantic_generated, semantic_target)
            losses["scanner_dino"] = _masked_mean(
                distance, scanner_pair_mask, generated
            )
        else:
            losses["scanner_dino"] = generated.sum() * 0.0
        if semantic_generated is not None and semantic_source is not None:
            distance = 1 - F.cosine_similarity(semantic_generated, semantic_source)
            losses["cross_stain_dino_morphology"] = _masked_mean(
                distance, cross_stain_mask, generated
            )
        else:
            losses["cross_stain_dino_morphology"] = generated.sum() * 0.0
        losses["total"] = sum(
            self.weights.get(name, 0.0) * value for name, value in losses.items()
        )
        return losses
