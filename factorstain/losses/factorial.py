from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def edge_consistency_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gx, gy = image_gradients(generated.mean(1, keepdim=True))
    tx, ty = image_gradients(target.mean(1, keepdim=True))
    return F.l1_loss(gx, tx) + F.l1_loss(gy, ty)


def factorial_swap_loss(predicted_by: torch.Tensor, target_by: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(predicted_by, target_by)


class FactorStainLoss(nn.Module):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        factorial_generated: torch.Tensor | None = None,
        factorial_target: torch.Tensor | None = None,
        semantic_generated: torch.Tensor | None = None,
        semantic_target: torch.Tensor | None = None,
        stain_logits: torch.Tensor | None = None,
        scanner_logits: torch.Tensor | None = None,
        stain_target: torch.Tensor | None = None,
        scanner_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {
            "reconstruction": F.l1_loss(generated, target),
            "morphology": edge_consistency_loss(generated, target),
        }
        if factorial_generated is not None and factorial_target is not None:
            losses["factorial"] = factorial_swap_loss(factorial_generated, factorial_target)
        else:
            losses["factorial"] = generated.new_zeros(())
        if semantic_generated is not None and semantic_target is not None:
            losses["morphology"] = losses["morphology"] + (1 - F.cosine_similarity(semantic_generated, semantic_target).mean())
        correctness = generated.new_zeros(())
        if stain_logits is not None and stain_target is not None:
            correctness = correctness + F.cross_entropy(stain_logits, stain_target)
        if scanner_logits is not None and scanner_target is not None:
            correctness = correctness + F.cross_entropy(scanner_logits, scanner_target)
        if stain_logits is None and scanner_logits is None:
            # Differentiable independent appearance constraints: RGB/OD moments must match the real target.
            generated_od = -torch.log(generated.clamp(min=1 / 255))
            target_od = -torch.log(target.clamp(min=1 / 255))
            for source_values, target_values in ((generated, target), (generated_od, target_od)):
                correctness = correctness + F.l1_loss(source_values.mean(dim=(-2, -1)), target_values.mean(dim=(-2, -1)))
                correctness = correctness + F.l1_loss(source_values.std(dim=(-2, -1)), target_values.std(dim=(-2, -1)))
        losses["factor_correctness"] = correctness
        # Isolation is measured through unchanged morphology; direct target/non-target classifier deltas are used at evaluation.
        losses["isolation"] = edge_consistency_loss(generated, target)
        losses["total"] = sum(self.weights.get(name, 0.0) * value for name, value in losses.items())
        return losses
