from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.weight * grad_output, None


def gradient_reverse(x: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, weight)


class FactorAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_tissues: int,
        num_stains: int,
        num_scanners: int,
        bottleneck_dim: int = 256,
        grl_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl_weight = grl_weight
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, feature_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.tissue_head = nn.Linear(feature_dim, num_tissues)
        self.stain_adversary = nn.Linear(feature_dim, num_stains)
        self.scanner_adversary = nn.Linear(feature_dim, num_scanners)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.scale * self.adapter(features)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        adapted = self.encode(features)
        reversed_features = gradient_reverse(adapted, self.grl_weight)
        return {
            "features": adapted,
            "tissue_logits": self.tissue_head(adapted),
            "stain_logits": self.stain_adversary(reversed_features),
            "scanner_logits": self.scanner_adversary(reversed_features),
        }

