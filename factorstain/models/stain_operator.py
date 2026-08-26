from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalResidualBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels, affine=False)
        self.norm2 = nn.GroupNorm(min(8, channels), channels, affine=False)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, channels * 4))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.film(condition).chunk(4, dim=1)
        apply = lambda value, gamma, beta: value * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        hidden = F.silu(apply(self.norm1(self.conv1(x)), gamma1, beta1))
        hidden = apply(self.norm2(self.conv2(hidden)), gamma2, beta2)
        return F.silu(x + hidden)


class StainOperator(nn.Module):
    """Conditional residual stain renderer operating on morphology features."""
    def __init__(self, num_stains: int, channels: int = 64, condition_dim: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_stains, condition_dim)
        self.blocks = nn.ModuleList([ConditionalResidualBlock(channels, condition_dim) for _ in range(blocks)])
        self.to_rgb = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(), nn.Conv2d(channels, 3, 1))

    def forward(self, morphology: torch.Tensor, stain_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.embedding(stain_id)
        features = morphology
        for block in self.blocks:
            features = block(features, condition)
        stain_image = torch.sigmoid(self.to_rgb(features))
        return stain_image, features


class ImageStainOperator(nn.Module):
    """Stain operator for the reverse baseline, where scanning is applied first."""
    def __init__(self, num_stains: int, channels: int = 48, condition_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_stains, condition_dim)
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([ConditionalResidualBlock(channels, condition_dim) for _ in range(3)])
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, image: torch.Tensor, stain_id: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.stem(image))
        condition = self.embedding(stain_id)
        for block in self.blocks:
            features = block(features, condition)
        return (image + 0.25 * torch.tanh(self.out(features))).clamp(0, 1)

