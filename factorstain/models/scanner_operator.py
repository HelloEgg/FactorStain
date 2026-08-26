from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .stain_operator import ConditionalResidualBlock


class ScannerOperator(nn.Module):
    """Small conditional image operator modeling scanner response and local transfer."""
    def __init__(self, num_scanners: int, channels: int = 48, condition_dim: int = 64, blocks: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_scanners, condition_dim)
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([ConditionalResidualBlock(channels, condition_dim) for _ in range(blocks)])
        self.response = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(), nn.Conv2d(channels, 3, 3, padding=1))
        # Per-scanner color mixing approximates device spectral response.
        self.color_matrix = nn.Embedding(num_scanners, 12)
        with torch.no_grad():
            identity = torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=1).flatten()
            self.color_matrix.weight.copy_(identity.repeat(num_scanners, 1))

    def forward(self, stain_image: torch.Tensor, scanner_id: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.stem(stain_image))
        condition = self.embedding(scanner_id)
        for block in self.blocks:
            features = block(features, condition)
        residual = 0.20 * torch.tanh(self.response(features))
        affine = self.color_matrix(scanner_id).view(-1, 3, 4)
        pixels = stain_image.permute(0, 2, 3, 1)
        mixed = torch.einsum("bhwc,bkc->bhwk", pixels, affine[:, :, :3]) + affine[:, None, None, :, 3]
        return (mixed.permute(0, 3, 1, 2) + residual).clamp(0, 1)

