from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .morphology import MorphologyEncoder
from .stain_operator import ConditionalResidualBlock


class JointConditionalGenerator(nn.Module):
    """Lumped generator whose two IDs are fused before all rendering blocks."""
    def __init__(self, num_stains: int, num_scanners: int, width: int = 64, condition_dim: int = 96) -> None:
        super().__init__()
        self.encoder = MorphologyEncoder(width)
        self.stain_embedding = nn.Embedding(num_stains, condition_dim // 2)
        self.scanner_embedding = nn.Embedding(num_scanners, condition_dim // 2)
        self.fusion = nn.Sequential(nn.Linear(condition_dim, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.blocks = nn.ModuleList([ConditionalResidualBlock(width, condition_dim) for _ in range(6)])
        self.out = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 3, 1))

    def forward(self, source: torch.Tensor, stain_id: torch.Tensor, scanner_id: torch.Tensor) -> torch.Tensor:
        features = self.encoder(source)
        condition = self.fusion(torch.cat([self.stain_embedding(stain_id), self.scanner_embedding(scanner_id)], dim=1))
        for block in self.blocks:
            features = block(features, condition)
        return torch.sigmoid(self.out(features))


class ParallelFactorGenerator(nn.Module):
    """Separate factor embeddings injected simultaneously without ordered operators."""
    def __init__(self, num_stains: int, num_scanners: int, width: int = 64, condition_dim: int = 64) -> None:
        super().__init__()
        self.encoder = MorphologyEncoder(width)
        self.stain_embedding = nn.Embedding(num_stains, condition_dim)
        self.scanner_embedding = nn.Embedding(num_scanners, condition_dim)
        self.stain_blocks = nn.ModuleList([ConditionalResidualBlock(width, condition_dim) for _ in range(3)])
        self.scanner_blocks = nn.ModuleList([ConditionalResidualBlock(width, condition_dim) for _ in range(3)])
        self.out = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(), nn.Conv2d(width, 3, 1))

    def forward(self, source: torch.Tensor, stain_id: torch.Tensor, scanner_id: torch.Tensor) -> torch.Tensor:
        features = self.encoder(source)
        stain_features, scanner_features = features, features
        for block in self.stain_blocks:
            stain_features = block(stain_features, self.stain_embedding(stain_id))
        for block in self.scanner_blocks:
            scanner_features = block(scanner_features, self.scanner_embedding(scanner_id))
        return torch.sigmoid(self.out(0.5 * (stain_features + scanner_features)))
