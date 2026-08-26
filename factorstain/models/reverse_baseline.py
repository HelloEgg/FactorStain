from __future__ import annotations

import torch
from torch import nn

from .morphology import MorphologyEncoder
from .scanner_operator import ScannerOperator
from .stain_operator import ImageStainOperator, StainOperator


class ReverseFactorStain(nn.Module):
    """Ablation applying scanner rendering before stain rendering (Q→S)."""
    def __init__(self, num_stains: int, num_scanners: int, width: int = 64) -> None:
        super().__init__()
        self.encoder = MorphologyEncoder(width)
        self.neutral_renderer = StainOperator(1, channels=width)
        self.scanner_operator = ScannerOperator(num_scanners)
        self.stain_operator = ImageStainOperator(num_stains)

    def forward(self, source: torch.Tensor, stain_id: torch.Tensor, scanner_id: torch.Tensor) -> torch.Tensor:
        morphology = self.encoder(source)
        neutral = torch.zeros_like(stain_id)
        base, _ = self.neutral_renderer(morphology, neutral)
        scanned = self.scanner_operator(base, scanner_id)
        return self.stain_operator(scanned, stain_id)

