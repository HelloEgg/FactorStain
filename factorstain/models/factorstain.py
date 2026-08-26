from __future__ import annotations

import torch
from torch import nn

from .morphology import MorphologyEncoder
from .scanner_operator import ScannerOperator
from .stain_operator import StainOperator


class FactorStain(nn.Module):
    """Explicit physically ordered morphology → stain → scanner generator."""
    def __init__(self, num_stains: int, num_scanners: int, width: int = 64) -> None:
        super().__init__()
        self.morphology_encoder = MorphologyEncoder(width=width)
        self.stain_operator = StainOperator(num_stains, channels=width)
        self.scanner_operator = ScannerOperator(num_scanners)

    def forward(
        self,
        source: torch.Tensor,
        stain_id: torch.Tensor,
        scanner_id: torch.Tensor,
        return_intermediates: bool = False,
    ):
        morphology = self.morphology_encoder(source)
        stained, stain_features = self.stain_operator(morphology, stain_id)
        rendered = self.scanner_operator(stained, scanner_id)
        if return_intermediates:
            return {"image": rendered, "stain_image": stained, "morphology": morphology, "stain_features": stain_features}
        return rendered

