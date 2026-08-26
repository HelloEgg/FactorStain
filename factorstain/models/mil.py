from __future__ import annotations

import torch
from torch import nn


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.value = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(self.value(features) * self.gate(features)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1)


class ABMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25))
        self.attention = GatedAttention(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, bag: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        projected = self.project(bag)
        weights = self.attention(projected, mask)
        pooled = torch.einsum("bn,bnd->bd", weights, projected)
        return {"logits": self.classifier(pooled), "attention": weights, "embedding": pooled}

