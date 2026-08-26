from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def rgb_to_od(image: torch.Tensor, eps: float = 1.0 / 255.0) -> torch.Tensor:
    return -torch.log(image.clamp(min=eps, max=1.0))


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvNormAct(channels, channels), nn.Conv2d(channels, channels, 3, padding=1))
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x + self.block(x)))


class MorphologyEncoder(nn.Module):
    """OD-aware spatial encoder biased toward gradients and local topology."""
    def __init__(self, width: int = 64, blocks: int = 4) -> None:
        super().__init__()
        # RGB + optical density + luminance gradient magnitude.
        self.stem = ConvNormAct(7, width)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.projection = nn.Conv2d(width, width, 1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        od = rgb_to_od(image)
        gray = image.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        gradient = torch.sqrt(gx.square() + gy.square() + 1e-8)
        return self.projection(self.body(self.stem(torch.cat([image, od, gradient], dim=1))))

