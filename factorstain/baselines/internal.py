from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from factorstain.training.renderer import build_renderer

from .base import AcquisitionMethod, FitContext, MethodUnavailable
from .classical import as_uint8_rgb


class RendererAcquisitionMethod(AcquisitionMethod):
    """Common-API adapter for the exact existing M1 checkpoints."""

    def __init__(self, method_name: str, checkpoint: str | Path, width: int = 64):
        if method_name not in {"joint", "parallel", "factorstain"}:
            raise KeyError(method_name)
        self.method_name = method_name
        self.checkpoint = Path(checkpoint)
        self.width = width

    def fit(self, context: FitContext, val_data: pd.DataFrame | None = None) -> None:
        if not self.checkpoint.exists():
            raise MethodUnavailable(
                f"Existing M1 checkpoint is missing: {self.checkpoint}"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stains = sorted(context.train_index.stain_id.astype(str).unique())
        scanners = sorted(context.train_index.scanner_id.astype(str).unique())
        self.stain_map = {value: position for position, value in enumerate(stains)}
        self.scanner_map = {value: position for position, value in enumerate(scanners)}
        self.image_size = context.image_size
        self.model = build_renderer(
            self.method_name, len(stains), len(scanners), self.width
        ).to(self.device)
        payload = torch.load(
            self.checkpoint, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    @torch.inference_mode()
    def translate(
        self,
        source_image: Image.Image | np.ndarray,
        source_stain: str,
        source_scanner: str,
        target_stain: str,
        target_scanner: str,
        *,
        track: str = "C",
    ) -> np.ndarray:
        if track == "A":
            target_scanner = source_scanner
        elif track == "B":
            target_stain = source_stain
        rgb = as_uint8_rgb(source_image)
        image = Image.fromarray(rgb).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        tensor = (
            torch.from_numpy(np.asarray(image).copy())
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(self.device)
        )
        stain = torch.tensor([self.stain_map[str(target_stain)]], device=self.device)
        scanner = torch.tensor(
            [self.scanner_map[str(target_scanner)]], device=self.device
        )
        generated = self.model(tensor, stain, scanner)[0]
        return generated.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()

    def supports_strict_composition(self) -> bool:
        return True

    def load(self, path: str | Path) -> None:
        self.checkpoint = Path(path)
