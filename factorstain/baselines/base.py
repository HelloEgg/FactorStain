from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


class MethodUnavailable(RuntimeError):
    """Raised when an optional official implementation or weight is unavailable."""


@dataclass
class FitContext:
    """Training-only material passed to every image acquisition method."""

    train_index: pd.DataFrame
    reference_policy: dict[str, Any]
    scanner_pairs: pd.DataFrame
    output_dir: Path
    image_size: int
    seed: int
    fast_dev_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AcquisitionMethod(ABC):
    """Common interface for image-level acquisition interventions."""

    method_name: str

    @abstractmethod
    def fit(self, context: FitContext, val_data: pd.DataFrame | None = None) -> None:
        """Fit using training-only data. Held-out joint targets are forbidden."""

    @abstractmethod
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
        """Return uint8 RGB output for one declared evaluation track."""

    def supports_image_output(self) -> bool:
        return True

    def supports_feature_output(self) -> bool:
        return False

    def supports_strict_composition(self) -> bool:
        return False

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.method_name} does not implement save()")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.method_name} does not implement load()")


class FeatureHarmonizer(ABC):
    """Common interface for Track-D embedding-space interventions."""

    method_name: str

    @abstractmethod
    def fit(self, features: np.ndarray, metadata: pd.DataFrame) -> None:
        """Fit only on rows explicitly labeled as training rows."""

    @abstractmethod
    def transform(self, features: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        """Transform features without accessing evaluation labels."""

    def supports_image_output(self) -> bool:
        return False

    def supports_feature_output(self) -> bool:
        return True

    def supports_strict_composition(self) -> bool:
        return False

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.method_name} does not implement save()")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(f"{self.method_name} does not implement load()")
