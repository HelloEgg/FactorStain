from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from .base import FeatureHarmonizer


class IdentityHarmonizer(FeatureHarmonizer):
    method_name = "raw_dinov3"

    def fit(self, features: np.ndarray, metadata: pd.DataFrame) -> None:
        return None

    def transform(self, features: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        return np.asarray(features, dtype=np.float32).copy()


class FEATMAPHarmonizer(FeatureHarmonizer):
    """Paired scanner-specific affine embedding mappings (FEATMAP adaptation)."""

    method_name = "featmap"

    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self.models: dict[str, Ridge] = {}

    def fit(self, features: np.ndarray, metadata: pd.DataFrame) -> None:
        frame = metadata.reset_index(drop=True).copy()
        if "is_training" in frame and not frame.is_training.all():
            raise ValueError("FEATMAP.fit received non-training rows")
        group_columns = ["aligned_group_id", "stain_id"]
        counts = frame.groupby("scanner_id").size().sort_values(ascending=False)
        if counts.empty:
            raise ValueError("No scanner labels for FEATMAP")
        self.reference_scanner = str(counts.index[0])
        reference = frame[frame.scanner_id.astype(str).eq(self.reference_scanner)]
        reference_lookup = {
            tuple(str(getattr(row, column)) for column in group_columns): row.Index
            for row in reference.itertuples()
        }
        for scanner, scanner_frame in frame.groupby("scanner_id", sort=True):
            scanner = str(scanner)
            if scanner == self.reference_scanner:
                continue
            pairs = []
            for row in scanner_frame.itertuples():
                key = tuple(str(getattr(row, column)) for column in group_columns)
                if key in reference_lookup:
                    pairs.append((row.Index, reference_lookup[key]))
            if len(pairs) < 2:
                continue
            source_positions, target_positions = zip(*pairs)
            model = Ridge(alpha=self.ridge, fit_intercept=True)
            model.fit(
                features[list(source_positions)], features[list(target_positions)]
            )
            self.models[scanner] = model

    def transform(self, features: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        output = np.asarray(features, dtype=np.float32).copy()
        for scanner, positions in metadata.groupby(
            "scanner_id", sort=True
        ).groups.items():
            model = self.models.get(str(scanner))
            if model is not None:
                indices = np.asarray(list(positions), dtype=int)
                output[indices] = model.predict(output[indices]).astype(np.float32)
        return output

    def save(self, path: str | Path) -> None:
        import pickle

        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    def load(self, path: str | Path) -> None:
        import pickle

        with Path(path).open("rb") as handle:
            restored = pickle.load(handle)
        self.__dict__.update(restored.__dict__)


class _Projection(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=-1)


class ScanGenHarmonizer(FeatureHarmonizer):
    """Paper-faithful ScanGen projection and contrastive attraction/repulsion loss."""

    method_name = "scangen"

    def __init__(
        self,
        hidden: int = 96,
        alpha: float = 0.16,
        radius: float = 1.0,
        biological_weight: float = 1.0,
        epochs: int = 200,
        seed: int = 42,
    ) -> None:
        self.hidden = hidden
        self.alpha = alpha
        self.radius = radius
        self.biological_weight = biological_weight
        self.epochs = epochs
        self.seed = seed

    def fit(self, features: np.ndarray, metadata: pd.DataFrame) -> None:
        if "is_training" in metadata and not metadata.is_training.all():
            raise ValueError("ScanGen.fit received non-training rows")
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
        self.model = _Projection(values.shape[1], min(self.hidden, values.shape[1])).to(
            device
        )
        tissues = pd.Categorical(metadata.tissue_type.astype(str))
        tissue_targets = torch.from_numpy(tissues.codes.astype(np.int64)).to(device)
        tissue_head = nn.Linear(
            min(self.hidden, values.shape[1]), len(tissues.categories)
        ).to(device)
        groups = metadata.aligned_group_id.astype(str).to_numpy()
        scanners = metadata.scanner_id.astype(str).to_numpy()
        stains = metadata.stain_id.astype(str).to_numpy()
        same_group = torch.from_numpy(groups[:, None] == groups[None, :]).to(device)
        same_scanner = torch.from_numpy(scanners[:, None] == scanners[None, :]).to(
            device
        )
        same_stain = torch.from_numpy(stains[:, None] == stains[None, :]).to(device)
        diagonal = torch.eye(len(values), dtype=torch.bool, device=device)
        # PLISM contains multiple stain sections per aligned group. Holding stain
        # fixed prevents the scanner loss from treating a real stain change as a
        # scanner effect.
        attract_mask = same_group & same_stain & ~same_scanner
        repel_mask = same_scanner & same_stain & ~same_group & ~diagonal
        if not attract_mask.any() or not repel_mask.any():
            raise ValueError(
                "ScanGen requires same-specimen multi-scanner training pairs"
            )
        optimizer = torch.optim.AdamW(
            [*self.model.parameters(), *tissue_head.parameters()],
            lr=1e-3,
            weight_decay=1e-4,
        )
        for _ in range(self.epochs):
            projected = self.model(values)
            distance = 1 - projected @ projected.T
            attraction = distance[attract_mask].square().mean()
            repulsion = F.relu(self.radius - distance[repel_mask]).square().mean()
            scan_loss = (1 - self.alpha) * attraction + self.alpha * repulsion
            biology_loss = F.cross_entropy(tissue_head(projected), tissue_targets)
            loss = scan_loss + self.biological_weight * biology_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.model = self.model.cpu().eval()
        self.tissue_categories = list(tissues.categories)

    def transform(self, features: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
        with torch.inference_mode():
            values = torch.from_numpy(np.asarray(features, dtype=np.float32))
            return self.model(values).numpy().astype(np.float32)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "input_dimension": self.model.network[0].in_features,
                "hidden": self.model.network[-1].out_features,
                "settings": {
                    "alpha": self.alpha,
                    "radius": self.radius,
                    "biological_weight": self.biological_weight,
                    "tissues": self.tissue_categories,
                },
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model = _Projection(payload["input_dimension"], payload["hidden"])
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        settings = payload["settings"]
        self.alpha = settings["alpha"]
        self.radius = settings["radius"]
        self.biological_weight = settings["biological_weight"]
        self.tissue_categories = settings["tissues"]


def pathorob_inspired_index(
    biological_ba: float,
    acquisition_ba: float,
    biological_classes: int,
    acquisition_classes: int,
) -> float:
    """Transparent PLISM analogue, deliberately not the official PathoROB RI."""
    biological_chance = 1 / max(1, biological_classes)
    acquisition_chance = 1 / max(1, acquisition_classes)
    biology = (biological_ba - biological_chance) / max(1e-8, 1 - biological_chance)
    leakage = (acquisition_ba - acquisition_chance) / max(1e-8, 1 - acquisition_chance)
    return float(np.clip(0.5 * (biology + (1 - leakage)), 0, 1))
