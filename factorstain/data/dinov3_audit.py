from __future__ import annotations

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .domain_audit import read_midog_patch


class DINOAuditDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, dataset_name: str) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item: int) -> dict:
        row = self.metadata.iloc[item]
        if self.dataset_name == "plism":
            with Image.open(row.image_path) as opened:
                image = opened.convert("RGB")
        elif self.dataset_name == "midog21":
            image = read_midog_patch(row)
        else:
            raise KeyError(f"Unknown audit dataset {self.dataset_name!r}")
        return {"image": image, "sample_id": str(row.sample_id)}


class ProcessorCollator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        processed = self.processor(
            images=[item["image"] for item in batch], return_tensors="pt"
        )
        return {
            "pixel_values": processed["pixel_values"],
            "sample_id": [item["sample_id"] for item in batch],
        }
