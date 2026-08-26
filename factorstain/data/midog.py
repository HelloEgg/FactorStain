from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCANNER_RANGES = [
    (1, 50, "Hamamatsu XR"),
    (51, 100, "Hamamatsu S360"),
    (101, 150, "Aperio CS2"),
    (151, 200, "Leica GT450"),
]


def midog_scanner(case_id: int | str) -> str:
    number = int(Path(str(case_id)).stem)
    for lower, upper, scanner in SCANNER_RANGES:
        if lower <= number <= upper:
            return scanner
    raise ValueError(f"MIDOG case {case_id} falls outside the documented 001–200 mapping")


def parse_midog_annotations(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    annotation_path = root / "MIDOG.json"
    if not annotation_path.exists():
        matches = list(root.rglob("MIDOG.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one MIDOG.json under {root}; found {len(matches)}")
        annotation_path = matches[0]
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload.get("images", [])}
    categories = {item["id"]: str(item.get("name", item["id"])) for item in payload.get("categories", [])}
    rows = []
    for annotation in payload.get("annotations", []):
        image = images[annotation["image_id"]]
        case_id = Path(image.get("file_name", str(image["id"]))).stem
        declared_path = Path(image["file_name"])
        direct_path = declared_path if declared_path.is_absolute() else root / declared_path
        if not direct_path.exists():
            matches = list(root.rglob(declared_path.name))
            if len(matches) != 1:
                raise FileNotFoundError(f"Could not uniquely resolve MIDOG image {declared_path} under {root}")
            direct_path = matches[0]
        bbox = annotation.get("bbox", [0, 0, 0, 0])
        rows.append(
            {
                "case_id": case_id,
                "image_path": str(direct_path.resolve()),
                "x": float(bbox[0] + bbox[2] / 2),
                "y": float(bbox[1] + bbox[3] / 2),
                "category_id": int(annotation.get("category_id", 1)),
                "category_name": categories.get(annotation.get("category_id", 1), str(annotation.get("category_id", 1))),
                "label": int("hard" not in categories.get(annotation.get("category_id", 1), "").lower()),
                "scanner_id": midog_scanner(case_id),
            }
        )
    return pd.DataFrame(rows)
