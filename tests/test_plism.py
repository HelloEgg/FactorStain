from pathlib import Path

import pandas as pd
from PIL import Image

from factorstain.data.plism import build_plism_index, discover_plism_metadata


def test_official_plism_metadata_columns_are_inferred(tmp_path: Path):
    for stain in ("A", "B"):
        for device in ("X", "Y"):
            path = tmp_path / f"{stain}_{device}_100_200.png"
            Image.new("RGB", (8, 8), "white").save(path)
    metadata = pd.DataFrame(
        [
            {
                "Tissue Type": "breast",
                "Stain Type": stain,
                "Device Type": device,
                "Coordinate": "100_200",
                "Image Path": f"{stain}_{device}_100_200.png",
            }
            for stain in ("A", "B")
            for device in ("X", "Y")
        ]
    )
    metadata.to_csv(tmp_path / "quality_control_list.csv", index=False)
    assert discover_plism_metadata(tmp_path).name == "quality_control_list.csv"
    index = build_plism_index(tmp_path)
    assert index.image_exists.all()
    assert index.aligned_group_id.nunique() == 1
    assert index.stain_id.nunique() == 2
    assert index.scanner_id.nunique() == 2
