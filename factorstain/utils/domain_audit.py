from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import load_config, save_resolved_config


def load_domain_audit_config(
    path: str | Path = "configs/m_minus1_domain_audit.yaml",
) -> dict[str, Any]:
    config = load_config(path)
    if os.getenv("DINOV3_MODEL"):
        config["dinov3"]["model_name"] = os.environ["DINOV3_MODEL"]
    config["force_reextract"] = os.getenv("FORCE_REEXTRACT", "0") == "1"
    config["resample"] = os.getenv("RESAMPLE", "0") == "1"
    return config


def prepare_domain_audit_output(config: dict[str, Any]) -> Path:
    out = Path(config["paths"]["outputs_root"]) / config["milestone"]
    for child in (
        "logs",
        "metadata",
        "features",
        "figures",
        "figures/plism",
        "figures/midog21",
    ):
        (out / child).mkdir(parents=True, exist_ok=True)
    # torchrun invokes this helper on every rank; only rank zero owns shared metadata.
    if int(os.getenv("RANK", "0")) == 0:
        save_resolved_config(config, out / "config_resolved.yaml")
    return out
