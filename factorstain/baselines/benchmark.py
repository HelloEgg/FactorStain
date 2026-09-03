from __future__ import annotations

from pathlib import Path


def benchmark_milestone(config: dict) -> str:
    """Keep diagnostic FAST_DEV artifacts separate from scientific results."""
    milestone = str(config["milestone"])
    if config.get("fast_dev_run", False) and not milestone.endswith("_fast_dev"):
        return f"{milestone}_fast_dev"
    return milestone


def benchmark_output(config: dict) -> Path:
    return Path(config["paths"]["outputs_root"]) / benchmark_milestone(config)


def benchmark_output_config(config: dict) -> dict:
    resolved = dict(config)
    resolved["milestone"] = benchmark_milestone(config)
    return resolved
