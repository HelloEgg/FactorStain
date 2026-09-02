from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(
    config_path: str | Path, paths_path: str | Path = "configs/paths.yaml"
) -> dict[str, Any]:
    """Load and merge path/milestone YAML with environment-aware root overrides."""
    with Path(paths_path).open("r", encoding="utf-8") as handle:
        paths = yaml.safe_load(handle) or {}
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    overrides = {
        "project_root": os.getenv("PROJECT_ROOT"),
        "data_root": os.getenv("DATA_ROOT"),
        "plism_root": os.getenv("PLISM_ROOT"),
        "midog_root": os.getenv("MIDOG_ROOT"),
        "camelyon_root": os.getenv("CAMELYON_ROOT"),
        "hf_home": os.getenv("HF_HOME"),
        "outputs_root": os.getenv("OUTPUTS_ROOT"),
    }
    for key, value in overrides.items():
        if value:
            paths[key] = value
    # A local checkout remains runnable even when the default cluster project path is absent.
    if not Path(paths.get("project_root", ".")).exists():
        paths["project_root"] = str(Path.cwd().resolve())
    if not os.getenv("OUTPUTS_ROOT"):
        paths["outputs_root"] = str(Path(paths["project_root"]) / "outputs")
    os.environ.setdefault("HF_HOME", str(paths["hf_home"]))
    merged = _deep_merge({"paths": paths}, config)
    declared_seeds = [
        int(seed) for seed in merged.get("seeds", [merged.get("seed", 42)])
    ]
    seed_override = os.getenv("SOTA_SEEDS", "").strip()
    if seed_override:
        run_seeds = [
            int(value.strip()) for value in seed_override.split(",") if value.strip()
        ]
        if not run_seeds:
            raise ValueError("SOTA_SEEDS must contain at least one integer seed")
        undeclared = sorted(set(run_seeds) - set(declared_seeds))
        if undeclared:
            raise ValueError(
                f"SOTA_SEEDS contains seeds not declared by the config: {undeclared}"
            )
    else:
        run_seeds = declared_seeds
    merged["run_seeds"] = list(dict.fromkeys(run_seeds))
    merged["fast_dev_run"] = os.getenv("FAST_DEV_RUN", "0") == "1"
    merged["force_continue"] = os.getenv("FORCE_CONTINUE", "0") == "1"
    merged["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
    return _expand(merged)


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
