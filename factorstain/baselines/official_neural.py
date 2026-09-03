from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageFilter
from scipy.spatial import cKDTree
from torch import nn

from factorstain.utils.runtime import atomic_json_dump

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "external_neural_baselines.yaml"
SUPPORTED_METHODS = {
    "stainnet",
    "staingan",
    "cyclegan",
    "pix2pix",
    "histaugan",
    "cagan",
    "sastaindiff",
}
STAIN_METHODS = SUPPORTED_METHODS - {"pix2pix"}
SOURCE_DIRS = {
    "stainnet": "StainNet",
    "staingan": "StainGAN",
    "cyclegan": "pytorch-CycleGAN-and-pix2pix",
    "pix2pix": "pytorch-CycleGAN-and-pix2pix",
    "histaugan": "HistAuGAN",
    "cagan": "CAGAN",
    "sastaindiff": "SAStainDiff",
}
TRAINING_PROTOCOL_REVISIONS = {
    "stainnet": 1,
    "staingan": 1,
    "cyclegan": 1,
    "pix2pix": 1,
    "histaugan": 2,
    "cagan": 1,
    "sastaindiff": 1,
}


class AdapterError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_ids(frame: pd.DataFrame) -> pd.Series:
    column = "sample_id" if "sample_id" in frame else "image_id"
    if column not in frame:
        raise AdapterError("Training manifest needs sample_id or image_id")
    return frame[column].astype(str)


def _safe_name(value: str) -> str:
    prefix = "".join(char if char.isalnum() else "_" for char in str(value))[:32]
    suffix = hashlib.sha256(str(value).encode()).hexdigest()[:10]
    return f"{prefix}_{suffix}"


def _read_settings(method: str, fast_dev: bool) -> tuple[dict, dict]:
    path = Path(os.getenv("FACTORSTAIN_EXTERNAL_BASELINE_CONFIG", str(CONFIG_PATH)))
    if not path.is_file():
        raise AdapterError(f"External baseline config is missing: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    common = dict(config["common"])
    settings = {**common, **dict(config[method])}
    if fast_dev:
        dev = dict(config["fast_dev"])
        common.update(
            {
                key: dev[key]
                for key in (
                    "max_train_images_per_domain",
                    "max_validation_images_per_domain",
                )
            }
        )
        settings.update(common)
        settings["batch_size"] = min(settings["batch_size"], dev["batch_size"])
        settings["steps_per_epoch"] = dev["steps_per_epoch"]
        if "epochs" in settings:
            settings["epochs"] = dev["epochs"]
        if "epochs_constant" in settings:
            settings["epochs_constant"] = dev["epochs"]
            settings["epochs_decay"] = 0
        if "train_steps" in settings:
            settings["train_steps"] = dev["train_steps"]
            settings["model_channels"] = dev["sastaindiff_model_channels"]
            settings["num_res_blocks"] = dev["sastaindiff_num_res_blocks"]
    return config, settings


def _source_commit(method: str) -> tuple[Path, str]:
    from factorstain.baselines.registry import BASELINES

    root = PROJECT_ROOT / "third_party" / SOURCE_DIRS[method]
    if not root.is_dir() or not any(root.iterdir()):
        raise AdapterError(
            f"Pinned source for {method} is missing at {root}. Run: "
            f"python third_party/fetch_baselines.py --methods {method}"
        )
    spec = BASELINES[method]
    marker = root / ".factorstain-source.json"
    observed = ""
    if (root / ".git").exists():
        try:
            observed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            observed = ""
    elif marker.is_file():
        observed = str(json.loads(marker.read_text(encoding="utf-8"))["source_commit"])
    if observed != spec.source_commit:
        raise AdapterError(
            f"{method} source commit is {observed or 'unverified'}, expected "
            f"{spec.source_commit}. Re-run third_party/fetch_baselines.py."
        )
    return root, observed


def _previous_fit_is_compatible(
    previous: dict,
    *,
    method: str,
    protocol_revision: int,
    source_commit: str,
    seed: int,
    image_size: int,
    fast_dev_run: bool,
    settings: dict,
    input_hashes: dict[str, str],
) -> bool:
    """Allow old checkpoints to survive inference-only adapter changes."""
    previous_revision = int(previous.get("training_protocol_revision", 1))
    return (
        previous.get("method") == method
        and previous_revision == protocol_revision
        and previous.get("official_source_commit") == source_commit
        and previous.get("seed") == seed
        and previous.get("image_size") == image_size
        and previous.get("fast_dev_run") == fast_dev_run
        and previous.get("settings") == settings
        and previous.get("training_manifest_sha256")
        == input_hashes["train_manifest"]
        and previous.get("validation_manifest_sha256")
        == input_hashes["validation_manifest"]
        and previous.get("reference_policy_sha256")
        == input_hashes["reference_policy"]
        and previous.get("scanner_fit_pairs_sha256")
        == input_hashes["scanner_fit_pairs"]
    )


def _load_fit_request(method: str, request: dict) -> dict:
    required = (
        "train_manifest",
        "validation_manifest",
        "reference_policy",
        "scanner_fit_pairs",
        "checkpoint_dir",
        "forbidden_target_ids_sha256",
    )
    missing = [key for key in required if not request.get(key)]
    if missing:
        raise AdapterError(f"Fit request is missing: {', '.join(missing)}")
    paths = {key: Path(request[key]).resolve() for key in required[:4]}
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        raise AdapterError(f"Required adapter inputs are missing: {absent}")
    train = pd.read_parquet(paths["train_manifest"])
    validation = pd.read_parquet(paths["validation_manifest"])
    pairs = pd.read_parquet(paths["scanner_fit_pairs"])
    policy = json.loads(paths["reference_policy"].read_text(encoding="utf-8"))
    for column in ("stain_id", "scanner_id", "aligned_group_id", "image_path"):
        if column not in train:
            raise AdapterError(f"Training manifest is missing column {column}")
    if request["forbidden_target_ids_sha256"] != policy.get(
        "forbidden_target_ids_sha256"
    ):
        raise AdapterError(
            "Fit request and reference policy forbidden-ID hashes differ"
        )
    forbidden = set(map(str, policy.get("forbidden_target_sample_ids", [])))
    observed_forbidden_hash = hashlib.sha256(
        "\n".join(sorted(forbidden)).encode()
    ).hexdigest()
    if observed_forbidden_hash != request["forbidden_target_ids_sha256"]:
        raise AdapterError("Forbidden target IDs do not match their declared hash")
    leaked_train = set(_sample_ids(train)) & forbidden
    leaked_validation = set(_sample_ids(validation)) & forbidden
    if leaked_train or leaked_validation:
        raise AdapterError(
            "Held-out target leakage detected in adapter fitting: "
            f"train={len(leaked_train)}, validation={len(leaked_validation)}"
        )
    image_missing = [
        str(path)
        for frame in (train, validation)
        for path in frame.image_path
        if not Path(path).is_file()
    ]
    if image_missing:
        raise AdapterError(
            f"Fit manifest contains missing images; first: {image_missing[0]}"
        )
    pair_columns = {
        "source_sample_id",
        "target_sample_id",
        "source_path",
        "target_path",
    }
    if not pairs.empty and not pair_columns <= set(pairs):
        raise AdapterError(
            f"Scanner fit pairs are missing: {sorted(pair_columns - set(pairs))}"
        )
    if pair_columns <= set(pairs):
        pair_ids = set(pairs.source_sample_id.astype(str)) | set(
            pairs.target_sample_id.astype(str)
        )
        if pair_ids & forbidden:
            raise AdapterError("Held-out target leaked into scanner fit pairs")
        declared_pair_hash = policy.get("scanner_pair_sample_ids_sha256")
        observed_pair_hash = hashlib.sha256(
            "\n".join(sorted(pair_ids)).encode()
        ).hexdigest()
        if declared_pair_hash and observed_pair_hash != declared_pair_hash:
            raise AdapterError("Scanner fit pairs do not match their declared ID hash")
        missing_pair_images = [
            str(path)
            for column in ("source_path", "target_path")
            for path in pairs[column]
            if not Path(path).is_file()
        ]
        if missing_pair_images:
            raise AdapterError(
                "Scanner fit pairs contain missing images; first: "
                f"{missing_pair_images[0]}"
            )
    _, settings = _read_settings(method, bool(request.get("fast_dev_run")))
    source_root, commit = _source_commit(method)
    checkpoint_dir = Path(request["checkpoint_dir"]).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {key: _sha256(path) for key, path in paths.items()}
    adapter_code_sha256 = _sha256(Path(__file__))
    protocol_revision = TRAINING_PROTOCOL_REVISIONS[method]
    fingerprint_payload = {
        "schema_version": 2,
        "method": method,
        "seed": int(request.get("seed", 42)),
        "image_size": int(request.get("image_size", 256)),
        "fast_dev_run": bool(request.get("fast_dev_run")),
        "settings": settings,
        "source_commit": commit,
        "input_hashes": input_hashes,
        "training_protocol_revision": protocol_revision,
    }
    fit_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    previous_state_path = checkpoint_dir / "adapter_state.json"
    if previous_state_path.is_file():
        previous = json.loads(previous_state_path.read_text(encoding="utf-8"))
        compatible = _previous_fit_is_compatible(
            previous,
            method=method,
            protocol_revision=protocol_revision,
            source_commit=commit,
            seed=int(request.get("seed", 42)),
            image_size=int(request.get("image_size", 256)),
            fast_dev_run=bool(request.get("fast_dev_run")),
            settings=settings,
            input_hashes=input_hashes,
        )
        if compatible and previous.get("fit_fingerprint"):
            # Migrate checkpoints made before protocol revisions replaced the
            # overly broad whole-file code hash in the fit fingerprint.
            fit_fingerprint = str(previous["fit_fingerprint"])
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "scanner_pairs": pairs.reset_index(drop=True),
        "policy": policy,
        "paths": paths,
        "settings": settings,
        "source_root": source_root,
        "source_commit": commit,
        "input_hashes": input_hashes,
        "adapter_code_sha256": adapter_code_sha256,
        "training_protocol_revision": protocol_revision,
        "fit_fingerprint": fit_fingerprint,
        "checkpoint_dir": checkpoint_dir,
        "seed": int(request.get("seed", 42)),
        "image_size": int(request.get("image_size", 256)),
        "fast_dev": bool(request.get("fast_dev_run")),
    }


def describe_plan(method: str, request: dict) -> dict:
    context = _load_fit_request(method, request)
    train = context["train"]
    domains = (
        sorted(train.scanner_id.astype(str).unique())
        if method == "pix2pix"
        else sorted(train.stain_id.astype(str).unique())
    )
    return {
        "method": method,
        "protocol": context["settings"]["protocol"],
        "source_root": str(context["source_root"]),
        "source_commit": context["source_commit"],
        "training_rows": len(train),
        "validation_rows": len(context["validation"]),
        "domains": domains,
        "domain_models": 1 if method == "histaugan" else len(domains),
        "checkpoint_dir": str(context["checkpoint_dir"]),
        "fast_dev_run": context["fast_dev"],
        "training_protocol_revision": context["training_protocol_revision"],
        "settings": context["settings"],
    }


def _require_device(settings: dict) -> torch.device:
    if settings.get("require_cuda", True) and not torch.cuda.is_available():
        raise AdapterError(
            "CUDA is required for external neural baseline training, but it is not visible"
        )
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _limit(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if not limit or len(frame) <= limit:
        return frame.reset_index(drop=True)
    ids = _sample_ids(frame)
    keys = ids.map(lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    return (
        frame.assign(_selection_key=keys)
        .sort_values("_selection_key", kind="stable")
        .head(limit)
        .drop(columns="_selection_key")
        .reset_index(drop=True)
    )


def _rgb_tensor(path: str | Path, size: int, *, unit: bool = False) -> torch.Tensor:
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return tensor if unit else tensor.mul(2).sub(1)


def _gray3(tensor: torch.Tensor) -> torch.Tensor:
    weights = tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (tensor * weights).sum(1, keepdim=True).repeat(1, 3, 1, 1)


def _batch_paths(
    paths: list[str], batch_size: int, rng: np.random.Generator
) -> list[str]:
    if not paths:
        raise AdapterError("Cannot sample an empty image pool")
    positions = rng.integers(0, len(paths), size=batch_size)
    return [paths[int(position)] for position in positions]


def _load_batch(
    paths: list[str], size: int, device: torch.device, *, unit: bool = False
) -> torch.Tensor:
    return torch.stack([_rgb_tensor(path, size, unit=unit) for path in paths]).to(
        device, non_blocking=True
    )


def _load_random_crop_batch(
    paths: list[str],
    resize_size: int,
    crop_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    if crop_size > resize_size:
        raise AdapterError("Training crop cannot exceed the resized image size")
    tensors = []
    for path in paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize(
                (resize_size, resize_size), Image.Resampling.BICUBIC
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        top = int(rng.integers(0, resize_size - crop_size + 1))
        left = int(rng.integers(0, resize_size - crop_size + 1))
        array = array[top : top + crop_size, left : left + crop_size]
        tensors.append(torch.from_numpy(array.transpose(2, 0, 1).copy()).mul(2).sub(1))
    return torch.stack(tensors).to(device, non_blocking=True)


@contextlib.contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"Cannot import official source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stainnet_model(source_root: Path, settings: dict) -> nn.Module:
    module = _module("factorstain_upstream_stainnet", source_root / "models.py")
    return module.StainNet(
        3,
        3,
        int(settings.get("layers", 3)),
        int(settings.get("channels", 32)),
    )


def _cycle_networks(method: str, source_root: Path):
    name = f"factorstain_upstream_{method}_networks"
    return _module(name, source_root / "models" / "networks.py")


class _Dot(SimpleNamespace):
    pass


def _cagan_options(settings: dict, batch_size: int) -> _Dot:
    return _Dot(
        run=_Dot(
            opt_run={
                "gpu_ids": [],
                "batchSize": batch_size,
                "lr_G": settings["learning_rate_g"],
                "lr_D": settings["learning_rate_d"],
                "lambda_content": settings["lambda_content"],
                "lambda_l1": settings["lambda_l1"],
            }
        ),
        model=_Dot(
            opt_G={
                "input_nc": 3,
                "output_nc": 3,
                "ngf": 64,
                "which_model_netG": "twodecoder_unet",
                "norm": "batch",
                "no_dropout": True,
                "init_type": "normal",
            },
            opt_D={
                "input_nc": 6,
                "output_nc": 3,
                "ndf": 64,
                "which_model_netD": "n_layers",
                "n_layers": 3,
                "norm": "batch",
                "no_dropout": True,
                "init_type": "normal",
                "use_sigmoid": False,
            },
        ),
    )


def _cagan_models(source_root: Path, settings: dict, batch_size: int):
    module = _module(
        "factorstain_upstream_cagan_network", source_root / "network" / "network.py"
    )
    opts = _cagan_options(settings, batch_size)
    generator = module.TwoDecoderUnetGenerator(opts)
    discriminator = module.NLayerDiscriminator(opts)
    module.init_weights(generator, "normal")
    module.init_weights(discriminator, "normal")
    return generator, discriminator


def _histaugan_model(source_root: Path, settings: dict, domains: int):
    network_module = _module(
        "factorstain_upstream_histaugan_networks",
        source_root / "histaugan" / "networks.py",
    )
    previous = sys.modules.get("networks")
    sys.modules["networks"] = network_module
    try:
        model_module = _module(
            "factorstain_upstream_histaugan_model",
            source_root / "histaugan" / "model.py",
        )
    finally:
        if previous is None:
            sys.modules.pop("networks", None)
        else:
            sys.modules["networks"] = previous
    options = _Dot(
        concat=1,
        input_dim=3,
        dis_norm="None",
        dis_spectral_norm=False,
        num_domains=domains,
        crop_size=int(settings["crop_size"]),
        lambda_rec=settings["lambda_rec"],
        lambda_cls=settings["lambda_cls"],
        lambda_cls_G=settings["lambda_cls_G"],
        lr_policy="step",
        n_ep=settings["epochs"],
        n_ep_decay=max(1, settings["epochs"] // 2),
    )
    return model_module.MD_multi(options), options


def _sastaindiff_components(settings: dict, size: int, *, inference: bool):
    root = PROJECT_ROOT / "third_party" / "SAStainDiff"
    with _temporary_sys_path(root):
        from guided_diffusion_.guided_diffusion.script_util import (  # type: ignore[import-not-found]
            create_model_and_diffusion,
        )

    kwargs = {
        "image_size": size,
        "class_cond": False,
        "learn_sigma": False,
        "num_channels": int(settings["model_channels"]),
        "num_res_blocks": int(settings["num_res_blocks"]),
        "channel_mult": "",
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "attention_resolutions": str(settings["attention_resolutions"]),
        "dropout": 0.0,
        "diffusion_steps": int(settings["diffusion_steps"]),
        "noise_schedule": "linear",
        "timestep_respacing": str(settings["inference_respacing"]) if inference else "",
        "timestep_step": int(settings["inference_timestep"])
        if inference
        else int(settings["diffusion_steps"]),
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_fp16": False,
        "use_new_attention_order": False,
        "is_train": not inference,
    }
    return create_model_and_diffusion(**kwargs)


def _steps(pool_size: int, batch_size: int, configured: int) -> int:
    return int(configured) if configured else max(1, math.ceil(pool_size / batch_size))


def _paired_stain_rows(
    frame: pd.DataFrame, target_stain: str, target_scanner: str
) -> pd.DataFrame:
    target = frame[
        frame.stain_id.astype(str).eq(target_stain)
        & frame.scanner_id.astype(str).eq(target_scanner)
    ].copy()
    source = frame[~frame.stain_id.astype(str).eq(target_stain)].copy()
    keys = ["aligned_group_id", "scanner_id"]
    pairs = source.merge(target, on=keys, suffixes=("_source", "_target"))
    if pairs.empty:
        return pairs
    return pairs.sort_values(
        ["aligned_group_id", "scanner_id", "stain_id_source"], kind="stable"
    ).reset_index(drop=True)


def _checkpoint_path(context: dict, domain: str) -> Path:
    directory = context["checkpoint_dir"] / "official_adapter"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (f"{_safe_name(domain)}_{context['fit_fingerprint'][:12]}.pt")


def _checkpoint_metadata(context: dict, domain: str) -> dict:
    return {
        "target_domain": domain,
        "settings": context["settings"],
        "fit_fingerprint": context["fit_fingerprint"],
        "training_protocol_revision": context["training_protocol_revision"],
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_stainnet(context: dict) -> dict[str, dict]:
    settings, device = context["settings"], _require_device(context["settings"])
    train, validation = context["train"], context["validation"]
    domains: dict[str, dict] = {}
    stains = sorted(train.stain_id.astype(str).unique())
    for number, stain in enumerate(stains):
        checkpoint = _checkpoint_path(context, stain)
        reference_scanner = str(
            context["policy"]["stain_prototypes"][stain]["scanner_id"]
        )
        train_pairs = _paired_stain_rows(train, stain, reference_scanner)
        val_pairs = _paired_stain_rows(validation, stain, reference_scanner)
        if train_pairs.empty:
            raise AdapterError(
                f"StainNet has no aligned-group/same-scanner training pairs for stain {stain}"
            )
        train_pairs = _limit(
            train_pairs.rename(columns={"sample_id_source": "sample_id"}),
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number,
        )
        val_pairs = (
            _limit(
                val_pairs.rename(columns={"sample_id_source": "sample_id"}),
                int(settings["max_validation_images_per_domain"]),
                context["seed"] + number + 1000,
            )
            if not val_pairs.empty
            else val_pairs
        )
        if not checkpoint.exists():
            _seed_everything(context["seed"] + number, settings["deterministic"])
            model = _stainnet_model(context["source_root"], settings).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=float(settings["learning_rate"])
            )
            epochs = int(settings["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            rng = np.random.default_rng(context["seed"] + number)
            best_loss = float("inf")
            best_state = None
            count = _steps(
                len(train_pairs),
                int(settings["batch_size"]),
                int(settings["steps_per_epoch"]),
            )
            for epoch in range(epochs):
                model.train()
                for _ in range(count):
                    positions = rng.integers(
                        0, len(train_pairs), size=int(settings["batch_size"])
                    )
                    rows = train_pairs.iloc[positions]
                    source = _load_batch(
                        rows.image_path_source.astype(str).tolist(),
                        context["image_size"],
                        device,
                    )
                    target = _load_batch(
                        rows.image_path_target.astype(str).tolist(),
                        context["image_size"],
                        device,
                    )
                    loss = nn.functional.l1_loss(model(source), target)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                if val_pairs.empty or (epoch + 1) % 5:
                    continue
                model.eval()
                losses = []
                with torch.no_grad():
                    for start in range(0, len(val_pairs), int(settings["batch_size"])):
                        rows = val_pairs.iloc[
                            start : start + int(settings["batch_size"])
                        ]
                        source = _load_batch(
                            rows.image_path_source.astype(str).tolist(),
                            context["image_size"],
                            device,
                        )
                        target = _load_batch(
                            rows.image_path_target.astype(str).tolist(),
                            context["image_size"],
                            device,
                        )
                        losses.append(
                            float(nn.functional.l1_loss(model(source), target))
                        )
                observed = float(np.mean(losses))
                if observed < best_loss:
                    best_loss = observed
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
            state = best_state or {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            _save_checkpoint(
                checkpoint,
                {
                    "model": state,
                    **_checkpoint_metadata(context, stain),
                    "validation_l1": None if best_loss == float("inf") else best_loss,
                },
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        domains[stain] = {
            "checkpoint": str(checkpoint.relative_to(context["checkpoint_dir"])),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(val_pairs),
            "reference_scanner": reference_scanner,
        }
    return domains


class _ReplayPool:
    def __init__(self, size: int, rng: random.Random):
        self.size = size
        self.rng = rng
        self.images: deque[torch.Tensor] = deque()

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        if self.size <= 0:
            return batch.detach()
        output = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(image.cpu())
                output.append(image)
            elif self.rng.random() > 0.5:
                position = self.rng.randrange(len(self.images))
                old = self.images[position].to(image.device)
                self.images[position] = image.cpu()
                output.append(old)
            else:
                output.append(image)
        return torch.cat(output)


def _gan_loss(prediction: torch.Tensor, real: bool) -> torch.Tensor:
    target = torch.ones_like(prediction) if real else torch.zeros_like(prediction)
    return nn.functional.mse_loss(prediction, target)


def _set_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)


def _cagan_learning_rate(base: float, epoch: int, epochs: int, decay: float) -> float:
    minimum = base * decay**3
    return minimum + (base - minimum) * (1.0 + math.cos(math.pi * epoch / epochs)) / 2.0


def _make_cycle_models(context: dict):
    method, settings = context["method"], context["settings"]
    module = _cycle_networks(method, context["source_root"])
    if method == "staingan":
        define_g = lambda: module.define_G(
            3,
            3,
            64,
            settings["generator"],
            norm="instance",
            use_dropout=False,
            init_type="normal",
            gpu_ids=[],
        )
        define_d = lambda: module.define_D(
            3,
            64,
            "basic",
            norm="instance",
            use_sigmoid=False,
            init_type="normal",
            gpu_ids=[],
        )
    else:
        define_g = lambda: module.define_G(
            3, 3, 64, settings["generator"], "instance", False, "normal", 0.02
        )
        define_d = lambda: module.define_D(
            3, 64, "basic", 3, "instance", "normal", 0.02
        )
    models = (define_g(), define_g(), define_d(), define_d())
    if method != "staingan":
        for model in models:
            module.init_weights(model, "normal", 0.02)
    return models


def _train_cycle_bank(context: dict) -> dict[str, dict]:
    settings, device = context["settings"], _require_device(context["settings"])
    train = context["train"]
    stains = sorted(train.stain_id.astype(str).unique())
    domains = {}
    for number, stain in enumerate(stains):
        checkpoint = _checkpoint_path(context, stain)
        reference_scanner = str(
            context["policy"]["stain_prototypes"][stain]["scanner_id"]
        )
        target = _limit(
            train[
                train.stain_id.astype(str).eq(stain)
                & train.scanner_id.astype(str).eq(reference_scanner)
            ],
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number,
        )
        source = _limit(
            train[~train.stain_id.astype(str).eq(stain)],
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number + 1000,
        )
        if target.empty or source.empty:
            raise AdapterError(
                f"Insufficient unpaired training pools for stain {stain}"
            )
        if not checkpoint.exists():
            _seed_everything(context["seed"] + number, settings["deterministic"])
            g_ab, g_ba, d_a, d_b = [
                model.to(device) for model in _make_cycle_models(context)
            ]
            optimizer_g = torch.optim.Adam(
                list(g_ab.parameters()) + list(g_ba.parameters()),
                lr=float(settings["learning_rate"]),
                betas=(float(settings["beta1"]), 0.999),
            )
            optimizer_d = torch.optim.Adam(
                list(d_a.parameters()) + list(d_b.parameters()),
                lr=float(settings["learning_rate"]),
                betas=(float(settings["beta1"]), 0.999),
            )
            rng = np.random.default_rng(context["seed"] + number)
            replay_rng = random.Random(context["seed"] + number)
            pool_a = _ReplayPool(int(settings["replay_size"]), replay_rng)
            pool_b = _ReplayPool(int(settings["replay_size"]), replay_rng)
            constant = int(settings["epochs_constant"])
            decay = int(settings["epochs_decay"])
            epochs = constant + decay
            batch_size = int(settings["batch_size"])
            count = _steps(
                max(len(source), len(target)),
                batch_size,
                int(settings["steps_per_epoch"]),
            )
            source_paths = source.image_path.astype(str).tolist()
            target_paths = target.image_path.astype(str).tolist()
            for epoch in range(epochs):
                factor = (
                    1.0
                    if epoch < constant
                    else 1.0 - (epoch - constant + 1) / max(decay, 1)
                )
                for optimizer in (optimizer_g, optimizer_d):
                    for group in optimizer.param_groups:
                        group["lr"] = float(settings["learning_rate"]) * max(
                            factor, 0.0
                        )
                for _ in range(count):
                    real_a = _load_batch(
                        _batch_paths(source_paths, batch_size, rng),
                        context["image_size"],
                        device,
                    )
                    real_b = _load_batch(
                        _batch_paths(target_paths, batch_size, rng),
                        context["image_size"],
                        device,
                    )
                    fake_b = g_ab(real_a)
                    fake_a = g_ba(real_b)
                    rec_a = g_ba(fake_b)
                    rec_b = g_ab(fake_a)
                    id_a = g_ba(real_a)
                    id_b = g_ab(real_b)
                    loss_g = (
                        _gan_loss(d_b(fake_b), True)
                        + _gan_loss(d_a(fake_a), True)
                        + float(settings["lambda_cycle"])
                        * (
                            nn.functional.l1_loss(rec_a, real_a)
                            + nn.functional.l1_loss(rec_b, real_b)
                        )
                        + float(settings["lambda_identity"])
                        * (
                            nn.functional.l1_loss(id_a, real_a)
                            + nn.functional.l1_loss(id_b, real_b)
                        )
                    )
                    optimizer_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    optimizer_g.step()
                    pooled_a, pooled_b = pool_a.query(fake_a), pool_b.query(fake_b)
                    loss_d = 0.5 * (
                        _gan_loss(d_a(real_a), True)
                        + _gan_loss(d_a(pooled_a), False)
                        + _gan_loss(d_b(real_b), True)
                        + _gan_loss(d_b(pooled_b), False)
                    )
                    optimizer_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    optimizer_d.step()
            _save_checkpoint(
                checkpoint,
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in g_ab.state_dict().items()
                    },
                    **_checkpoint_metadata(context, stain),
                },
            )
            del g_ab, g_ba, d_a, d_b
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        domains[stain] = {
            "checkpoint": str(checkpoint.relative_to(context["checkpoint_dir"])),
            "source_rows": len(source),
            "target_rows": len(target),
            "reference_scanner": reference_scanner,
        }
    return domains


def _make_pix2pix_models(context: dict):
    module = _cycle_networks("pix2pix", context["source_root"])
    settings = context["settings"]
    generator = module.define_G(
        3, 3, 64, settings["generator"], "batch", True, "normal", 0.02
    )
    discriminator = module.define_D(6, 64, "basic", 3, "batch", "normal", 0.02)
    module.init_weights(generator, "normal", 0.02)
    module.init_weights(discriminator, "normal", 0.02)
    return generator, discriminator


def _train_pix2pix(context: dict) -> dict[str, dict]:
    settings, device = context["settings"], _require_device(context["settings"])
    pairs = context["scanner_pairs"].copy()
    required = {
        "source_scanner",
        "target_scanner",
        "source_path",
        "target_path",
        "source_sample_id",
        "target_sample_id",
    }
    if not required <= set(pairs):
        raise AdapterError(
            f"Scanner fit pairs are missing: {sorted(required - set(pairs))}"
        )
    forbidden = set(map(str, context["policy"]["forbidden_target_sample_ids"]))
    used = set(pairs.source_sample_id.astype(str)) | set(
        pairs.target_sample_id.astype(str)
    )
    if used & forbidden:
        raise AdapterError("Held-out target leaked into Pix2Pix scanner pairs")
    scanners = sorted(context["train"].scanner_id.astype(str).unique())
    domains = {}
    for number, scanner in enumerate(scanners):
        checkpoint = _checkpoint_path(context, scanner)
        frame = pairs[pairs.target_scanner.astype(str).eq(scanner)].reset_index(
            drop=True
        )
        if frame.empty:
            raise AdapterError(
                f"Pix2Pix has no registered training pairs for scanner {scanner}"
            )
        frame = _limit(
            frame.rename(columns={"source_sample_id": "sample_id"}),
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number,
        )
        if not checkpoint.exists():
            _seed_everything(context["seed"] + number, settings["deterministic"])
            generator, discriminator = [
                model.to(device) for model in _make_pix2pix_models(context)
            ]
            optimizer_g = torch.optim.Adam(
                generator.parameters(),
                lr=float(settings["learning_rate"]),
                betas=(float(settings["beta1"]), 0.999),
            )
            optimizer_d = torch.optim.Adam(
                discriminator.parameters(),
                lr=float(settings["learning_rate"]),
                betas=(float(settings["beta1"]), 0.999),
            )
            rng = np.random.default_rng(context["seed"] + number)
            constant = int(settings["epochs_constant"])
            decay = int(settings["epochs_decay"])
            epochs = constant + decay
            batch_size = int(settings["batch_size"])
            count = _steps(len(frame), batch_size, int(settings["steps_per_epoch"]))
            for epoch in range(epochs):
                factor = (
                    1.0
                    if epoch < constant
                    else 1.0 - (epoch - constant + 1) / max(decay, 1)
                )
                for optimizer in (optimizer_g, optimizer_d):
                    for group in optimizer.param_groups:
                        group["lr"] = float(settings["learning_rate"]) * max(
                            factor, 0.0
                        )
                for _ in range(count):
                    positions = rng.integers(0, len(frame), size=batch_size)
                    rows = frame.iloc[positions]
                    source = _load_batch(
                        rows.source_path.astype(str).tolist(),
                        context["image_size"],
                        device,
                    )
                    target = _load_batch(
                        rows.target_path.astype(str).tolist(),
                        context["image_size"],
                        device,
                    )
                    generated = generator(source)
                    fake_prediction = discriminator(
                        torch.cat([source, generated], dim=1)
                    )
                    loss_g = nn.functional.binary_cross_entropy_with_logits(
                        fake_prediction, torch.ones_like(fake_prediction)
                    ) + float(settings["lambda_l1"]) * nn.functional.l1_loss(
                        generated, target
                    )
                    optimizer_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    optimizer_g.step()
                    real_prediction = discriminator(torch.cat([source, target], dim=1))
                    fake_prediction = discriminator(
                        torch.cat([source, generated.detach()], dim=1)
                    )
                    loss_d = 0.5 * (
                        nn.functional.binary_cross_entropy_with_logits(
                            real_prediction, torch.ones_like(real_prediction)
                        )
                        + nn.functional.binary_cross_entropy_with_logits(
                            fake_prediction, torch.zeros_like(fake_prediction)
                        )
                    )
                    optimizer_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    optimizer_d.step()
            _save_checkpoint(
                checkpoint,
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in generator.state_dict().items()
                    },
                    **_checkpoint_metadata(context, scanner),
                },
            )
            del generator, discriminator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        domains[scanner] = {
            "checkpoint": str(checkpoint.relative_to(context["checkpoint_dir"])),
            "registered_pairs": len(frame),
        }
    return domains


def _train_cagan(context: dict) -> dict[str, dict]:
    settings, device = context["settings"], _require_device(context["settings"])
    train = context["train"]
    stains = sorted(train.stain_id.astype(str).unique())
    domains = {}
    for number, stain in enumerate(stains):
        checkpoint = _checkpoint_path(context, stain)
        reference_scanner = str(
            context["policy"]["stain_prototypes"][stain]["scanner_id"]
        )
        target = _limit(
            train[
                train.stain_id.astype(str).eq(stain)
                & train.scanner_id.astype(str).eq(reference_scanner)
            ],
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number,
        )
        source = _limit(
            train[~train.stain_id.astype(str).eq(stain)],
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number + 1000,
        )
        if target.empty or source.empty:
            raise AdapterError(
                f"CAGAN lacks source/target training rows for stain {stain}"
            )
        if not checkpoint.exists():
            _seed_everything(context["seed"] + number, settings["deterministic"])
            batch_size = int(settings["batch_size"])
            generator, discriminator = [
                model.to(device)
                for model in _cagan_models(context["source_root"], settings, batch_size)
            ]
            loss_module = _module(
                "factorstain_upstream_cagan_loss",
                context["source_root"] / "utils" / "loss.py",
            )
            histogram = loss_module.RGBuvHistBlock(
                h=64,
                insz=150,
                resizing="interpolation",
                intensity_scale=True,
                method="inverse-quadratic",
                device=str(device),
            ).to(device)
            try:
                from torchvision.models import VGG16_Weights, vgg16

                perceptual = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
            except Exception as exc:
                raise AdapterError(
                    "CAGAN requires the official ImageNet VGG16 perceptual backbone. "
                    "Allow torchvision to download it once or pre-populate the Torch cache."
                ) from exc
            perceptual.eval()
            perceptual.requires_grad_(False)
            optimizer_g = torch.optim.Adam(
                generator.parameters(),
                lr=float(settings["learning_rate_g"]),
                betas=(0.9, 0.999),
            )
            optimizer_d = torch.optim.Adam(
                discriminator.parameters(),
                lr=float(settings["learning_rate_d"]),
                betas=(0.9, 0.999),
            )
            rng = np.random.default_rng(context["seed"] + number)
            source_paths = source.image_path.astype(str).tolist()
            target_paths = target.image_path.astype(str).tolist()
            count = _steps(
                max(len(source), len(target)),
                batch_size,
                int(settings["steps_per_epoch"]),
            )
            epochs = int(settings["epochs"])
            for epoch in range(epochs):
                for _ in range(count):
                    source_rgb = _load_batch(
                        _batch_paths(source_paths, batch_size, rng),
                        context["image_size"],
                        device,
                        unit=True,
                    )
                    target_rgb = _load_batch(
                        _batch_paths(target_paths, batch_size, rng),
                        context["image_size"],
                        device,
                        unit=True,
                    )
                    source_gray, target_gray = _gray3(source_rgb), _gray3(target_rgb)
                    joined = torch.cat([target_gray, source_gray], dim=0)
                    output_1, output_2 = generator(joined)
                    target_1, source_1 = output_1.chunk(2)
                    target_2, source_2 = output_2.chunk(2)
                    real_pair = torch.cat([target_gray, target_rgb], dim=1)
                    fake_1 = torch.cat([target_gray, target_1], dim=1)
                    fake_2 = torch.cat([target_gray, target_2], dim=1)
                    _set_trainable(discriminator, True)
                    pred_real = discriminator(real_pair)
                    pred_fake_1 = discriminator(fake_1.detach())
                    pred_fake_2 = discriminator(fake_2.detach())
                    loss_d = (
                        nn.functional.binary_cross_entropy_with_logits(
                            pred_real, torch.full_like(pred_real, 0.8)
                        )
                        + nn.functional.binary_cross_entropy_with_logits(
                            pred_fake_1, torch.zeros_like(pred_fake_1)
                        )
                        + nn.functional.binary_cross_entropy_with_logits(
                            pred_fake_2, torch.zeros_like(pred_fake_2)
                        )
                    )
                    optimizer_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    optimizer_d.step()
                    _set_trainable(discriminator, False)
                    pred_1 = discriminator(torch.cat([joined, output_1], dim=1))
                    pred_2 = discriminator(torch.cat([joined, output_2], dim=1))
                    loss_adv = nn.functional.binary_cross_entropy_with_logits(
                        pred_1, torch.ones_like(pred_1)
                    ) + nn.functional.binary_cross_entropy_with_logits(
                        pred_2, torch.ones_like(pred_2)
                    )
                    loss_target = (
                        nn.functional.l1_loss(target_1, target_rgb)
                        + nn.functional.l1_loss(target_2, target_rgb)
                    ) * float(settings["lambda_l1"])
                    loss_consistency = nn.functional.l1_loss(
                        source_1, source_2
                    ) * float(settings["lambda_l1"])
                    loss_content = (
                        nn.functional.mse_loss(
                            perceptual(target_1), perceptual(target_rgb)
                        )
                        + nn.functional.mse_loss(
                            perceptual(target_2), perceptual(target_rgb)
                        )
                    ) * float(settings["lambda_content"])
                    target_hist = histogram(target_rgb)
                    source_hist_1 = histogram(source_1)
                    source_hist_2 = histogram(source_2)
                    loss_histogram = (
                        torch.sqrt(
                            torch.sum(
                                (
                                    torch.sqrt(target_hist + 1e-6)
                                    - torch.sqrt(source_hist_1 + 1e-6)
                                )
                                ** 2
                            )
                        )
                        + torch.sqrt(
                            torch.sum(
                                (
                                    torch.sqrt(target_hist + 1e-6)
                                    - torch.sqrt(source_hist_2 + 1e-6)
                                )
                                ** 2
                            )
                        )
                    ) / (math.sqrt(2.0) * batch_size)
                    loss_g = (
                        loss_adv
                        + loss_target
                        + loss_consistency
                        + loss_content
                        + loss_histogram
                    )
                    optimizer_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    optimizer_g.step()
                for optimizer, key in (
                    (optimizer_g, "learning_rate_g"),
                    (optimizer_d, "learning_rate_d"),
                ):
                    rate = _cagan_learning_rate(
                        float(settings[key]),
                        epoch,
                        epochs,
                        float(settings["lr_decay_rate"]),
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = rate
            _save_checkpoint(
                checkpoint,
                {
                    "model": {
                        key: value.detach().cpu()
                        for key, value in generator.state_dict().items()
                    },
                    **_checkpoint_metadata(context, stain),
                },
            )
            del generator, discriminator, perceptual, histogram
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        domains[stain] = {
            "checkpoint": str(checkpoint.relative_to(context["checkpoint_dir"])),
            "source_rows": len(source),
            "target_rows": len(target),
            "reference_scanner": reference_scanner,
        }
    return domains


def _train_histaugan(context: dict) -> dict[str, dict]:
    settings, device = context["settings"], _require_device(context["settings"])
    if device.type != "cuda":
        raise AdapterError("The pinned HistAuGAN implementation requires CUDA")
    train = context["train"]
    stains = sorted(train.stain_id.astype(str).unique())
    checkpoint = (
        context["checkpoint_dir"]
        / "official_adapter"
        / f"multidomain_{context['fit_fingerprint'][:12]}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pools = {
        stain: _limit(
            train[
                train.stain_id.astype(str).eq(stain)
                & train.scanner_id.astype(str).eq(
                    str(context["policy"]["stain_prototypes"][stain]["scanner_id"])
                )
            ],
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number,
        )
        .image_path.astype(str)
        .tolist()
        for number, stain in enumerate(stains)
    }
    if any(not paths for paths in pools.values()):
        raise AdapterError("HistAuGAN requires at least one training image per stain")
    if not checkpoint.exists():
        _seed_everything(context["seed"], settings["deterministic"])
        model, options = _histaugan_model(context["source_root"], settings, len(stains))
        model.initialize()
        model.setgpu(0)
        # The upstream training script uses -1 for a fresh optimizer. Passing 0 is
        # interpreted as a resume by modern PyTorch and requires an initial_lr key.
        model.set_scheduler(options, last_ep=-1)
        rng = np.random.default_rng(context["seed"])
        batch_size = int(settings["batch_size"])
        if batch_size % 2:
            raise AdapterError("HistAuGAN batch_size must be even")
        count = _steps(
            max(map(len, pools.values())),
            batch_size,
            int(settings["steps_per_epoch"]),
        )
        update = 0
        for _epoch in range(int(settings["epochs"])):
            for _ in range(count):
                indices = rng.integers(0, len(stains), size=batch_size)
                paths = [
                    pools[stains[int(index)]][
                        int(rng.integers(0, len(pools[stains[int(index)]])))
                    ]
                    for index in indices
                ]
                images = _load_random_crop_batch(
                    paths,
                    context["image_size"],
                    int(settings["crop_size"]),
                    device,
                    rng,
                )
                labels = torch.zeros(batch_size, len(stains), device=device)
                labels[
                    torch.arange(batch_size, device=device),
                    torch.as_tensor(indices, device=device),
                ] = 1
                update += 1
                if update % int(settings["d_iter"]):
                    model.update_D_content(images, labels)
                else:
                    model.update_D(images, labels)
                    model.update_EG()
            model.update_lr()
        _save_checkpoint(
            checkpoint,
            {
                "enc_c": {
                    key: value.detach().cpu()
                    for key, value in model.enc_c.state_dict().items()
                },
                "enc_a": {
                    key: value.detach().cpu()
                    for key, value in model.enc_a.state_dict().items()
                },
                "gen": {
                    key: value.detach().cpu()
                    for key, value in model.gen.state_dict().items()
                },
                "stains": stains,
                **_checkpoint_metadata(context, "MULTIDOMAIN"),
            },
        )
        del model
        torch.cuda.empty_cache()
    return {
        stain: {
            "checkpoint": str(checkpoint.relative_to(context["checkpoint_dir"])),
            "training_rows": len(pools[stain]),
            "domain_index": position,
            "prototype_path": context["policy"]["stain_prototypes"][stain][
                "image_path"
            ],
            "reference_scanner": context["policy"]["stain_prototypes"][stain][
                "scanner_id"
            ],
        }
        for position, stain in enumerate(stains)
    }


def _stain_matrix(image: np.ndarray, *, io: float = 240.0) -> np.ndarray:
    pixels = image.reshape(-1, 3).astype(np.float64)
    optical_density = -np.log((pixels + 1.0) / io)
    filtered = optical_density[~np.any(optical_density < 0.15, axis=1)]
    if len(filtered) < 10:
        raise ValueError("insufficient tissue pixels")
    _, eigenvectors = np.linalg.eigh(np.cov(filtered.T))
    projection = filtered @ eigenvectors[:, 1:3]
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    low, high = np.percentile(angles, (1, 99))
    first = eigenvectors[:, 1:3] @ np.array([np.cos(low), np.sin(low)])
    second = eigenvectors[:, 1:3] @ np.array([np.cos(high), np.sin(high)])
    return (
        np.stack([first, second], axis=1)
        if first[0] > second[0]
        else np.stack([second, first], axis=1)
    )


def _normalize_to_stain(
    image: np.ndarray, reference: np.ndarray, *, io: float = 240.0
) -> np.ndarray:
    height, width = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float64)
    source = _stain_matrix(image, io=io)
    optical_density = -np.log((pixels + 1.0) / io).T
    concentrations = np.linalg.lstsq(source, optical_density, rcond=None)[0]
    maxima = np.maximum(np.percentile(concentrations, 99, axis=1), 1e-6)
    concentrations = concentrations / (maxima / np.array([1.9705, 1.0308]))[:, None]
    normalized = io * np.exp(-(reference @ concentrations))
    return np.clip(normalized.T.reshape(height, width, 3), 0, 255).astype(np.uint8)


def _build_stain_database(context: dict) -> np.ndarray:
    destination = (
        context["checkpoint_dir"]
        / "official_adapter"
        / f"train_stain_database_{context['fit_fingerprint'][:12]}.npz"
    )
    if destination.is_file():
        return np.load(destination)["matrices"]
    limit = int(context["settings"]["stain_database_limit"])
    frame = _limit(context["train"], limit, context["seed"] + 9999)
    matrices = []
    for path in frame.image_path.astype(str):
        try:
            with Image.open(path) as opened:
                image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
            matrices.append(_stain_matrix(image))
        except (OSError, ValueError, np.linalg.LinAlgError):
            continue
    if len(matrices) < 2:
        raise AdapterError("SAStainDiff could not derive a train-only stain database")
    array = np.stack(matrices)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, matrices=array)
    return array


def _augment_stain(
    path: str,
    size: int,
    database: np.ndarray,
    tree: cKDTree,
    settings: dict,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as opened:
        target = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if target.shape[0] >= size and target.shape[1] >= size:
        top = int(rng.integers(0, target.shape[0] - size + 1))
        left = int(rng.integers(0, target.shape[1] - size + 1))
        target = target[top : top + size, left : left + size].copy()
    else:
        target = np.asarray(
            Image.fromarray(target).resize((size, size), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
    if rng.random() < 0.5:
        target = np.flip(target, axis=0).copy()
    if rng.random() < 0.5:
        target = np.flip(target, axis=1).copy()
    if rng.random() < 0.5:
        target = np.rot90(target, int(rng.integers(0, 4))).copy()

    neighbours = min(int(settings["nearest_neighbours"]), len(database))
    radius = float(settings["sigma_perturb"])

    def accepted(matrix: np.ndarray) -> bool:
        distances, positions = tree.query(
            matrix.reshape(6), k=neighbours, distance_upper_bound=radius
        )
        return bool(
            np.all(np.isfinite(np.atleast_1d(distances)))
            and np.all(np.atleast_1d(positions) < len(database))
        )

    if rng.random() < float(settings["color_augmentation_probability"]):
        augmented = target.copy()
        shift = int(settings["shift_value"])
        for _ in range(int(settings["color_attempts"])):
            try:
                cv2 = importlib.import_module("cv2")
                hsv = cv2.cvtColor(target, cv2.COLOR_RGB2HSV).astype(np.int16)
                hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-shift, shift + 1))) % 180
                for channel in (1, 2):
                    hsv[..., channel] = np.clip(
                        hsv[..., channel] + int(rng.integers(-shift, shift + 1)),
                        0,
                        255,
                    )
                augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            except ModuleNotFoundError:
                hsv = np.asarray(Image.fromarray(target).convert("HSV")).astype(
                    np.int16
                )
                hue_shift = round(int(rng.integers(-shift, shift + 1)) * 255 / 180)
                hsv[..., 0] = (hsv[..., 0] + hue_shift) % 256
                for channel in (1, 2):
                    hsv[..., channel] = np.clip(
                        hsv[..., channel] + int(rng.integers(-shift, shift + 1)),
                        0,
                        255,
                    )
                augmented = np.asarray(
                    Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
                )
            try:
                if accepted(_stain_matrix(augmented)):
                    break
            except (ValueError, np.linalg.LinAlgError):
                continue
    else:
        reference = database[int(rng.integers(0, len(database)))].copy()
        for _ in range(int(settings["stain_attempts"])):
            reference = database[int(rng.integers(0, len(database)))].copy()
            reference *= rng.uniform(1 - settings["sigma1"], 1 + settings["sigma1"])
            reference += rng.uniform(-settings["sigma2"], settings["sigma2"])
            if accepted(reference):
                break
        try:
            augmented = _normalize_to_stain(target, reference)
        except (ValueError, np.linalg.LinAlgError):
            augmented = target.copy()
    if settings["gaussian_blur"] and rng.random() < 0.5:
        kernel = int(rng.integers(5, 21))
        kernel += 1 - kernel % 2
        sigma = float(rng.uniform(0.1, 2.0))
        try:
            cv2 = importlib.import_module("cv2")
            augmented = cv2.GaussianBlur(augmented, (kernel, kernel), sigmaX=sigma)
        except ModuleNotFoundError:
            augmented = np.asarray(
                Image.fromarray(augmented).filter(ImageFilter.GaussianBlur(sigma))
            )
    target_tensor = (
        torch.from_numpy(target.transpose(2, 0, 1).copy()).float() / 127.5 - 1
    )
    augmented_tensor = (
        torch.from_numpy(augmented.transpose(2, 0, 1).copy()).float() / 127.5 - 1
    )
    return target_tensor, augmented_tensor


def _ema_update(ema: dict[str, torch.Tensor], model: nn.Module, rate: float) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema[key].mul_(rate).add_(value.detach(), alpha=1.0 - rate)


def _train_sastaindiff(context: dict) -> dict[str, dict]:
    settings, device = context["settings"], _require_device(context["settings"])
    train = context["train"]
    stains = sorted(train.stain_id.astype(str).unique())
    database = _build_stain_database(context)
    stain_tree = cKDTree(database.reshape(len(database), 6))
    domains = {}
    for number, stain in enumerate(stains):
        checkpoint = _checkpoint_path(context, stain)
        reference_scanner = str(
            context["policy"]["stain_prototypes"][stain]["scanner_id"]
        )
        target = _limit(
            train[
                train.stain_id.astype(str).eq(stain)
                & train.scanner_id.astype(str).eq(reference_scanner)
            ],
            int(settings["max_train_images_per_domain"]),
            context["seed"] + number,
        )
        if target.empty:
            raise AdapterError(
                f"SAStainDiff has no target training rows for stain {stain}"
            )
        if not checkpoint.exists():
            _seed_everything(context["seed"] + number, settings["deterministic"])
            model, diffusion = _sastaindiff_components(
                settings, context["image_size"], inference=False
            )
            model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(settings["learning_rate"])
            )
            ema = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            rng = np.random.default_rng(context["seed"] + number)
            paths = target.image_path.astype(str).tolist()
            batch_size = int(settings["batch_size"])
            microbatch = int(settings["microbatch"])
            if microbatch <= 0:
                microbatch = batch_size
            for _step in range(int(settings["train_steps"])):
                selected = _batch_paths(paths, batch_size, rng)
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, batch_size, microbatch):
                    values = [
                        _augment_stain(
                            path,
                            context["image_size"],
                            database,
                            stain_tree,
                            settings,
                            rng,
                        )
                        for path in selected[start : start + microbatch]
                    ]
                    clean = torch.stack([value[0] for value in values]).to(device)
                    augmented = torch.stack([value[1] for value in values]).to(device)
                    timesteps = torch.randint(
                        0, len(diffusion.betas), (len(values),), device=device
                    )
                    losses = diffusion.training_losses(
                        model, clean, augmented, timesteps
                    )
                    (losses["loss"].mean() * len(values) / batch_size).backward()
                optimizer.step()
                _ema_update(ema, model, float(settings["ema_rate"]))
            _save_checkpoint(
                checkpoint,
                {
                    "model": {key: value.detach().cpu() for key, value in ema.items()},
                    **_checkpoint_metadata(context, stain),
                },
            )
            del model, diffusion, ema
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        domains[stain] = {
            "checkpoint": str(checkpoint.relative_to(context["checkpoint_dir"])),
            "target_rows": len(target),
            "stain_database_rows": len(database),
            "reference_scanner": reference_scanner,
        }
    return domains


def fit_adapter(method: str, request: dict) -> None:
    if method not in SUPPORTED_METHODS:
        raise AdapterError(f"Unsupported external method: {method}")
    context = _load_fit_request(method, request)
    context["method"] = method
    _seed_everything(context["seed"], context["settings"]["deterministic"])
    trainers = {
        "stainnet": _train_stainnet,
        "staingan": _train_cycle_bank,
        "cyclegan": _train_cycle_bank,
        "pix2pix": _train_pix2pix,
        "histaugan": _train_histaugan,
        "cagan": _train_cagan,
        "sastaindiff": _train_sastaindiff,
    }
    domains = trainers[method](context)
    state = {
        "schema_version": 2,
        "method": method,
        "protocol": context["settings"]["protocol"],
        "implementation": "FactorStain adapter around pinned official architecture",
        "official_source_root": str(context["source_root"]),
        "official_source_commit": context["source_commit"],
        "adapter_code_sha256": context["adapter_code_sha256"],
        "training_protocol_revision": context["training_protocol_revision"],
        "fit_fingerprint": context["fit_fingerprint"],
        "seed": context["seed"],
        "image_size": context["image_size"],
        "fast_dev_run": context["fast_dev"],
        "training_manifest": str(context["paths"]["train_manifest"]),
        "training_manifest_sha256": context["input_hashes"]["train_manifest"],
        "validation_manifest": str(context["paths"]["validation_manifest"]),
        "validation_manifest_sha256": context["input_hashes"]["validation_manifest"],
        "reference_policy_sha256": context["input_hashes"]["reference_policy"],
        "scanner_fit_pairs_sha256": context["input_hashes"]["scanner_fit_pairs"],
        "forbidden_target_ids_sha256": request["forbidden_target_ids_sha256"],
        "evaluation_manifest_accessed": False,
        "settings": context["settings"],
        "domains": domains,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "adaptation_notes": {
            "target_bank": method not in {"histaugan"},
            "scanner_handling": "FactorStain training-only ScannerLUT appended outside adapter"
            if method != "pix2pix"
            else "native registered scanner-pair training",
            "cagan_external_pretraining": "ImageNet VGG16 perceptual loss"
            if method == "cagan"
            else "none",
            "histaugan_inference_style": "train-only stain prototype attribute mean"
            if method == "histaugan"
            else None,
            "sastaindiff_stain_database": "derived only from frozen training manifest"
            if method == "sastaindiff"
            else None,
        },
    }
    atomic_json_dump(state, context["checkpoint_dir"] / "adapter_state.json")


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class AdapterRuntime:
    def __init__(self, method: str, request: dict):
        self.method = method
        self.checkpoint_dir = Path(request["checkpoint_dir"]).resolve()
        state_path = self.checkpoint_dir / "adapter_state.json"
        if not state_path.is_file():
            raise AdapterError(f"Adapter state is missing: {state_path}; run fit first")
        self.state = json.loads(state_path.read_text(encoding="utf-8"))
        if self.state.get("method") != method:
            raise AdapterError(
                f"Adapter state belongs to {self.state.get('method')}, not {method}"
            )
        observed_revision = int(self.state.get("training_protocol_revision", 1))
        expected_revision = TRAINING_PROTOCOL_REVISIONS[method]
        if observed_revision != expected_revision:
            raise AdapterError(
                "External training protocol changed after fitting; refit the method"
            )
        self.settings = self.state["settings"]
        self.image_size = int(self.state["image_size"])
        self.source_root, observed = _source_commit(method)
        if observed != self.state["official_source_commit"]:
            raise AdapterError("Official source commit changed after adapter fitting")
        self.device = _require_device(self.settings)
        self.loaded_domain: str | None = None
        self.model: Any = None
        self.auxiliary: Any = None

    def _domain_checkpoint(self, domain: str) -> tuple[Path, dict]:
        if domain not in self.state["domains"]:
            raise AdapterError(
                f"No fitted {self.method} target model for domain {domain}"
            )
        details = self.state["domains"][domain]
        path = self.checkpoint_dir / details["checkpoint"]
        if not path.is_file():
            raise AdapterError(f"Adapter checkpoint is missing: {path}")
        return path, details

    def _clear(self) -> None:
        self.model = None
        self.auxiliary = None
        self.loaded_domain = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, domain: str) -> None:
        if self.loaded_domain == domain:
            return
        self._clear()
        checkpoint_path, _ = self._domain_checkpoint(domain)
        payload = _load_torch(checkpoint_path)
        if payload.get("fit_fingerprint") != self.state.get("fit_fingerprint"):
            raise AdapterError(
                f"Checkpoint fingerprint does not match adapter state: {checkpoint_path}"
            )
        if self.method == "stainnet":
            model = _stainnet_model(self.source_root, self.settings)
            model.load_state_dict(payload["model"])
        elif self.method in {"staingan", "cyclegan"}:
            context = {
                "method": self.method,
                "settings": self.settings,
                "source_root": self.source_root,
            }
            model, reverse, d_a, d_b = _make_cycle_models(context)
            model.load_state_dict(payload["model"])
            del reverse, d_a, d_b
        elif self.method == "pix2pix":
            context = {"settings": self.settings, "source_root": self.source_root}
            model, discriminator = _make_pix2pix_models(context)
            model.load_state_dict(payload["model"])
            del discriminator
        elif self.method == "cagan":
            model, discriminator = _cagan_models(
                self.source_root, self.settings, int(self.settings["batch_size"])
            )
            model.load_state_dict(payload["model"])
            del discriminator
        elif self.method == "histaugan":
            model, _ = _histaugan_model(
                self.source_root, self.settings, len(self.state["domains"])
            )
            model.enc_c.load_state_dict(payload["enc_c"])
            model.enc_a.load_state_dict(payload["enc_a"])
            model.gen.load_state_dict(payload["gen"])
        elif self.method == "sastaindiff":
            model, diffusion = _sastaindiff_components(
                self.settings, self.image_size, inference=True
            )
            model.load_state_dict(payload["model"])
            self.auxiliary = diffusion
        else:  # pragma: no cover - guarded by CLI and registry
            raise AdapterError(self.method)
        model.to(self.device).eval()
        model.requires_grad_(False)
        self.model = model
        self.loaded_domain = domain

    def infer(self, request: dict) -> np.ndarray:
        target = (
            str(request["target_scanner"])
            if self.method == "pix2pix"
            else str(request["target_stain"])
        )
        self._load(target)
        source_path = Path(request["source_path"])
        if not source_path.is_file():
            raise AdapterError(f"Inference source image is missing: {source_path}")
        if self.method == "cagan":
            source = _rgb_tensor(source_path, self.image_size, unit=True).unsqueeze(0)
            source = source.to(self.device)
            with torch.inference_mode():
                first, second = self.model(_gray3(source))
                output = (first + second) / 2
            return _unit_to_uint8(output)
        source = _rgb_tensor(source_path, self.image_size).unsqueeze(0).to(self.device)
        if self.method == "histaugan":
            details = self.state["domains"][target]
            prototype = (
                _rgb_tensor(details["prototype_path"], self.image_size)
                .unsqueeze(0)
                .to(self.device)
            )
            domain_index = int(details["domain_index"])
            onehot = torch.zeros(1, len(self.state["domains"]), device=self.device)
            onehot[:, domain_index] = 1
            with torch.inference_mode():
                content = self.model.enc_c(source)
                mean, _ = self.model.enc_a(prototype, onehot)
                output = self.model.gen(content, mean, onehot)
            return _signed_to_uint8(output)
        if self.method == "sastaindiff":
            digest = hashlib.sha256(
                source.detach().cpu().numpy().tobytes() + target.encode()
            ).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**31)
            torch.manual_seed(seed)
            diffusion = self.auxiliary
            timestep = torch.full(
                (1,), len(diffusion.betas) - 1, device=self.device, dtype=torch.long
            )
            noise = diffusion.q_sample(source, timestep)

            def model_fn(x, he, t, y=None, ref_img=None):
                return self.model(x, he, t, None)

            with torch.inference_mode():
                output = diffusion.ddim_sample_loop(
                    model_fn,
                    source.shape,
                    HE=source,
                    noise=noise,
                    clip_denoised=True,
                    device=self.device,
                    progress=False,
                )
            return _signed_to_uint8(output)
        with torch.inference_mode():
            output = self.model(source)
        return _signed_to_uint8(output)


def _signed_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor[0].detach().float().clamp(-1, 1).add(1).mul(127.5)
    return array.byte().permute(1, 2, 0).cpu().numpy()


def _unit_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor[0].detach().float().clamp(0, 1).mul(255)
    return array.byte().permute(1, 2, 0).cpu().numpy()


def infer_one(
    method: str, request: dict, *, runtime: AdapterRuntime | None = None
) -> None:
    active = runtime or AdapterRuntime(method, request)
    output = active.infer(request)
    destination = Path(request["output_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    Image.fromarray(output, mode="RGB").save(temporary)
    temporary.replace(destination)
