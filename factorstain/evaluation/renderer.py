from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage.color import rgb2hed
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from factorstain.models.fm_registry import encode_batch, load_foundation_model, preprocess_tensor_batch
from factorstain.training.renderer import _load_image, build_renderer


class EvaluationPairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, index: pd.DataFrame, image_size: int, stain_map: dict, scanner_map: dict) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.index = index
        self.image_size = image_size
        self.stain_map = stain_map
        self.scanner_map = scanner_map

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, item: int) -> dict[str, Any]:
        pair = self.pairs.iloc[item]
        source = self.index.loc[int(pair.source_index)]
        target = self.index.loc[int(pair.target_index)]
        return {
            "source": _load_image(source.image_path, self.image_size),
            "target": _load_image(target.image_path, self.image_size),
            "source_stain": self.stain_map[source.stain_id],
            "source_scanner": self.scanner_map[source.scanner_id],
            "target_stain": self.stain_map[target.stain_id],
            "target_scanner": self.scanner_map[target.scanner_id],
            "group": str(target.aligned_group_id),
            "target_index": int(pair.target_index),
            "source_index": int(pair.source_index),
            "protocol": str(pair.protocol),
        }


def build_evaluation_pairs(index: pd.DataFrame, split: dict, seed: int = 42, limit: int | None = None) -> pd.DataFrame:
    heldout = {(item["stain_id"], item["scanner_id"]) for item in split["heldout_cells"]}
    train_cells = {(item["stain_id"], item["scanner_id"]) for item in split["train_cells"]}
    rows = []
    rng = np.random.default_rng(seed)
    existing = index[index.image_exists].copy()
    existing["cell"] = list(zip(existing.stain_id.astype(str), existing.scanner_id.astype(str)))
    for group_id, group in existing.groupby("aligned_group_id"):
        train_sources = group[group.cell.isin(train_cells)]
        if train_sources.empty:
            continue
        for target_index, target in group.iterrows():
            cell = target.cell
            if cell in heldout and target.morphology_split == "train":
                protocol = "unseen_combo_seen_morphology"
            elif cell in heldout and target.morphology_split == "test":
                protocol = "unseen_combo_unseen_morphology"
            elif cell in train_cells and target.morphology_split == "test":
                protocol = "seen_combo_unseen_morphology"
            else:
                continue
            candidates = train_sources[train_sources.index != target_index]
            if candidates.empty:
                continue
            preferred = candidates[candidates.stain_id.ne(target.stain_id) & candidates.scanner_id.ne(target.scanner_id)]
            chosen_from = preferred if not preferred.empty else candidates
            source_index = int(rng.choice(chosen_from.index.to_numpy()))
            rows.append({"source_index": source_index, "target_index": int(target_index), "protocol": protocol, "aligned_group_id": group_id})
    pairs = pd.DataFrame(rows)
    if limit and len(pairs):
        sampled = [frame.sample(min(limit, len(frame)), random_state=seed) for _, frame in pairs.groupby("protocol")]
        pairs = pd.concat(sampled, ignore_index=True)
    return pairs


class RealAcquisitionProbes:
    def __init__(self, feature_cache: Path, metadata: pd.DataFrame, train_cells: set[tuple[str, str]]) -> None:
        with h5py.File(feature_cache, "r") as handle:
            image_ids = [v.decode() if isinstance(v, bytes) else str(v) for v in handle["image_id"][:]]
            features = handle["features"][:].astype(np.float32)
        feature_frame = pd.DataFrame({"image_id": image_ids, "feature_position": np.arange(len(image_ids))})
        merged = feature_frame.merge(metadata, on="image_id", how="inner")
        merged = merged[
            merged.morphology_split.eq("train")
            & merged.apply(lambda row: (str(row.stain_id), str(row.scanner_id)) in train_cells, axis=1)
        ]
        if merged.empty:
            raise RuntimeError("No cached real training features are available for independent acquisition probes")
        train_features = features[merged.feature_position.to_numpy()]
        self.stain_encoder = LabelEncoder().fit(metadata.stain_id.astype(str))
        self.scanner_encoder = LabelEncoder().fit(metadata.scanner_id.astype(str))
        self.stain = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1))
        self.scanner = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1))
        self.stain.fit(train_features, self.stain_encoder.transform(merged.stain_id.astype(str)))
        self.scanner.fit(train_features, self.scanner_encoder.transform(merged.scanner_id.astype(str)))

    def probabilities(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.stain.predict_proba(features), self.scanner.predict_proba(features)


def _hed_distance(generated: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    gen = generated.detach().cpu().float().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    tgt = target.detach().cpu().float().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    distances = []
    for a, b in zip(gen, tgt):
        ah, bh = rgb2hed(a), rgb2hed(b)
        statistics_a = np.concatenate([ah[..., :2].mean((0, 1)), ah[..., :2].std((0, 1))])
        statistics_b = np.concatenate([bh[..., :2].mean((0, 1)), bh[..., :2].std((0, 1))])
        distances.append(np.linalg.norm(statistics_a - statistics_b))
    return np.asarray(distances)


@torch.inference_mode()
def evaluate_model(
    model_name: str,
    checkpoint: Path,
    config: dict,
    index: pd.DataFrame,
    pairs: pd.DataFrame,
    foundation,
    probes: RealAcquisitionProbes,
    lpips_model,
    device: torch.device,
    preview_limit: int = 3,
) -> tuple[pd.DataFrame, list[dict[str, torch.Tensor]]]:
    stain_map = {value: i for i, value in enumerate(sorted(index.stain_id.unique()))}
    scanner_map = {value: i for i, value in enumerate(sorted(index.scanner_id.unique()))}
    dataset = EvaluationPairDataset(pairs, index, config["image_size"], stain_map, scanner_map)
    loader = DataLoader(dataset, batch_size=max(1, min(config["batch_size"], 32)), num_workers=config["num_workers"], pin_memory=True)
    model = build_renderer(model_name, len(stain_map), len(scanner_map)).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    rows, previews = [], []
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for batch in loader:
        source, target = batch["source"].to(device), batch["target"].to(device)
        source_stain, source_scanner = batch["source_stain"].to(device), batch["source_scanner"].to(device)
        target_stain, target_scanner = batch["target_stain"].to(device), batch["target_scanner"].to(device)
        with torch.autocast("cuda", dtype=autocast_dtype):
            generated = model(source, target_stain, target_scanner)
            stain_cf = model(source, target_stain, source_scanner)
            scanner_cf = model(source, source_stain, target_scanner)
            fm_input = torch.cat([source, target, generated, stain_cf, scanner_cf])
            features = encode_batch(foundation, preprocess_tensor_batch(foundation, fm_input)).float()
        generated, stain_cf, scanner_cf = generated.float(), stain_cf.float(), scanner_cf.float()
        source_f, target_f, generated_f, stain_f, scanner_f = features.chunk(5)
        stain_source_p, scanner_source_p = probes.probabilities(source_f.cpu().numpy())
        stain_generated_p, scanner_generated_p = probes.probabilities(generated_f.cpu().numpy())
        stain_cf_p, scanner_stain_cf_p = probes.probabilities(stain_f.cpu().numpy())
        stain_scanner_cf_p, scanner_cf_p = probes.probabilities(scanner_f.cpu().numpy())
        lpips_values = lpips_model(generated * 2 - 1, target * 2 - 1).flatten().float().cpu().numpy()
        from torch.nn import functional as F
        cosine = F.cosine_similarity(generated_f, target_f).float().cpu().numpy()
        hed = _hed_distance(generated, target)
        generated_np = generated.float().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
        target_np = target.float().cpu().numpy().transpose(0, 2, 3, 1)
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        for i in range(len(source)):
            target_stain_idx, target_scanner_idx = int(target_stain[i]), int(target_scanner[i])
            source_stain_idx, source_scanner_idx = int(source_stain[i]), int(source_scanner[i])
            stain_gain = stain_cf_p[i, target_stain_idx] - stain_source_p[i, target_stain_idx]
            stain_nontarget_change = 0.5 * np.abs(scanner_stain_cf_p[i] - scanner_source_p[i]).sum()
            scanner_gain = scanner_cf_p[i, target_scanner_idx] - scanner_source_p[i, target_scanner_idx]
            scanner_nontarget_change = 0.5 * np.abs(stain_scanner_cf_p[i] - stain_source_p[i]).sum()
            fis = 0.5 * ((stain_gain - stain_nontarget_change) + (scanner_gain - scanner_nontarget_change))
            rows.append(
                {
                    "method": model_name,
                    "protocol": batch["protocol"][i],
                    "aligned_group_id": batch["group"][i],
                    "source_index": int(batch["source_index"][i]),
                    "target_index": int(batch["target_index"][i]),
                    "lpips": float(lpips_values[i]),
                    "ssim": float(structural_similarity(target_np[i], generated_np[i], channel_axis=2, data_range=1.0)),
                    "psnr": float(peak_signal_noise_ratio(target_np[i], generated_np[i], data_range=1.0)),
                    "morphology_cosine": float(cosine[i]),
                    "he_distance": float(hed[i]),
                    "stain_target_correct": float(stain_generated_p[i].argmax() == target_stain_idx),
                    "scanner_target_correct": float(scanner_generated_p[i].argmax() == target_scanner_idx),
                    "stain_non_target_preservation": float(1 - stain_nontarget_change),
                    "scanner_non_target_preservation": float(1 - scanner_nontarget_change),
                    "factor_isolation_score": float(fis),
                }
            )
            if len(previews) < preview_limit:
                previews.append({"source": source[i].cpu(), "target": target[i].cpu(), "generated": generated[i].cpu(), "stain_cf": stain_cf[i].cpu(), "scanner_cf": scanner_cf[i].cpu(), "target_index": int(batch["target_index"][i])})
    return pd.DataFrame(rows), previews
