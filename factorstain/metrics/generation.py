from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn import functional as F


def generation_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    lpips_model=None,
    semantic_generated: torch.Tensor | None = None,
    semantic_target: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    gen = generated.detach().float().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    tgt = target.detach().float().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    lpips_values = None
    if lpips_model is not None:
        with torch.inference_mode():
            lpips_values = lpips_model(generated * 2 - 1, target * 2 - 1).flatten().detach().cpu().numpy()
    cosine = None
    if semantic_generated is not None and semantic_target is not None:
        cosine = F.cosine_similarity(semantic_generated, semantic_target).detach().cpu().numpy()
    rows = []
    for i, (a, b) in enumerate(zip(gen, tgt)):
        rows.append(
            {
                "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=1.0)),
                "psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
                "lpips": float(lpips_values[i]) if lpips_values is not None else float("nan"),
                "morphology_cosine": float(cosine[i]) if cosine is not None else float("nan"),
            }
        )
    return rows


def factor_isolation_score(
    before_target_probability: np.ndarray,
    after_target_probability: np.ndarray,
    before_non_target_probability: np.ndarray,
    after_non_target_probability: np.ndarray,
    penalty: float = 1.0,
) -> np.ndarray:
    target_delta = np.abs(after_target_probability - before_target_probability)
    non_target_delta = np.abs(after_non_target_probability - before_non_target_probability)
    return target_delta - penalty * non_target_delta

