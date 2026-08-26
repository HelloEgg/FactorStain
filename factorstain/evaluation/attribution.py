from __future__ import annotations

import numpy as np


def factorial_sensitivity(matrix: np.ndarray) -> dict[str, float | np.ndarray]:
    """Balanced two-way ANOVA decomposition P_ij = mu + alpha_i + beta_j + gamma_ij."""
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("Sensitivity decomposition requires a 2D factorial matrix with at least 2×2 cells")
    if not np.isfinite(matrix).all():
        raise ValueError("Sensitivity matrix has missing/non-finite cells")
    mu = matrix.mean()
    alpha = matrix.mean(axis=1) - mu
    beta = matrix.mean(axis=0) - mu
    gamma = matrix - mu - alpha[:, None] - beta[None, :]
    stain = float(np.mean(alpha**2))
    scanner = float(np.mean(beta**2))
    interaction = float(np.mean(gamma**2))
    total = stain + scanner + interaction
    return {
        "mu": float(mu),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "stain_sensitivity": stain,
        "scanner_sensitivity": scanner,
        "interaction_sensitivity": interaction,
        "stain_fraction": stain / max(total, 1e-12),
        "scanner_fraction": scanner / max(total, 1e-12),
        "interaction_fraction": interaction / max(total, 1e-12),
    }
