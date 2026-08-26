from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def summarize_samples(values: np.ndarray | pd.Series, confidence: float = 0.95) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    alpha = 1 - confidence
    sem = array.std(ddof=1) / np.sqrt(len(array)) if len(array) > 1 else 0.0
    from scipy.stats import t
    delta = t.ppf(1 - alpha / 2, max(1, len(array) - 1)) * sem
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci_low": float(array.mean() - delta),
        "ci_high": float(array.mean() + delta),
        "n": int(len(array)),
    }


def paired_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    group_column: str = "aligned_group_id",
    method_column: str = "method",
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired cluster bootstrap, resampling aligned morphologies rather than images."""
    pivot = frame.pivot_table(index=group_column, columns=method_column, values=metric, aggfunc="mean")
    pairs = pivot[[method_a, method_b]].dropna()
    if pairs.empty:
        raise ValueError(f"No paired {metric} samples for {method_a} and {method_b}")
    differences = (pairs[method_a] - pairs[method_b]).to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        bootstrap[i] = rng.choice(differences, size=len(differences), replace=True).mean()
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "mean_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_superiority": float((bootstrap > 0).mean()),
        "n_groups": int(len(differences)),
    }
