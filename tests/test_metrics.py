import numpy as np
import pandas as pd

from factorstain.evaluation.attribution import factorial_sensitivity
from factorstain.metrics.statistics import paired_bootstrap


def test_factorial_decomposition_recovers_main_effects():
    matrix = 0.5 + np.array([-0.2, 0.0, 0.2])[:, None] + np.array([-0.1, 0.1])[None, :]
    result = factorial_sensitivity(matrix)
    assert result["stain_sensitivity"] > result["scanner_sensitivity"] > 0
    assert result["interaction_sensitivity"] < 1e-12


def test_paired_bootstrap_resamples_groups():
    rows = []
    for group in range(30):
        rows.extend([{"aligned_group_id": group, "method": "a", "score": 1.0}, {"aligned_group_id": group, "method": "b", "score": 0.5}])
    result = paired_bootstrap(pd.DataFrame(rows), "score", "a", "b", n_bootstrap=200, seed=42)
    assert result["ci_low"] > 0
    assert result["n_groups"] == 30
