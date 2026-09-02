from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    distances = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1, keepdims=True).T
        - 2.0 * left @ right.T
    )
    return np.maximum(distances, 0.0, out=distances)


def median_bandwidth(
    left: np.ndarray,
    right: np.ndarray,
    max_distance_pairs: int = 20000,
    seed: int = 42,
) -> float:
    pooled = np.concatenate([left, right], axis=0)
    if len(pooled) < 2:
        raise ValueError("Median bandwidth requires at least two pooled observations")
    rng = np.random.default_rng(seed)
    pair_count = min(max_distance_pairs, len(pooled) * (len(pooled) - 1) // 2)
    if pair_count == len(pooled) * (len(pooled) - 1) // 2:
        distances = squared_distances(pooled, pooled)
        values = distances[np.triu_indices(len(pooled), k=1)]
    else:
        first = rng.integers(0, len(pooled), size=pair_count)
        second = rng.integers(0, len(pooled) - 1, size=pair_count)
        second = second + (second >= first)
        differences = pooled[first] - pooled[second]
        values = np.einsum("ij,ij->i", differences, differences)
    nonzero = values[np.isfinite(values) & (values > 1e-12)]
    if not len(nonzero):
        return 1.0
    # The heuristic is computed from squared distances; sigma is their median
    # Euclidean distance so the RBF denominator remains 2*sigma^2.
    return float(np.sqrt(np.median(nonzero)))


def rbf_kernel_from_distances(
    distances: np.ndarray, sigmas: Iterable[float]
) -> np.ndarray:
    sigma_values = np.asarray(list(sigmas), dtype=np.float64)
    if not len(sigma_values) or np.any(sigma_values <= 0):
        raise ValueError("All RBF bandwidths must be positive")
    kernel = np.zeros_like(distances, dtype=np.float64)
    for sigma in sigma_values:
        kernel += np.exp(-np.asarray(distances, dtype=np.float64) / (2.0 * sigma**2))
    return kernel / len(sigma_values)


def unbiased_mmd2_from_kernels(
    kernel_xx: np.ndarray,
    kernel_yy: np.ndarray,
    kernel_xy: np.ndarray,
) -> float:
    n_x, n_y = len(kernel_xx), len(kernel_yy)
    if n_x < 2 or n_y < 2:
        raise ValueError("Unbiased MMD requires at least two samples per domain")
    within_x = (kernel_xx.sum() - np.trace(kernel_xx)) / (n_x * (n_x - 1))
    within_y = (kernel_yy.sum() - np.trace(kernel_yy)) / (n_y * (n_y - 1))
    between = kernel_xy.mean()
    return float(within_x + within_y - 2.0 * between)


def mmd2_with_bandwidth(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    scales: Iterable[float] = (0.5, 1.0, 2.0),
) -> dict[str, float]:
    distances_xx = squared_distances(left, left)
    distances_yy = squared_distances(right, right)
    distances_xy = squared_distances(left, right)
    scale_values = tuple(float(scale) for scale in scales)
    multiscale_sigmas = [sigma * scale for scale in scale_values]
    multi_raw = unbiased_mmd2_from_kernels(
        rbf_kernel_from_distances(distances_xx, multiscale_sigmas),
        rbf_kernel_from_distances(distances_yy, multiscale_sigmas),
        rbf_kernel_from_distances(distances_xy, multiscale_sigmas),
    )
    single_raw = unbiased_mmd2_from_kernels(
        rbf_kernel_from_distances(distances_xx, [sigma]),
        rbf_kernel_from_distances(distances_yy, [sigma]),
        rbf_kernel_from_distances(distances_xy, [sigma]),
    )
    return {
        "sigma": float(sigma),
        "raw_mmd2": multi_raw,
        "reported_mmd2": max(multi_raw, 0.0),
        "single_raw_mmd2": single_raw,
        "single_reported_mmd2": max(single_raw, 0.0),
    }


def compute_mmd2(
    left: np.ndarray,
    right: np.ndarray,
    scales: Iterable[float] = (0.5, 1.0, 2.0),
    max_distance_pairs: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    sigma = median_bandwidth(left, right, max_distance_pairs, seed)
    return mmd2_with_bandwidth(left, right, sigma, scales)


def balanced_domain_indices(
    labels: Iterable[object],
    domain_a: object,
    domain_b: object,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    values = np.asarray(list(labels)).astype(str)
    positions_a = np.flatnonzero(values == str(domain_a))
    positions_b = np.flatnonzero(values == str(domain_b))
    count = min(len(positions_a), len(positions_b), max_samples)
    if count < 2:
        raise ValueError(
            f"Domains {domain_a!r}/{domain_b!r} have fewer than two usable samples"
        )
    rng = np.random.default_rng(seed)
    selected_a = np.sort(rng.choice(positions_a, size=count, replace=False))
    selected_b = np.sort(rng.choice(positions_b, size=count, replace=False))
    return (
        selected_a,
        selected_b,
        {
            "n_a_available": len(positions_a),
            "n_b_available": len(positions_b),
            "n_used": count,
            "selection": "equal_size_domain_subsample",
        },
    )


def controlled_scanner_indices(
    metadata: pd.DataFrame,
    scanner_a: object,
    scanner_b: object,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frame = metadata.reset_index(drop=True).copy()
    frame["feature_position"] = np.arange(len(frame))
    keys = ["aligned_group_id", "stain_id"]
    left = frame[frame.scanner_id.astype(str) == str(scanner_a)][
        [*keys, "tissue_type", "feature_position"]
    ].rename(columns={"feature_position": "position_a", "tissue_type": "tissue_a"})
    right = frame[frame.scanner_id.astype(str) == str(scanner_b)][
        [*keys, "tissue_type", "feature_position"]
    ].rename(columns={"feature_position": "position_b", "tissue_type": "tissue_b"})
    matched = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if matched.empty:
        raise ValueError(
            f"No aligned-group/stain matches for scanners {scanner_a}/{scanner_b}"
        )
    if not (matched.tissue_a.astype(str) == matched.tissue_b.astype(str)).all():
        raise ValueError(
            f"Tissue metadata disagrees within aligned groups for {scanner_a}/{scanner_b}"
        )
    matched = matched.sort_values([*keys, "position_a", "position_b"]).reset_index(
        drop=True
    )
    available = len(matched)
    if available > max_samples:
        rng = np.random.default_rng(seed)
        matched = matched.iloc[
            np.sort(rng.choice(available, max_samples, replace=False))
        ]
    if len(matched) < 2:
        raise ValueError(
            f"Fewer than two controlled matches for scanners {scanner_a}/{scanner_b}"
        )
    return (
        matched.position_a.to_numpy(dtype=int),
        matched.position_b.to_numpy(dtype=int),
        matched.aligned_group_id.astype(str).to_numpy(),
        {
            "n_matched_available": available,
            "n_used": len(matched),
            "n_aligned_groups": matched.aligned_group_id.nunique(),
            "selection": "same_aligned_group_and_stain",
            "bootstrap_unit": "aligned_group",
        },
    )


def balanced_stain_indices(
    metadata: pd.DataFrame,
    stain_a: object,
    stain_b: object,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    frame = metadata.reset_index(drop=True).copy()
    frame["feature_position"] = np.arange(len(frame))
    strata = ["tissue_type", "scanner_id"]
    left = frame[frame.stain_id.astype(str) == str(stain_a)]
    right = frame[frame.stain_id.astype(str) == str(stain_b)]
    rng = np.random.default_rng(seed)
    records: list[tuple[int, int, str]] = []
    common = (
        left[strata].drop_duplicates().merge(right[strata].drop_duplicates(), on=strata)
    )
    for _, stratum in common.sort_values(strata).iterrows():
        mask_a = np.ones(len(left), dtype=bool)
        mask_b = np.ones(len(right), dtype=bool)
        for column in strata:
            mask_a &= left[column].astype(str).to_numpy() == str(stratum[column])
            mask_b &= right[column].astype(str).to_numpy() == str(stratum[column])
        candidates_a = left.loc[mask_a, "feature_position"].to_numpy(dtype=int)
        candidates_b = right.loc[mask_b, "feature_position"].to_numpy(dtype=int)
        count = min(len(candidates_a), len(candidates_b))
        if not count:
            continue
        chosen_a = rng.choice(candidates_a, size=count, replace=False)
        chosen_b = rng.choice(candidates_b, size=count, replace=False)
        stratum_id = f"{stratum.tissue_type!s}|{stratum.scanner_id!s}"
        records.extend(
            (int(position_a), int(position_b), stratum_id)
            for position_a, position_b in zip(chosen_a, chosen_b)
        )
    available = len(records)
    if available > max_samples:
        selected = np.sort(rng.choice(available, max_samples, replace=False))
        records = [records[int(position)] for position in selected]

    if len(records) >= 2:
        return (
            np.asarray([record[0] for record in records], dtype=int),
            np.asarray([record[1] for record in records], dtype=int),
            np.asarray([record[2] for record in records]),
            {
                "n_balanced_available": available,
                "n_used": len(records),
                "n_strata": len({record[2] for record in records}),
                "selection": "exact_tissue_and_scanner_strata",
                "bootstrap_unit": "tissue_scanner_stratum",
                "joint_match_fraction": 1.0,
                "tissue_match_fraction": 1.0,
                "scanner_match_fraction": 1.0,
                "mean_categorical_mismatch": 0.0,
            },
        )

    # Some stain pairs have no usable overlap in the joint tissue/scanner
    # strata. Retaining all such pairs is preferable to silently dropping a
    # stain from the audit, but the fallback must not be described as exact
    # stratification. Find the deterministic minimum-cost bipartite matching,
    # where a tissue or scanner mismatch each costs one.
    candidates_a = left.feature_position.to_numpy(dtype=int)
    candidates_b = right.feature_position.to_numpy(dtype=int)
    if len(candidates_a) > max_samples:
        candidates_a = np.sort(
            rng.choice(candidates_a, size=max_samples, replace=False)
        )
    if len(candidates_b) > max_samples:
        candidates_b = np.sort(
            rng.choice(candidates_b, size=max_samples, replace=False)
        )
    if min(len(candidates_a), len(candidates_b)) < 2:
        raise ValueError(
            f"Fewer than two samples for stains {stain_a}/{stain_b}; "
            "minimum-mismatch balancing is unavailable"
        )

    metadata_a = frame.iloc[candidates_a]
    metadata_b = frame.iloc[candidates_b]
    tissue_match = (
        metadata_a.tissue_type.astype(str).to_numpy()[:, None]
        == metadata_b.tissue_type.astype(str).to_numpy()[None, :]
    )
    scanner_match = (
        metadata_a.scanner_id.astype(str).to_numpy()[:, None]
        == metadata_b.scanner_id.astype(str).to_numpy()[None, :]
    )
    mismatch_cost = (~tissue_match).astype(np.int8) + (~scanner_match).astype(np.int8)
    rows, columns = linear_sum_assignment(mismatch_cost)
    selected_a = candidates_a[rows]
    selected_b = candidates_b[columns]
    paired_tissue_match = tissue_match[rows, columns]
    paired_scanner_match = scanner_match[rows, columns]
    pair_ids = np.asarray(
        [f"minimum_mismatch_pair_{position:05d}" for position in range(len(rows))]
    )
    return (
        selected_a,
        selected_b,
        pair_ids,
        {
            "n_balanced_available": available,
            "n_used": len(rows),
            "n_strata": 0,
            "selection": "minimum_categorical_mismatch_fallback",
            "bootstrap_unit": "matched_categorical_pair",
            "joint_match_fraction": float(
                np.mean(paired_tissue_match & paired_scanner_match)
            ),
            "tissue_match_fraction": float(np.mean(paired_tissue_match)),
            "scanner_match_fraction": float(np.mean(paired_scanner_match)),
            "mean_categorical_mismatch": float(np.mean(mismatch_cost[rows, columns])),
        },
    )


def _kernel_matrices(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    scales: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigmas = [sigma * float(scale) for scale in scales]
    return (
        rbf_kernel_from_distances(squared_distances(left, left), sigmas),
        rbf_kernel_from_distances(squared_distances(right, right), sigmas),
        rbf_kernel_from_distances(squared_distances(left, right), sigmas),
    )


def _weighted_mmd_batch(
    kernel_xx: np.ndarray,
    kernel_yy: np.ndarray,
    kernel_xy: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
) -> np.ndarray:
    count_x = weights_x.sum(axis=1)
    count_y = weights_y.sum(axis=1)
    within_x = np.sum((weights_x @ kernel_xx) * weights_x, axis=1)
    within_x -= weights_x @ np.diag(kernel_xx)
    within_x /= count_x * (count_x - 1)
    within_y = np.sum((weights_y @ kernel_yy) * weights_y, axis=1)
    within_y -= weights_y @ np.diag(kernel_yy)
    within_y /= count_y * (count_y - 1)
    between = np.sum((weights_x @ kernel_xy) * weights_y, axis=1)
    between /= count_x * count_y
    return within_x + within_y - 2.0 * between


def bootstrap_mmd2(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    scales: Iterable[float],
    replicates: int,
    seed: int,
    cluster_ids: np.ndarray | None = None,
    strata_ids: np.ndarray | None = None,
    cluster_label: str | None = None,
) -> dict[str, float | int | str]:
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    kernel_xx, kernel_yy, kernel_xy = _kernel_matrices(left, right, sigma, scales)
    rng = np.random.default_rng(seed)
    if cluster_ids is not None:
        cluster_values, membership = np.unique(
            cluster_ids.astype(str), return_inverse=True
        )
        cluster_counts = rng.multinomial(
            len(cluster_values),
            np.full(len(cluster_values), 1.0 / len(cluster_values)),
            size=replicates,
        )
        weights_x = cluster_counts[:, membership].astype(np.float64)
        weights_y = weights_x.copy()
        method = f"{cluster_label or 'aligned_group'}_cluster_bootstrap"
    elif strata_ids is not None:
        strata_values = np.asarray(strata_ids).astype(str)
        weights_x = np.zeros((replicates, len(left)), dtype=np.float64)
        weights_y = np.zeros((replicates, len(right)), dtype=np.float64)
        for stratum in sorted(np.unique(strata_values)):
            positions = np.flatnonzero(strata_values == stratum)
            probability = np.full(len(positions), 1.0 / len(positions))
            weights_x[:, positions] = rng.multinomial(
                len(positions), probability, size=replicates
            )
            weights_y[:, positions] = rng.multinomial(
                len(positions), probability, size=replicates
            )
        method = "within_tissue_scanner_strata_bootstrap"
    else:
        weights_x = rng.multinomial(
            len(left), np.full(len(left), 1.0 / len(left)), size=replicates
        ).astype(np.float64)
        weights_y = rng.multinomial(
            len(right), np.full(len(right), 1.0 / len(right)), size=replicates
        ).astype(np.float64)
        method = "independent_sample_bootstrap"
    raw = _weighted_mmd_batch(kernel_xx, kernel_yy, kernel_xy, weights_x, weights_y)
    reported = np.maximum(raw, 0.0)
    low, high = np.quantile(reported, [0.025, 0.975])
    return {
        "bootstrap_mean_mmd2": float(reported.mean()),
        "bootstrap_raw_mean_mmd2": float(raw.mean()),
        "ci_lower": float(low),
        "ci_upper": float(high),
        "bootstrap_replicates": int(replicates),
        "bootstrap_method": method,
        "bootstrap_bandwidth": "fixed_pairwise_median_sigma",
    }


def permutation_test_mmd2(
    left: np.ndarray,
    right: np.ndarray,
    sigma: float,
    scales: Iterable[float],
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    pooled = np.concatenate([left, right], axis=0)
    kernel = rbf_kernel_from_distances(
        squared_distances(pooled, pooled), [sigma * float(scale) for scale in scales]
    )
    observed = mmd2_with_bandwidth(left, right, sigma, scales)["raw_mmd2"]
    n_left, total = len(left), len(pooled)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for replicate in range(permutations):
        order = rng.permutation(total)
        selected_left, selected_right = order[:n_left], order[n_left:]
        null[replicate] = unbiased_mmd2_from_kernels(
            kernel[np.ix_(selected_left, selected_left)],
            kernel[np.ix_(selected_right, selected_right)],
            kernel[np.ix_(selected_left, selected_right)],
        )
    p_value = (1.0 + float(np.sum(null >= observed))) / (permutations + 1.0)
    return {
        "observed_raw_mmd2": float(observed),
        "permutation_p_value": p_value,
        "permutations": int(permutations),
    }


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def matrix_from_pair_records(
    records: pd.DataFrame,
    domains: Iterable[object],
    value_column: str,
) -> pd.DataFrame:
    ordered = [str(domain) for domain in domains]
    matrix = pd.DataFrame(0.0, index=ordered, columns=ordered)
    for _, row in records.iterrows():
        left, right = str(row.domain_a), str(row.domain_b)
        matrix.loc[left, right] = float(row[value_column])
        matrix.loc[right, left] = float(row[value_column])
    # ``DataFrame.values`` is a read-only view under pandas Copy-on-Write
    # (and by default in newer pandas releases). Use pandas' scalar setter so
    # matrix construction works independently of the backing-array policy.
    for position in range(len(matrix)):
        matrix.iat[position, position] = 0.0
    return matrix


def off_diagonal_summary(matrix: pd.DataFrame) -> dict[str, float | int]:
    values = matrix.to_numpy()[np.triu_indices(len(matrix), k=1)]
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "iqr": float(q3 - q1),
        "min": float(values.min()),
        "max": float(values.max()),
        "n_pairs": len(values),
    }
