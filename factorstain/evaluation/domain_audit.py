from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupShuffleSplit

from factorstain.evaluation.probes import fit_grouped_probe


def l2_normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def project_features(
    features: np.ndarray, settings: dict, seed: int
) -> dict[str, np.ndarray | int]:
    values = l2_normalize(features)
    if len(values) < 3:
        raise ValueError("At least three samples are required for PCA/UMAP")
    dimensions = min(int(settings["pca_dimensions"]), values.shape[1], len(values) - 1)
    pca = PCA(n_components=dimensions, svd_solver="auto", random_state=seed)
    reduced = pca.fit_transform(values).astype(np.float32)
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP analysis requires umap-learn; run bash shell/setup.sh"
        ) from exc
    neighbors = max(2, min(int(settings["umap_neighbors"]), len(values) - 1))
    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=float(settings["umap_min_dist"]),
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    ).fit_transform(reduced)
    return {
        "pca_2d": reduced[:, :2],
        "pca_reduced": reduced,
        "umap_2d": np.asarray(embedding, dtype=np.float32),
        "pca_dimensions": dimensions,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


def normalized_advantage(balanced_accuracy: float, chance: float) -> float:
    return float((balanced_accuracy - chance) / max(1e-12, 1.0 - chance))


def signal_label(advantage: float) -> str:
    if advantage >= 0.50:
        return "STRONG"
    if advantage >= 0.25:
        return "MODERATE"
    return "WEAK"


def grouped_probe(
    features: np.ndarray,
    labels: Iterable[object],
    groups: Iterable[object],
    target: str,
    dataset: str,
    test_size: float,
    seed: int,
    max_iter: int,
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    labels_array = np.asarray(list(labels)).astype(str)
    groups_array = np.asarray(list(groups)).astype(str)
    if len(np.unique(labels_array)) < 2 or len(np.unique(groups_array)) < 2:
        raise ValueError(
            f"Probe {dataset}/{target} requires at least two labels and groups"
        )
    selected_seed = None
    selected_split = None
    for offset in range(100):
        candidate_seed = seed + offset
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=candidate_seed
        )
        train, test = next(splitter.split(features, labels_array, groups_array))
        class_count = len(np.unique(labels_array))
        if (
            len(np.unique(labels_array[train])) == class_count
            and len(np.unique(labels_array[test])) == class_count
        ):
            selected_seed, selected_split = candidate_seed, (train, test)
            break
    if selected_split is None:
        raise RuntimeError(
            f"Could not construct a useful group-separated split for {dataset}/{target}"
        )
    train, test = selected_split
    train_groups, test_groups = set(groups_array[train]), set(groups_array[test])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(
            f"Group leakage in {dataset}/{target}: {sorted(overlap)[:5]}"
        )
    metrics, confusion, returned_test, _ = fit_grouped_probe(
        features,
        labels_array,
        groups_array,
        classifier="logistic",
        test_size=test_size,
        seed=int(selected_seed),
        max_iter=max_iter,
    )
    if not np.array_equal(returned_test, test):
        raise AssertionError(
            "Probe split reconstruction disagrees with evaluated split"
        )
    advantage = normalized_advantage(metrics["balanced_accuracy"], metrics["chance"])
    metrics.update(
        {
            "dataset": dataset,
            "target": target,
            "normalized_advantage": advantage,
            "domain_signal": signal_label(advantage),
            "split_seed": int(selected_seed),
            "n_train_groups": len(train_groups),
            "n_test_groups": len(test_groups),
            "group_overlap_count": 0,
        }
    )
    split = pd.DataFrame(
        {
            "dataset": dataset,
            "target": target,
            "sample_position": np.arange(len(features)),
            "group_id": groups_array,
            "partition": np.where(
                np.isin(np.arange(len(features)), test), "test", "train"
            ),
        }
    )
    return metrics, confusion, split


def descriptive_silhouette(
    features: np.ndarray, labels: Iterable[object], max_samples: int, seed: int
) -> dict:
    values = np.asarray(features, dtype=np.float32)
    labels_array = np.asarray(list(labels)).astype(str)
    if len(np.unique(labels_array)) < 2 or len(values) <= len(np.unique(labels_array)):
        return {"value": float("nan"), "n": len(values)}
    if len(values) > max_samples:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(values), size=max_samples, replace=False))
        values, labels_array = values[positions], labels_array[positions]
    return {
        "value": float(silhouette_score(values, labels_array, metric="euclidean")),
        "n": len(values),
    }


def domain_centroids(
    features: np.ndarray, labels: Iterable[object]
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    values = l2_normalize(features)
    labels_array = np.asarray(list(labels)).astype(str)
    domains = sorted(np.unique(labels_array))
    centroid_values, dispersion = [], []
    for domain in domains:
        selected = values[labels_array == domain]
        feature_variance = float(
            np.mean((selected - selected.mean(axis=0, keepdims=True)) ** 2)
        )
        centroid = selected.mean(axis=0)
        centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-12, None)
        centroid_values.append(centroid)
        distances = 1.0 - selected @ centroid
        dispersion.append(
            {
                "domain": domain,
                "n": len(selected),
                "within_domain_feature_variance": feature_variance,
                "within_domain_mean_cosine_distance": float(distances.mean()),
                "within_domain_std_cosine_distance": float(distances.std(ddof=1))
                if len(distances) > 1
                else 0.0,
            }
        )
    centroids = np.stack(centroid_values)
    distance_matrix = np.clip(1.0 - centroids @ centroids.T, 0.0, 2.0)
    return (
        pd.DataFrame(distance_matrix, index=domains, columns=domains),
        centroids,
        pd.DataFrame(dispersion),
    )


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = left / max(float(np.linalg.norm(left)), 1e-12)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    return float(np.clip(1.0 - np.dot(left, right), 0.0, 2.0))


def plism_controlled_distances(
    features: np.ndarray, metadata: pd.DataFrame, seed: int
) -> pd.DataFrame:
    values = l2_normalize(features)
    frame = metadata.reset_index(drop=True).copy()
    rows: list[dict] = []
    pair_number = 0
    for (group_id, stain), subset in frame.groupby(
        ["aligned_group_id", "stain_id"], sort=True
    ):
        for left, right in combinations(subset.index.tolist(), 2):
            if str(frame.loc[left, "scanner_id"]) == str(
                frame.loc[right, "scanner_id"]
            ):
                continue
            pair_number += 1
            rows.append(
                {
                    "pair_id": pair_number,
                    "bootstrap_group": str(group_id),
                    "category": "same morphology / same stain / different scanner",
                    "distance": _cosine_distance(values[left], values[right]),
                    "left_sample_id": frame.loc[left, "sample_id"],
                    "right_sample_id": frame.loc[right, "sample_id"],
                    "left_scanner": frame.loc[left, "scanner_id"],
                    "right_scanner": frame.loc[right, "scanner_id"],
                    "stain_id": stain,
                }
            )
    if not rows:
        raise RuntimeError(
            "No PLISM aligned same-stain/across-scanner pairs were present in the feature subset"
        )
    anchor_table = pd.DataFrame(rows)[
        ["bootstrap_group", "left_sample_id", "stain_id", "left_scanner"]
    ].drop_duplicates("bootstrap_group")
    anchor_table = anchor_table.merge(
        frame[["sample_id", "tissue_type"]].assign(
            sample_id=frame.sample_id.astype(str)
        ),
        left_on="left_sample_id",
        right_on="sample_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="sample_id")
    rng = np.random.default_rng(seed)
    sample_to_position = {
        str(value): position
        for position, value in enumerate(frame.sample_id.astype(str))
    }
    for _, anchor in anchor_table.iterrows():
        left = sample_to_position[str(anchor.left_sample_id)]
        base = frame[
            (frame.stain_id.astype(str) == str(anchor.stain_id))
            & (frame.aligned_group_id.astype(str) != str(anchor.bootstrap_group))
        ]
        tissue_matched = base[base.tissue_type.astype(str) == str(anchor.tissue_type)]
        if not tissue_matched.empty:
            base = tissue_matched
        same_scanner = base[base.scanner_id.astype(str) == str(anchor.left_scanner)]
        different_scanner = base[
            base.scanner_id.astype(str) != str(anchor.left_scanner)
        ]
        for category, candidates in (
            ("different morphology / same stain / same scanner", same_scanner),
            (
                "different morphology / same stain / different scanner",
                different_scanner,
            ),
        ):
            if candidates.empty:
                continue
            chosen = candidates.iloc[int(rng.integers(len(candidates)))]
            right = sample_to_position[str(chosen.sample_id)]
            pair_number += 1
            rows.append(
                {
                    "pair_id": pair_number,
                    "bootstrap_group": str(anchor.bootstrap_group),
                    "category": category,
                    "distance": _cosine_distance(values[left], values[right]),
                    "left_sample_id": frame.loc[left, "sample_id"],
                    "right_sample_id": chosen.sample_id,
                    "left_scanner": frame.loc[left, "scanner_id"],
                    "right_scanner": chosen.scanner_id,
                    "stain_id": anchor.stain_id,
                }
            )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    distances: pd.DataFrame,
    reference: str,
    comparison: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict:
    pivot = distances.pivot_table(
        index="bootstrap_group", columns="category", values="distance", aggfunc="mean"
    )
    paired = pivot[[reference, comparison]].dropna()
    if paired.empty:
        return {
            "reference": reference,
            "comparison": comparison,
            "mean_difference": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_groups": 0,
        }
    differences = (paired[reference] - paired[comparison]).to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            rng.choice(differences, size=len(differences), replace=True).mean()
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "reference": reference,
        "comparison": comparison,
        "mean_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_groups": len(differences),
        "bootstrap_unit": "aligned_group_id",
        "n_bootstrap": int(n_bootstrap),
    }


def sampled_within_between_distances(
    features: np.ndarray,
    labels: Iterable[object],
    sample_ids: Iterable[object],
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    values = l2_normalize(features)
    labels_array = np.asarray(list(labels)).astype(str)
    ids = np.asarray(list(sample_ids)).astype(str)
    rng = np.random.default_rng(seed)
    within: list[tuple[int, int]] = []
    between: list[tuple[int, int]] = []
    for domain in sorted(np.unique(labels_array)):
        positions = np.flatnonzero(labels_array == domain)
        domain_pairs = list(combinations(positions.tolist(), 2))
        if len(domain_pairs) > max_pairs // max(1, len(np.unique(labels_array))):
            chosen = rng.choice(
                len(domain_pairs),
                size=max_pairs // len(np.unique(labels_array)),
                replace=False,
            )
            domain_pairs = [domain_pairs[int(index)] for index in chosen]
        within.extend(domain_pairs)
    while len(between) < min(max_pairs, len(values) * 10):
        left, right = rng.integers(0, len(values), size=2)
        if left != right and labels_array[left] != labels_array[right]:
            between.append((int(left), int(right)))
    rows = []
    for category, pairs in (
        ("within scanner", within[:max_pairs]),
        ("between scanners", between[:max_pairs]),
    ):
        rows.extend(
            {
                "category": category,
                "distance": _cosine_distance(values[left], values[right]),
                "left_sample_id": ids[left],
                "right_sample_id": ids[right],
                "left_domain": labels_array[left],
                "right_domain": labels_array[right],
            }
            for left, right in pairs
        )
    return pd.DataFrame(rows)


def nearest_neighbors(
    features: np.ndarray,
    labels: Iterable[object],
    queries: int = 6,
    neighbors: int = 4,
    seed: int = 42,
) -> list[tuple[int, list[int]]]:
    values = l2_normalize(features)
    labels_array = np.asarray(list(labels)).astype(str)
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(len(values))
    selected: list[tuple[int, list[int]]] = []
    used_domains: set[str] = set()
    for query in candidates:
        similarities = values @ values[query]
        similarities[query] = -np.inf
        nearest = np.argsort(-similarities)[:neighbors]
        if not np.isfinite(similarities[nearest]).all():
            continue
        domain = labels_array[query]
        if domain in used_domains and len(used_domains) < len(np.unique(labels_array)):
            continue
        selected.append((int(query), nearest.astype(int).tolist()))
        used_domains.add(domain)
        if len(selected) == queries:
            break
    return selected
