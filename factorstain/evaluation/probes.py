from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_feature_cache(path: str | Path) -> tuple[list[str], np.ndarray]:
    with h5py.File(path, "r") as handle:
        image_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["image_id"][:]]
        features = handle["features"][:].astype(np.float32)
    return image_ids, features


def fit_grouped_probe(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classifier: str = "logistic",
    test_size: float = 0.20,
    seed: int = 42,
    max_iter: int = 1000,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(str))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(features, encoded, groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("Aligned-group leakage in foundation-model probe")
    if classifier == "mlp":
        estimator = MLPClassifier(hidden_layer_sizes=(256,), max_iter=min(max_iter, 200), early_stopping=True, random_state=seed)
    else:
        estimator = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=seed)
    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(features[train], encoded[train])
    predictions = pipeline.predict(features[test])
    metrics = {
        "accuracy": float(accuracy_score(encoded[test], predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(encoded[test], predictions)),
        "macro_f1": float(f1_score(encoded[test], predictions, average="macro")),
        "chance": float(1 / len(encoder.classes_)),
        "n_classes": int(len(encoder.classes_)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "classes": encoder.classes_.tolist(),
    }
    return metrics, confusion_matrix(encoded[test], predictions, labels=np.arange(len(encoder.classes_))), test, predictions


def probe_all_labels(
    features: np.ndarray,
    metadata: pd.DataFrame,
    classifier: str = "logistic",
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows, confusion = [], {}
    for label_name, column in (("tissue", "tissue_type"), ("stain", "stain_id"), ("scanner", "scanner_id")):
        metrics, matrix, _, _ = fit_grouped_probe(
            features,
            metadata[column].to_numpy(),
            metadata.aligned_group_id.astype(str).to_numpy(),
            classifier=classifier,
            test_size=test_size,
            seed=seed,
        )
        rows.extend({"target": label_name, "metric": key, "value": value} for key, value in metrics.items() if isinstance(value, (int, float)))
        confusion[label_name] = matrix
    return rows, confusion


def within_group_embedding_variance(features: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        if len(positions) < 2:
            continue
        values = features[positions]
        centered = values - values.mean(axis=0, keepdims=True)
        cosine = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)
        similarity = cosine @ cosine.T
        upper = similarity[np.triu_indices(len(values), k=1)]
        rows.append(
            {
                "aligned_group_id": group,
                "embedding_variance": float(np.mean(centered**2)),
                "mean_cosine_distance": float(np.mean(1 - upper)),
                "n_acquisitions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def tissue_prediction_variation(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure true-tissue probability and prediction flips across acquisitions of held-out morphologies."""
    encoder = LabelEncoder().fit(labels.astype(str))
    encoded = encoder.transform(labels.astype(str))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(splitter.split(features, encoded, groups))
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    estimator.fit(features[train], encoded[train])
    probabilities = estimator.predict_proba(features[test])
    predictions = estimator.predict(features[test])
    rows = []
    for group in np.unique(groups[test]):
        positions = np.flatnonzero(groups[test] == group)
        if len(positions) < 2:
            continue
        true_class = encoded[test][positions][0]
        class_position = list(estimator.classes_).index(true_class)
        group_predictions = predictions[positions]
        rows.append(
            {
                "aligned_group_id": str(group),
                "true_tissue_probability_std": float(probabilities[positions, class_position].std()),
                "prediction_flip_rate": float(1 - pd.Series(group_predictions).value_counts(normalize=True).max()),
                "n_acquisitions": int(len(positions)),
            }
        )
    return pd.DataFrame(rows)
