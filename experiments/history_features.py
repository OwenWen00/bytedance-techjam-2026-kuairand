from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from data import LABEL

HISTORY_BUCKET_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def _bucketize_rate(rate: float, edges: np.ndarray = HISTORY_BUCKET_EDGES) -> int:
    value = float(rate)
    if not np.isfinite(value):
        raise ValueError(f"Historical long_view rate is not finite: {value!r}")
    if value <= 0.0:
        return 0
    if value >= 1.0:
        return len(edges) - 2
    idx = int(np.searchsorted(edges, value, side="right")) - 1
    idx = max(0, min(idx, len(edges) - 2))
    return idx


def _get_user_author_key(row: Sequence[object]) -> Tuple[str, str]:
    return str(row[1]), str(row[3])


def _train_global_rate(rows: Sequence[Sequence[object]]) -> float:
    if not rows:
        return 0.5
    rate = 0.5
    if not np.isfinite(rate):
        raise ValueError("Train-only historical fallback rate is not finite.")
    return rate


def build_causal_history_features(train_rows: Sequence[Sequence[object]], valid_rows: Sequence[Sequence[object]] | None = None, fallback_rate: float | None = None) -> Tuple[np.ndarray, np.ndarray, float]:
    if fallback_rate is None:
        fallback_rate = _train_global_rate(train_rows)
    fallback_rate = float(fallback_rate)
    if not np.isfinite(fallback_rate):
        raise ValueError("Fallback rate is not finite.")

    counts: Dict[Tuple[str, str], int] = {}
    positives: Dict[Tuple[str, str], int] = {}
    train_features = np.empty(len(train_rows), dtype=np.int32)

    for index, row in enumerate(train_rows):
        key = _get_user_author_key(row)
        total = counts.get(key, 0)
        pos = positives.get(key, 0)
        prior_rate = (pos / total) if total > 0 else fallback_rate
        if not np.isfinite(prior_rate):
            raise ValueError(f"Non-finite historical rate encountered at train row {index}: {prior_rate!r}")
        train_features[index] = _bucketize_rate(prior_rate)
        counts[key] = total + 1
        positives[key] = pos + int(row[6])

    if valid_rows is None:
        return train_features, np.empty((0,), dtype=np.int32), fallback_rate

    valid_features = np.empty(len(valid_rows), dtype=np.int32)
    for index, row in enumerate(valid_rows):
        key = _get_user_author_key(row)
        total = counts.get(key, 0)
        pos = positives.get(key, 0)
        prior_rate = (pos / total) if total > 0 else fallback_rate
        if not np.isfinite(prior_rate):
            raise ValueError(f"Non-finite historical rate encountered at valid row {index}: {prior_rate!r}")
        valid_features[index] = _bucketize_rate(prior_rate)
    return train_features, valid_features, fallback_rate


def append_history_feature(X: np.ndarray, history_bucket: np.ndarray, bucket_count: int = len(HISTORY_BUCKET_EDGES) - 1) -> np.ndarray:
    X = np.asarray(X, dtype=np.int32)
    history_bucket = np.asarray(history_bucket, dtype=np.int32)
    if X.ndim != 2:
        raise ValueError("Feature matrix must be 2D.")
    if history_bucket.ndim != 1 or len(history_bucket) != X.shape[0]:
        raise ValueError("History bucket vector must be 1D and align to the row count.")
    if bucket_count <= 0:
        raise ValueError("History bucket count must be positive.")
    if np.any(~np.isfinite(history_bucket.astype(np.float64))):
        raise ValueError("History feature values must be finite.")
    max_bucket = int(history_bucket.max())
    if max_bucket >= bucket_count:
        raise ValueError(f"History bucket index {max_bucket} exceeds bucket_count={bucket_count}.")
    return np.concatenate([X, history_bucket[:, None]], axis=1)
