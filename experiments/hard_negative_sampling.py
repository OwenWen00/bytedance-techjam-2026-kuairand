from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def _user_rng(seed: int, epoch: int, user_index: int) -> np.random.Generator:
    """Deterministic RNG seeded by seed, epoch, and user index."""
    state = (
        np.uint64(seed)
        ^ (np.uint64(epoch + 1) * np.uint64(131071))
        ^ (np.uint64(user_index + 1) * np.uint64(65537))
    )
    return np.random.default_rng(state)


def sample_hard_negative_pairs(
    user_ids: List[str],
    labels: np.ndarray,
    model_predictions: np.ndarray,
    seed: int = 0,
    epoch: int = 0,
    max_pairs_per_user: int = 128,
    hard_negative_candidates: int = 64,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    """
    Sample hard-negative BPR pairs using model-scored candidate selection.

    Args:
        user_ids: List of user identifiers per row.
        labels: Binary labels (1 for positive, 0 for negative).
        model_predictions: Current model scores for all rows.
        seed: Random seed for deterministic sampling.
        epoch: Epoch index for RNG variation.
        max_pairs_per_user: Maximum pairs to create per user.
        hard_negative_candidates: Candidate pool size per positive.

    Returns:
        (pos_indices, neg_indices, pair_count, candidate_count, skipped_users)
    """
    if labels is None:
        raise ValueError("Labels are required for hard-negative sampling.")
    if model_predictions is None:
        raise ValueError("Model predictions are required for hard-negative sampling.")
    if len(user_ids) != len(labels):
        raise ValueError("user_ids and labels must have the same length.")
    if len(user_ids) != len(model_predictions):
        raise ValueError("user_ids and model_predictions must have the same length.")
    if max_pairs_per_user <= 0:
        raise ValueError("max_pairs_per_user must be positive.")
    if hard_negative_candidates <= 0:
        raise ValueError("hard_negative_candidates must be positive.")

    if not np.all(np.isfinite(model_predictions)):
        raise ValueError("Model predictions contain non-finite values.")

    grouped_pos: Dict[str, List[int]] = defaultdict(list)
    grouped_neg: Dict[str, List[int]] = defaultdict(list)

    for idx, user in enumerate(user_ids):
        if labels[idx] == 1:
            grouped_pos[user].append(idx)
        elif labels[idx] == 0:
            grouped_neg[user].append(idx)

    selected_pos: List[int] = []
    selected_neg: List[int] = []
    all_users = set(user_ids)
    valid_users = set(grouped_pos) & set(grouped_neg)
    skipped_users = len(all_users - valid_users)
    user_order = sorted(valid_users)
    total_candidate_count = 0

    for user_index, user in enumerate(user_order):
        pos_ids = np.asarray(grouped_pos[user], dtype=np.int64)
        neg_ids = np.asarray(grouped_neg[user], dtype=np.int64)

        if pos_ids.size == 0 or neg_ids.size == 0:
            skipped_users += 1
            continue

        user_rng = _user_rng(seed, epoch, user_index)
        perm_pos = user_rng.permutation(pos_ids)

        pair_limit = min(int(max_pairs_per_user), int(perm_pos.size))
        if pair_limit <= 0:
            skipped_users += 1
            continue

        for p_idx in perm_pos[:pair_limit]:
            p_idx_int = int(p_idx)

            candidate_pool_size = min(int(hard_negative_candidates), int(neg_ids.size))
            if candidate_pool_size <= 0:
                continue

            selected_neg_indices = user_rng.choice(neg_ids, size=candidate_pool_size, replace=False)
            candidate_scores = model_predictions[selected_neg_indices]

            if not np.all(np.isfinite(candidate_scores)):
                raise ValueError(f"Non-finite mining scores at user {user!r}, positive row {p_idx_int}")

            max_score = np.max(candidate_scores)
            max_positions = np.flatnonzero(candidate_scores == max_score)
            chosen_position = int(
                max_positions[np.argmin(selected_neg_indices[max_positions])]
            )

            n_idx_int = int(selected_neg_indices[chosen_position])
            selected_pos.append(p_idx_int)
            selected_neg.append(n_idx_int)
            total_candidate_count += int(candidate_pool_size)

    return (
        np.asarray(selected_pos, dtype=np.int64),
        np.asarray(selected_neg, dtype=np.int64),
        int(len(selected_pos)),
        int(total_candidate_count),
        int(skipped_users),
    )
