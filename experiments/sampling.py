from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def _user_rng(seed: int, epoch: int, user_index: int) -> np.random.Generator:
    state = (
        np.uint64(seed)
        ^ (np.uint64(epoch + 1) * np.uint64(131071))
        ^ (np.uint64(user_index + 1) * np.uint64(65537))
    )
    return np.random.default_rng(state)


def sample_same_user_pairs(
    user_ids: List[str],
    labels: np.ndarray,
    seed: int = 0,
    epoch: int = 0,
    max_pairs_per_user: int = 128,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if labels is None:
        raise ValueError("Labels are required to sample BPR pairs.")
    if len(user_ids) != len(labels):
        raise ValueError("user_ids and labels must have the same length.")
    if max_pairs_per_user <= 0:
        raise ValueError("max_pairs_per_user must be positive.")

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

    for user_index, user in enumerate(user_order):
        pos_ids = np.asarray(grouped_pos[user], dtype=np.int64)
        neg_ids = np.asarray(grouped_neg[user], dtype=np.int64)
        if pos_ids.size == 0 or neg_ids.size == 0:
            skipped_users += 1
            continue

        user_rng = _user_rng(seed, epoch, user_index)
        perm_pos = user_rng.permutation(pos_ids)
        perm_neg = user_rng.permutation(neg_ids)

        product_size = int(perm_pos.size * perm_neg.size)
        pair_limit = min(int(max_pairs_per_user), product_size)
        if pair_limit <= 0:
            skipped_users += 1
            continue

        pair_indices: List[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        if product_size <= max_pairs_per_user * 8:
            for p_idx in perm_pos:
                for n_idx in perm_neg:
                    pair = (int(p_idx), int(n_idx))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    pair_indices.append(pair)
                    if len(pair_indices) >= pair_limit:
                        break
                if len(pair_indices) >= pair_limit:
                    break
        else:
            step_p = max(1, perm_pos.size // max(1, int(np.ceil(np.sqrt(pair_limit)))))
            step_n = max(1, perm_neg.size // max(1, int(np.ceil(np.sqrt(pair_limit)))))
            for offset in range(pair_limit * 4):
                p_idx = (offset * step_p + int(user_rng.integers(0, max(1, step_p)))) % perm_pos.size
                n_idx = (offset * step_n + int(user_rng.integers(0, max(1, step_n)))) % perm_neg.size
                pair = (int(perm_pos[p_idx]), int(perm_neg[n_idx]))
                if pair in seen:
                    continue
                seen.add(pair)
                pair_indices.append(pair)
                if len(pair_indices) >= pair_limit:
                    break

        if not pair_indices:
            skipped_users += 1
            continue

        selected_pos.extend(int(p) for p, _ in pair_indices)
        selected_neg.extend(int(n) for _, n in pair_indices)

    return (
        np.asarray(selected_pos, dtype=np.int64),
        np.asarray(selected_neg, dtype=np.int64),
        int(len(selected_pos)),
        int(skipped_users),
    )
