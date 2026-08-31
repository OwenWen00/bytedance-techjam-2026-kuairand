import unittest
from unittest.mock import patch

import numpy as np

from experiments.hard_negative_sampling import sample_hard_negative_pairs


class HardNegativeSamplingTests(unittest.TestCase):
    def test_same_user_pairing(self):
        """Ensure all pairs are from the same user."""
        user_ids = ["u1", "u1", "u1", "u2", "u2", "u2"]
        labels = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
        scores = np.array([0.5, 0.3, 0.7, 0.6, 0.4, 0.8], dtype=np.float32)

        pos_idx, neg_idx, pair_count, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=2,
        )

        self.assertGreater(pair_count, 0)
        for p, n in zip(pos_idx, neg_idx):
            self.assertEqual(user_ids[p], user_ids[n])
            self.assertEqual(labels[p], 1)
            self.assertEqual(labels[n], 0)

    def test_correct_positive_negative_labels(self):
        """Positives must be 1, negatives must be 0."""
        user_ids = ["u1", "u1", "u1"]
        labels = np.array([1, 0, 0], dtype=np.float32)
        scores = np.array([0.5, 0.3, 0.7], dtype=np.float32)

        pos_idx, neg_idx, pair_count, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=2,
        )

        for p in pos_idx:
            self.assertEqual(int(labels[p]), 1)
        for n in neg_idx:
            self.assertEqual(int(labels[n]), 0)

    def test_highest_scoring_candidate_is_selected(self):
        """The hard negative should be the highest-scoring negative."""
        user_ids = ["u1", "u1", "u1", "u1"]
        labels = np.array([1, 0, 0, 0], dtype=np.float32)
        scores = np.array([0.5, 0.2, 0.9, 0.4], dtype=np.float32)

        pos_idx, neg_idx, pair_count, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=42, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=3,
        )

        self.assertEqual(pair_count, 1)
        candidate_scores = scores[[1, 2, 3]]
        self.assertEqual(scores[neg_idx[0]], np.max(candidate_scores))

    def test_non_adjacent_tied_maxima_choose_smallest_train_row(self):
        class FixedRng:
            @staticmethod
            def permutation(values):
                return np.asarray(values)

            @staticmethod
            def choice(values, size, replace):
                return np.asarray([3, 2, 1], dtype=np.int64)

        user_ids = ["u1", "u1", "u1", "u1"]
        labels = np.array([1, 0, 0, 0], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.1, 0.9], dtype=np.float32)

        with patch("experiments.hard_negative_sampling._user_rng", return_value=FixedRng()):
            _, neg_idx, pair_count, _, _ = sample_hard_negative_pairs(
                user_ids,
                labels,
                scores,
                seed=0,
                epoch=0,
                max_pairs_per_user=1,
                hard_negative_candidates=3,
            )

        self.assertEqual(pair_count, 1)
        self.assertEqual(int(neg_idx[0]), 1)
        self.assertEqual(scores[neg_idx[0]], np.max(scores[[1, 2, 3]]))

    def test_lower_candidate_between_tied_maxima_is_never_selected(self):
        class FixedRng:
            @staticmethod
            def permutation(values):
                return np.asarray(values)

            @staticmethod
            def choice(values, size, replace):
                return np.asarray([1, 2, 3], dtype=np.int64)

        user_ids = ["u1", "u1", "u1", "u1"]
        labels = np.array([1, 0, 0, 0], dtype=np.float32)
        scores = np.array([0.5, 0.9, 0.1, 0.9], dtype=np.float32)

        with patch("experiments.hard_negative_sampling._user_rng", return_value=FixedRng()):
            _, neg_idx, _, _, _ = sample_hard_negative_pairs(
                user_ids,
                labels,
                scores,
                seed=0,
                epoch=0,
                max_pairs_per_user=1,
                hard_negative_candidates=3,
            )

        self.assertNotEqual(int(neg_idx[0]), 2)
        self.assertEqual(scores[neg_idx[0]], np.max(scores[[1, 2, 3]]))

    def test_deterministic_same_seed_and_epoch(self):
        """Same seed and epoch must produce identical pairs."""
        user_ids = ["u1", "u1", "u1", "u2", "u2", "u2"]
        labels = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
        scores = np.array([0.5, 0.3, 0.7, 0.6, 0.4, 0.8], dtype=np.float32)

        pos1, neg1, count1, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=2,
        )
        pos2, neg2, count2, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=2,
        )

        self.assertEqual(count1, count2)
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(neg1, neg2)

    def test_deterministic_epoch_variation(self):
        """Different epochs with same seed should produce different pairs."""
        user_ids = ["u1", "u1", "u1", "u1", "u1", "u1"]
        labels = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
        scores = np.array([0.5, 0.6, 0.3, 0.7, 0.4, 0.8], dtype=np.float32)

        pos1, neg1, count1, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=2,
        )
        pos2, neg2, count2, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=1,
            max_pairs_per_user=10, hard_negative_candidates=2,
        )

        self.assertEqual(count1, count2)
        is_same = np.array_equal(pos1, pos2) and np.array_equal(neg1, neg2)
        self.assertFalse(is_same, "Different epochs should produce different pairs")

    def test_bounded_candidate_pool(self):
        """Candidate pool size must not exceed hard_negative_candidates."""
        user_ids = ["u1"] * 10
        labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        scores = np.arange(10, dtype=np.float32) * 0.1

        candidate_limit = 3
        pos_idx, neg_idx, pair_count, candidate_count, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=candidate_limit,
        )

        if pair_count > 0:
            avg_candidates = candidate_count / pair_count
            self.assertLessEqual(avg_candidates, candidate_limit)

    def test_no_cartesian_materialization(self):
        """The function should not create a full positive-negative Cartesian product."""
        user_ids = ["u1"] * 102
        labels = np.array([1] * 51 + [0] * 51, dtype=np.float32)
        scores = np.random.RandomState(0).uniform(0, 1, 102).astype(np.float32)

        pos_idx, neg_idx, pair_count, _, _ = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=5,
        )

        max_possible = 51 * 51
        self.assertLess(pair_count, max_possible, "Pair count should be much less than Cartesian product")
        self.assertLessEqual(pair_count, 10, "Pair count should respect max_pairs_per_user")

    def test_users_without_both_labels_are_skipped(self):
        """Users with only positives or only negatives must be skipped."""
        user_ids = ["u1", "u1", "u2", "u2", "u3", "u3"]
        labels = np.array([1, 0, 0, 0, 1, 1], dtype=np.float32)
        scores = np.array([0.5, 0.3, 0.4, 0.7, 0.6, 0.5], dtype=np.float32)

        pos_idx, neg_idx, pair_count, _, skipped = sample_hard_negative_pairs(
            user_ids, labels, scores, seed=0, epoch=0,
            max_pairs_per_user=10, hard_negative_candidates=2,
        )

        self.assertEqual(skipped, 2, "u2 (all negatives) and u3 (all positives) should be skipped")
        self.assertGreater(pair_count, 0, "u1 should produce pairs")

    def test_non_finite_mining_scores_are_rejected(self):
        """Non-finite scores should raise an error."""
        user_ids = ["u1", "u1", "u1"]
        labels = np.array([1, 0, 0], dtype=np.float32)
        for invalid_score in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid_score=invalid_score):
                scores = np.array([0.5, invalid_score, 0.7], dtype=np.float32)
                with self.assertRaises(ValueError):
                    sample_hard_negative_pairs(
                        user_ids,
                        labels,
                        scores,
                        seed=0,
                        epoch=0,
                        max_pairs_per_user=10,
                        hard_negative_candidates=2,
                    )


if __name__ == "__main__":
    unittest.main()
