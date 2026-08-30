import unittest

import numpy as np

from experiments.sampling import sample_same_user_pairs


class BPRSamplingTests(unittest.TestCase):
    def test_same_user_pairing_and_train_only_sampling(self):
        user_ids = ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3"]
        labels = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=np.float32)

        pos_idx, neg_idx, pair_count, skipped_users = sample_same_user_pairs(
            user_ids,
            labels,
            seed=7,
            epoch=0,
            max_pairs_per_user=4,
        )

        self.assertGreater(pair_count, 0)
        self.assertEqual(skipped_users, 0)
        for p_idx, n_idx in zip(pos_idx, neg_idx):
            self.assertEqual(user_ids[p_idx], user_ids[n_idx])
            self.assertEqual(labels[p_idx], 1)
            self.assertEqual(labels[n_idx], 0)

    def test_users_without_both_classes_are_skipped(self):
        user_ids = ["u1", "u1", "u2", "u2", "u3"]
        labels = np.array([1, 0, 1, 1, 0], dtype=np.float32)

        pos_idx, neg_idx, pair_count, skipped_users = sample_same_user_pairs(
            user_ids,
            labels,
            seed=1,
            epoch=0,
            max_pairs_per_user=8,
        )

        self.assertEqual(pair_count, 1)
        self.assertEqual(skipped_users, 2)

    def test_same_seed_and_epoch_are_deterministic(self):
        user_ids = ["u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2"]
        labels = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=np.float32)

        a = sample_same_user_pairs(user_ids, labels, seed=11, epoch=2, max_pairs_per_user=8)
        b = sample_same_user_pairs(user_ids, labels, seed=11, epoch=2, max_pairs_per_user=8)
        self.assertEqual(a[0].tolist(), b[0].tolist())
        self.assertEqual(a[1].tolist(), b[1].tolist())

    def test_different_epochs_can_produce_different_pair_samples(self):
        user_ids = ["u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2", "u2"]
        labels = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)

        epoch0 = sample_same_user_pairs(user_ids, labels, seed=3, epoch=0, max_pairs_per_user=8)
        epoch1 = sample_same_user_pairs(user_ids, labels, seed=3, epoch=1, max_pairs_per_user=8)
        self.assertNotEqual(epoch0[0].tolist(), epoch1[0].tolist())


if __name__ == "__main__":
    unittest.main()
