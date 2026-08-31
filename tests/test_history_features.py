import unittest

import numpy as np

from experiments.history_features import _bucketize_rate, build_causal_history_features


class HistoryFeatureTests(unittest.TestCase):
    def test_causal_prefix_construction_for_train_rows(self):
        train_rows = [
            (20220408, "u1", "v1", "a1", "tab1", 100.0, 1),
            (20220408, "u1", "v2", "a1", "tab1", 200.0, 0),
            (20220408, "u2", "v3", "a2", "tab1", 150.0, 1),
        ]
        train_hist, _, _ = build_causal_history_features(train_rows, valid_rows=[])
        self.assertEqual(train_hist[0], _bucketize_rate(0.5))
        self.assertEqual(train_hist[1], _bucketize_rate(1.0))
        self.assertEqual(train_hist[2], _bucketize_rate(0.5))

    def test_current_row_label_cannot_influence_its_own_feature(self):
        train_rows = [
            (20220408, "u1", "v1", "a1", "tab1", 100.0, 1),
            (20220408, "u1", "v2", "a1", "tab1", 200.0, 0),
        ]
        train_hist, _, _ = build_causal_history_features(train_rows, valid_rows=[])
        self.assertEqual(train_hist[1], _bucketize_rate(1.0))

    def test_validation_features_use_train_statistics_only(self):
        train_rows = [
            (20220408, "u1", "v1", "a1", "tab1", 100.0, 1),
            (20220408, "u1", "v2", "a1", "tab1", 200.0, 0),
        ]
        valid_rows = [
            (20220422, "u1", "v3", "a1", "tab2", 150.0, 1),
            (20220422, "u3", "v4", "a3", "tab2", 130.0, 0),
        ]
        _, valid_hist, _ = build_causal_history_features(train_rows, valid_rows)
        self.assertEqual(valid_hist[0], _bucketize_rate(0.5))
        self.assertEqual(valid_hist[1], _bucketize_rate(0.5))

    def test_validation_labels_cannot_change_validation_features(self):
        train_rows = [
            (20220408, "u1", "v1", "a1", "tab1", 100.0, 1),
            (20220408, "u1", "v2", "a1", "tab1", 200.0, 0),
        ]
        valid_rows_a = [
            (20220422, "u1", "v3", "a1", "tab2", 150.0, 1),
        ]
        valid_rows_b = [
            (20220422, "u1", "v3", "a1", "tab2", 150.0, 0),
        ]
        _, valid_hist_a, _ = build_causal_history_features(train_rows, valid_rows_a)
        _, valid_hist_b, _ = build_causal_history_features(train_rows, valid_rows_b)
        self.assertTrue(np.array_equal(valid_hist_a, valid_hist_b))

    def test_deterministic_output_and_unseen_pair_fallback(self):
        train_rows = [
            (20220408, "u1", "v1", "a1", "tab1", 100.0, 1),
            (20220408, "u2", "v2", "a2", "tab1", 200.0, 0),
        ]
        valid_rows = [
            (20220422, "u3", "v3", "a3", "tab2", 150.0, 1),
        ]
        _, valid_hist, fallback = build_causal_history_features(train_rows, valid_rows)
        self.assertEqual(valid_hist[0], _bucketize_rate(fallback))
        self.assertTrue(np.all(np.isfinite(valid_hist.astype(np.float64))))

    def test_finite_values(self):
        train_rows = [
            (20220408, "u1", "v1", "a1", "tab1", 100.0, 1),
            (20220408, "u1", "v2", "a1", "tab1", 200.0, 0),
        ]
        train_hist, valid_hist, _ = build_causal_history_features(train_rows, [
            (20220422, "u1", "v3", "a1", "tab2", 150.0, 1),
        ])
        self.assertTrue(np.all(np.isfinite(train_hist.astype(np.float64))))
        self.assertTrue(np.all(np.isfinite(valid_hist.astype(np.float64))))


if __name__ == "__main__":
    unittest.main()
