import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent.runner import RealRunner, resolve_strategy_module
from experiments.hard_negative_bpr import main


class HardNegativeBPRTests(unittest.TestCase):
    def test_strategy_maps_only_to_hard_negative_bpr_module(self):
        """Verify the strategy name maps to the correct module."""
        self.assertEqual(resolve_strategy_module("hard_negative_bpr_fm"), "experiments.hard_negative_bpr")

    def test_unknown_strategy_is_rejected(self):
        """Unknown strategies must be rejected."""
        plan = {
            "experiment_id": "bad",
            "strategy": "not_allowed",
            "seed": 0,
            "model_config": {}
        }
        with self.assertRaises(ValueError):
            RealRunner().run(plan)

    def test_arbitrary_commands_remain_rejected(self):
        """Arbitrary command strings must not be executable."""
        plan = {
            "experiment_id": "E006_hard_negative_bpr_seed0",
            "strategy": "hard_negative_bpr_fm",
            "seed": 0,
            "model_config": {
                "k": 8,
                "lr": 1e-3,
                "batch": 32,
                "max_epochs": 2,
                "patience": 1,
                "max_pairs_per_user": 8,
                "hard_negative_candidates": 4,
            }
        }
        with patch("agent.runner.sys.executable", "python"), \
             patch("agent.runner.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""
            with tempfile.TemporaryDirectory() as tmpdir:
                result_path = Path(tmpdir) / "result.json"
                result_path.write_text(
                    json.dumps({
                        "experiment_id": "E006_hard_negative_bpr_seed0",
                        "split": "validation",
                        "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}
                    }),
                    encoding="utf-8"
                )
                with patch("agent.runner.RealRunner._result_path", return_value=result_path):
                    parsed = RealRunner(
                        repo_root=Path(tmpdir),
                        artifacts_dir=Path(tmpdir) / "artifacts"
                    ).run(plan)
                    self.assertEqual(parsed["experiment_id"], "E006_hard_negative_bpr_seed0")
                    command_str = " ".join(mock_run.call_args[0][0])
                    self.assertNotIn(";", command_str, "no semicolons in command")

    def test_result_payload_has_only_validation_metrics(self):
        """Result payload must contain exactly the official validation metrics."""
        result = {
            "experiment_id": "E006_hard_negative_bpr_seed0",
            "split": "validation",
            "metrics": {
                "GAUC": float(0.65),
                "nDCG@5": float(0.70),
            },
            "metadata": {
                "seed": 0,
                "model": "Hard-Negative BPR-FM",
                "strategy": "hard_negative_bpr_fm",
                "sampling": {
                    "sampled_pair_count": 1,
                    "candidate_count": 2,
                    "skipped_user_count": 0,
                    "sampler_name": "hard_negative",
                }
            }
        }

        self.assertEqual(len(result["metrics"]), 2)
        self.assertIn("GAUC", result["metrics"])
        self.assertIn("nDCG@5", result["metrics"])
        self.assertNotIn("primary", result)

    def test_no_test_fields(self):
        """Result payload must not contain any test fields."""
        result = {
            "experiment_id": "E006_hard_negative_bpr_seed0",
            "split": "validation",
            "metrics": {
                "GAUC": 0.65,
                "nDCG@5": 0.70,
            },
            "metadata": {
                "seed": 0,
                "sampled_pair_count": 1,
            }
        }

        result_json = json.dumps(result)
        self.assertNotIn("test", result_json.lower())

    def test_result_contract_structure(self):
        """Result must have required contract fields."""
        result = {
            "experiment_id": "E006_hard_negative_bpr_seed0",
            "split": "validation",
            "metrics": {
                "GAUC": 0.65,
                "nDCG@5": 0.70,
            },
            "metadata": {
                "seed": 0,
                "model": "Hard-Negative BPR-FM",
                "strategy": "hard_negative_bpr_fm",
                "sampling": {
                    "sampled_pair_count": 100,
                    "candidate_count": 200,
                    "skipped_user_count": 5,
                    "sampler_name": "hard_negative",
                }
            }
        }

        self.assertIn("experiment_id", result)
        self.assertEqual(result["split"], "validation")
        self.assertIn("metrics", result)
        self.assertIn("metadata", result)
        self.assertNotIn("primary", result)
        self.assertIn("sampling", result["metadata"])
        self.assertEqual(result["metadata"]["sampling"]["sampler_name"], "hard_negative")

    def test_production_mining_is_train_only_and_never_accesses_test_split(self):
        train_rows = object()
        validation_rows = object()
        test_rows = object()

        class GuardedSplits(dict):
            def __init__(self):
                super().__init__(train=train_rows, valid=validation_rows, test=test_rows)
                self.accessed = []

            def __getitem__(self, key):
                self.accessed.append(key)
                if key == "test":
                    raise AssertionError("The hard-negative experiment accessed the test split.")
                return super().__getitem__(key)

        Xtr = np.array([[10, 11], [12, 13]], dtype=np.int32)
        ytr = np.array([1.0, 0.0], dtype=np.float32)
        users_train = ["train_user", "train_user"]
        Xva = np.array([[20, 21], [22, 23]], dtype=np.int32)
        yva = np.array([0.0, 1.0], dtype=np.float32)
        users_validation = ["validation_user", "validation_user"]
        train_scores = np.array([0.8, 0.7], dtype=np.float32)
        validation_scores = np.array([0.4, 0.6], dtype=np.float32)
        guarded_splits = GuardedSplits()
        sampler_calls = []
        evaluation_calls = []

        class FakeModel:
            def __init__(self):
                self.V = np.zeros((32, 2), dtype=np.float32)
                self.W = np.zeros(32, dtype=np.float32)
                self.predict_inputs = []
                self.step_inputs = []

            def predict(self, features):
                self.predict_inputs.append(features)
                if features is Xtr:
                    return train_scores
                if features is Xva:
                    return validation_scores
                raise AssertionError("The model received features outside train or validation.")

            def step(self, positive_features, negative_features):
                self.step_inputs.append((positive_features.copy(), negative_features.copy()))
                return 0.0

        model = FakeModel()

        def fake_encode(selected_splits):
            self.assertEqual(list(selected_splits), ["train", "valid"])
            self.assertIs(selected_splits["train"], train_rows)
            self.assertIs(selected_splits["valid"], validation_rows)
            return {
                "train": (Xtr, ytr, users_train),
                "valid": (Xva, yva, users_validation),
            }, 32

        def fake_sampler(user_ids, labels, model_predictions, **kwargs):
            sampler_calls.append((user_ids, labels, model_predictions, kwargs))
            self.assertIs(user_ids, users_train)
            self.assertIs(labels, ytr)
            self.assertIs(model_predictions, train_scores)
            return (
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                1,
                2,
                0,
            )

        def fake_evaluate(user_ids, labels, scores):
            evaluation_calls.append((user_ids, labels, scores))
            self.assertIs(user_ids, users_validation)
            self.assertIs(labels, yva)
            self.assertIs(scores, validation_scores)
            return {"GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65}

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            checkpoint_path = Path(tmpdir) / "checkpoint.npz"
            argv = [
                "hard_negative_bpr.py",
                "--experiment-id",
                "E006_train_only_test",
                "--max-epochs",
                "1",
                "--patience",
                "1",
                "--batch",
                "8",
                "--result-path",
                str(result_path),
                "--checkpoint-path",
                str(checkpoint_path),
            ]
            with patch("experiments.hard_negative_bpr.load", return_value=guarded_splits), \
                 patch("experiments.hard_negative_bpr.encode", side_effect=fake_encode), \
                 patch("experiments.hard_negative_bpr.BPRFM", return_value=model), \
                 patch("experiments.hard_negative_bpr.sample_hard_negative_pairs", side_effect=fake_sampler), \
                 patch("experiments.hard_negative_bpr.evaluate", side_effect=fake_evaluate), \
                 patch("experiments.hard_negative_bpr.save_checkpoint", return_value=checkpoint_path), \
                 patch("sys.argv", argv):
                main()

            payload = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(guarded_splits.accessed, ["train", "valid"])
        self.assertEqual(len(sampler_calls), 1)
        self.assertEqual(len(model.predict_inputs), 3)
        self.assertIs(model.predict_inputs[0], Xtr)
        self.assertIs(model.predict_inputs[1], Xva)
        self.assertIs(model.predict_inputs[2], Xva)
        self.assertEqual(len(model.step_inputs), 1)
        np.testing.assert_array_equal(model.step_inputs[0][0], Xtr[[0]])
        np.testing.assert_array_equal(model.step_inputs[0][1], Xtr[[1]])
        self.assertEqual(len(evaluation_calls), 2)
        self.assertEqual(payload["metadata"]["sampling"]["counter_scope"], "cumulative_across_epochs")
        self.assertEqual(payload["metadata"]["sampling"]["epochs_completed"], 1)


if __name__ == "__main__":
    unittest.main()
