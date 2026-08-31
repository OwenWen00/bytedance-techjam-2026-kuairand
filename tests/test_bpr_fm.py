import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent.runner import RealRunner, resolve_strategy_module
from experiments.bpr_fm import BPRFM, main


class BPRFMTests(unittest.TestCase):
    def test_duplicate_feature_ids_are_accumulated_correctly(self):
        model = BPRFM(dim=8, k=2, lr=0.1, seed=0)
        X_pos = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int32)
        X_neg = np.array([[0, 2, 2], [1, 2, 2]], dtype=np.int32)
        bias_before = float(model.b)

        before = model.predict(np.vstack([X_pos, X_neg])).copy()
        model.step(X_pos, X_neg)
        after = model.predict(np.vstack([X_pos, X_neg])).copy()

        self.assertTrue(np.isfinite(before).all())
        self.assertTrue(np.isfinite(after).all())
        self.assertAlmostEqual(float(model.b), bias_before)
        self.assertGreater(float(after[0] - after[2]), float(before[0] - before[2]))

    def test_bpr_fm_training_increases_positive_negative_margin(self):
        model = BPRFM(dim=6, k=4, lr=0.2, seed=0)
        X_pos = np.array([[0, 1], [1, 2]], dtype=np.int32)
        X_neg = np.array([[2, 3], [3, 4]], dtype=np.int32)

        before = float(model.predict(X_pos[0:1])[0] - model.predict(X_neg[0:1])[0])
        model.step(X_pos, X_neg)
        after = float(model.predict(X_pos[0:1])[0] - model.predict(X_neg[0:1])[0])

        self.assertGreater(after, before)

    def test_validation_output_has_one_score_per_row_and_no_test_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            train_path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
            valid_path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
            train_path.write_text("date,user_id,video_id,author_id,tab,duration_ms,long_view\n20220408,u1,v1,a1,tab1,100,1\n20220408,u1,v2,a2,tab1,200,0\n", encoding="utf-8")
            valid_path.write_text("date,user_id,video_id,author_id,tab,duration_ms,long_view\n20220422,u1,v3,a3,tab2,150,1\n20220422,u1,v4,a4,tab2,180,0\n", encoding="utf-8")
            (data_dir / "video_features_basic_pure.csv").write_text("video_id,author_id\nv1,a1\nv2,a2\nv3,a3\nv4,a4\n", encoding="utf-8")

            result_file = Path(tmpdir) / "result.json"
            with patch("experiments.bpr_fm.load", return_value={"train": [(20220408, "u1", "v1", "a1", "tab1", 100.0, 1), (20220408, "u1", "v2", "a2", "tab1", 200.0, 0)], "valid": [(20220422, "u1", "v3", "a3", "tab2", 150.0, 1), (20220422, "u1", "v4", "a4", "tab2", 180.0, 0)], "test": [(20220429, "u1", "v5", "a5", "tab1", 120.0, 1)]}), patch("experiments.bpr_fm.encode", return_value=({"train": (np.array([[1, 2, 3, 4, 0], [1, 2, 3, 4, 1]], dtype=np.int32), np.array([1.0, 0.0], dtype=np.float32), ["u1", "u1"]), "valid": (np.array([[1, 2, 3, 4, 0], [1, 2, 3, 4, 1]], dtype=np.int32), np.array([1.0, 0.0], dtype=np.float32), ["u1", "u1"])}, 32)):
                with patch("sys.argv", ["bpr_fm.py", "--experiment-id", "E004_bpr_fm_seed0", "--seed", "0", "--k", "8", "--lr", "0.01", "--batch", "32", "--max-epochs", "5", "--patience", "2", "--max-pairs-per-user", "8", "--data-dir", str(data_dir), "--result-path", str(result_file)]):
                    main()

            payload = json.loads(result_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_id"], "E004_bpr_fm_seed0")
            self.assertEqual(payload["split"], "validation")
            self.assertEqual(len(payload["metrics"]), 2)
            self.assertNotIn("primary", payload)
            self.assertNotIn("test", json.dumps(payload).lower())

    def test_unknown_strategy_is_rejected(self):
        plan = {"experiment_id": "bad", "strategy": "not_allowed", "seed": 0, "model_config": {} }
        with self.assertRaises(ValueError):
            RealRunner().run(plan)

    def test_strategy_registry_maps_only_to_bpr_module(self):
        self.assertEqual(resolve_strategy_module("pairwise_bpr_fm"), "experiments.bpr_fm")
        self.assertEqual(resolve_strategy_module("fm_validation"), "experiments.fm_validation")
        with self.assertRaises(ValueError):
            resolve_strategy_module("shell_command")

    def test_agent_runner_does_not_execute_arbitrary_command_strings(self):
        plan = {"experiment_id": "E004_bpr_fm_seed0", "strategy": "pairwise_bpr_fm", "seed": 0, "model_config": {"k": 8, "lr": 1e-3, "batch": 32, "max_epochs": 2, "patience": 1, "max_pairs_per_user": 8}}
        with patch("agent.runner.sys.executable", "python"), patch("agent.runner.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = ""
            with tempfile.TemporaryDirectory() as tmpdir:
                result_path = Path(tmpdir) / "result.json"
                result_path.write_text(json.dumps({"experiment_id": "E004_bpr_fm_seed0", "split": "validation", "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}}), encoding="utf-8")
                with patch("agent.runner.RealRunner._result_path", return_value=result_path):
                    parsed = RealRunner(repo_root=Path(tmpdir), artifacts_dir=Path(tmpdir) / "artifacts").run(plan)
                    self.assertEqual(parsed["experiment_id"], "E004_bpr_fm_seed0")
                    self.assertFalse(";" in " ".join(mock_run.call_args[0][0]))


if __name__ == "__main__":
    unittest.main()
