import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.runner import RealRunner, _reject_test_leak


class RealRunnerTests(unittest.TestCase):
    def test_real_runner_uses_sys_executable_argument_list_and_repo_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            plan = {
                "experiment_id": "E003_fm_repro",
                "seed": 0,
                "model_config": {"k": 16, "lr": 0.001, "batch": 8192, "max_epochs": 40, "patience": 4},
            }
            runner = RealRunner(repo_root=repo_root, artifacts_dir=repo_root / "artifacts")

            captured = {}

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                captured["cmd"] = cmd
                captured["cwd"] = cwd
                captured["shell"] = shell
                captured["capture_output"] = capture_output
                captured["text"] = text
                captured["check"] = check
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                payload = {
                    "experiment_id": "E003_fm_repro",
                    "split": "validation",
                    "metrics": {"GAUC": 0.7, "nDCG@5": 0.6},
                    "metadata": {"seed": 0, "model": "FM", "configuration": {"k": 16, "lr": 0.001, "batch": 8192, "max_epochs": 40, "patience": 4}},
                }
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                parsed = runner.run(plan)

            self.assertEqual(captured["cmd"][0], sys.executable)
            self.assertEqual(captured["cmd"][1], "-m")
            self.assertEqual(captured["cmd"][2], "experiments.fm_validation")
            self.assertFalse(captured["shell"])
            self.assertEqual(captured["cwd"], str(repo_root))
            self.assertTrue(parsed["GAUC"] > 0)
            self.assertEqual(parsed["split"], "validation")

    def test_real_runner_rejects_mismatched_experiment_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = RealRunner(repo_root=repo_root, artifacts_dir=repo_root / "artifacts")
            plan = {"experiment_id": "E003_fm_repro", "seed": 0, "model_config": {"k": 16, "lr": 0.001, "batch": 8192, "max_epochs": 40, "patience": 4}}
            result_path = runner._result_path()

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                payload = {"experiment_id": "other", "split": "validation", "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}, "metadata": {"seed": 0, "model": "FM"}}
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(ValueError, "Experiment id mismatch"):
                    runner.run(plan)

    def test_real_runner_rejects_wrong_split_and_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = RealRunner(repo_root=repo_root, artifacts_dir=repo_root / "artifacts")
            plan = {"experiment_id": "E003_fm_repro", "seed": 0, "model_config": {"k": 16, "lr": 0.001, "batch": 8192, "max_epochs": 40, "patience": 4}}

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                payload = {"experiment_id": "E003_fm_repro", "split": "test", "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}, "metadata": {"seed": 0, "model": "FM"}}
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                with self.assertRaises(ValueError):
                    runner.run(plan)

    def test_real_runner_rejects_non_finite_and_boolean_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = RealRunner(repo_root=repo_root, artifacts_dir=repo_root / "artifacts")
            plan = {"experiment_id": "E003_fm_repro", "seed": 0, "model_config": {"k": 16, "lr": 0.001, "batch": 8192, "max_epochs": 40, "patience": 4}}

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                payload = {"experiment_id": "E003_fm_repro", "split": "validation", "metrics": {"GAUC": True, "nDCG@5": float("inf")}, "metadata": {"seed": 0, "model": "FM"}}
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                with self.assertRaises(ValueError):
                    runner.run(plan)

    def test_real_runner_rejects_non_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = RealRunner(repo_root=repo_root, artifacts_dir=repo_root / "artifacts")
            plan = {"experiment_id": "E003_fm_repro", "seed": 0, "model_config": {"k": 16, "lr": 0.001, "batch": 8192, "max_epochs": 40, "patience": 4}}

            with patch("agent.runner.subprocess.run", return_value=subprocess.CompletedProcess(["python"], 2, stdout="", stderr="boom")):
                with self.assertRaisesRegex(RuntimeError, "exit=2"):
                    runner.run(plan)

    def test_recursive_test_field_rejection(self):
        payload = {"experiment_id": "E003_fm_repro", "split": "validation", "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}, "test_results": {"GAUC": 0.9}, "metadata": {"seed": 0, "model": "FM"}}
        with self.assertRaises(ValueError):
            _reject_test_leak(payload)


if __name__ == "__main__":
    unittest.main()
