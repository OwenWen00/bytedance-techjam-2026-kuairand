import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent.controller import RealController
from agent.runner import RealRunner
from experiments.bpr_fm import BPRFM
from experiments.fm_validation import FM, load_checkpoint, save_checkpoint


class E005RequirementsTests(unittest.TestCase):
    def test_real_runner_generates_and_confirms_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            artifact_root = repo_root / "artifacts"
            checkpoint_root = artifact_root / "checkpoints"
            runner = RealRunner(repo_root=repo_root, artifacts_dir=artifact_root, checkpoints_dir=checkpoint_root)
            plan = {
                "experiment_id": "E005_path_check",
                "seed": 0,
                "strategy": "fm_validation",
                "model_config": {"k": 4, "lr": 0.001, "batch": 16, "max_epochs": 2, "patience": 1},
            }

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                checkpoint_path = Path(cmd[cmd.index("--checkpoint-path") + 1])
                result = {
                    "experiment_id": "E005_path_check",
                    "split": "validation",
                    "metrics": {"GAUC": 0.7, "nDCG@5": 0.6},
                    "metadata": {"checkpoint_path": str(checkpoint_path)},
                }
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                model = FM(dim=8, k=4, lr=0.001, seed=0)
                save_checkpoint(model, checkpoint_path)
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                parsed = runner.run(plan)

            self.assertTrue(parsed["result_path"].endswith(".json"))
            self.assertTrue(parsed["checkpoint_path"].endswith(".npz"))
            self.assertTrue(Path(parsed["checkpoint_path"]).exists())

    def test_real_runner_rejects_checkpoint_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            artifact_root = repo_root / "artifacts"
            runner = RealRunner(repo_root=repo_root, artifacts_dir=artifact_root, checkpoints_dir=artifact_root / "checkpoints")
            plan = {
                "experiment_id": "E005_escape",
                "seed": 0,
                "strategy": "fm_validation",
                "model_config": {"k": 4, "lr": 0.001, "batch": 16, "max_epochs": 2, "patience": 1},
            }
            escape_path = repo_root / "evil.npz"
            escape_path.write_bytes(np.zeros(3, dtype=np.float32).tobytes())

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                result = {
                    "experiment_id": "E005_escape",
                    "split": "validation",
                    "metrics": {"GAUC": 0.7, "nDCG@5": 0.6},
                    "metadata": {"checkpoint_path": str(escape_path)},
                }
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                with self.assertRaisesRegex(ValueError, "Checkpoint path escapes"):
                    runner.run(plan)

    def test_fm_checkpoint_roundtrip_reproduces_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fm_state.npz"
            model = FM(dim=9, k=4, lr=0.001, seed=0)
            X = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 0]], dtype=np.int32)
            scores_before = model.predict(X)
            save_checkpoint(model, path)
            reloaded = load_checkpoint(path, FM, dim=9, k=4, lr=0.001, seed=0)
            scores_after = reloaded.predict(X)
            allclose = bool(np.allclose(scores_before, scores_after, rtol=0, atol=1e-6))
            print(f"FM checkpoint scores_before vs scores_after np.allclose={allclose}")
            self.assertTrue(allclose)

    def test_bpr_checkpoint_roundtrip_reproduces_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bpr_state.npz"
            model = BPRFM(dim=9, k=4, lr=0.01, seed=0)
            X = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 0]], dtype=np.int32)
            scores_before = model.predict(X)
            save_checkpoint(model, path)
            reloaded = load_checkpoint(path, BPRFM, dim=9, k=4, lr=0.01, seed=0)
            scores_after = reloaded.predict(X)
            allclose = bool(np.allclose(scores_before, scores_after, rtol=0, atol=1e-6))
            print(f"BPR checkpoint scores_before vs scores_after np.allclose={allclose}")
            self.assertTrue(allclose)

    def test_controller_retries_once_and_uses_final_retry_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "retry.jsonl"
            controller = RealController(log_path=log_path)
            plan = {"experiment_id": "E005_retry", "hypothesis": "Retry once on timeout.", "strategy": "fm_validation", "seed": 0, "model_config": {"k": 4, "lr": 0.001, "batch": 16, "max_epochs": 2, "patience": 1}}
            controller.planner = type("P", (), {"next_plan": lambda self, max_iterations=None: plan})()
            calls = {"count": 0}

            def fake_run(plan_arg, timeout=None):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise TimeoutError("synthetic timeout")
                return {"GAUC": 0.80, "nDCG@5": 0.70, "experiment_id": "E005_retry", "checkpoint_path": "ignored.npz"}

            controller.runner.run = fake_run
            controller.current_best_primary = 0.55
            controller.current_best_experiment_id = "E001_fm"
            controller.best_checkpoint = None
            with patch.object(controller, "_checkpoint_best_state", return_value={"experiment_id": "E001_fm", "primary": 0.55}):
                summary = controller.run_real_loop(max_iterations=1, timeout_sec=1.0)

            self.assertEqual(calls["count"], 2)
            self.assertEqual(summary["attempts_run"], 2)
            self.assertEqual(summary["recovery_attempts"], 1)
            self.assertEqual(summary["best_experiment_id"], "E005_retry")

    def test_failure_preserves_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "fail.jsonl"
            controller = RealController(log_path=log_path)
            controller.current_best_primary = 0.61
            controller.current_best_experiment_id = "E001_fm"
            controller.best_checkpoint = "/tmp/valid.npz"
            plan = {"experiment_id": "E005_fail", "hypothesis": "Failure should not replace best.", "strategy": "fm_validation", "seed": 0, "model_config": {"k": 4, "lr": 0.001, "batch": 16, "max_epochs": 2, "patience": 1}}
            controller.planner = type("P", (), {"next_plan": lambda self, max_iterations=None: plan})()

            def fake_run(plan_arg, timeout=None):
                raise RuntimeError("synthetic failure")

            controller.runner.run = fake_run
            with self.assertRaises(RuntimeError):
                controller.run_real_loop(max_iterations=1)
            self.assertEqual(controller.best_checkpoint, "/tmp/valid.npz")
            self.assertEqual(controller.current_best_primary, 0.61)

    def test_tiny_accept_updates_best_but_increments_no_significant_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = RealController(log_path=Path(tmpdir) / "tiny.jsonl")
            controller.current_best_primary = 0.6000
            controller.current_best_experiment_id = "E001_fm"
            controller.best_checkpoint = None
            controller.no_significant_improvement = 0
            controller._record_result(candidate_primary=0.6010, experiment_id="tiny_accept", plan={"experiment_id": "tiny_accept"}, accepted=True)
            self.assertEqual(controller.current_best_primary, 0.6010)
            self.assertEqual(controller.no_significant_improvement, 1)

    def test_improvement_gt_0_002_resets_counter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = RealController(log_path=Path(tmpdir) / "reset.jsonl")
            controller.current_best_primary = 0.6000
            controller.no_significant_improvement = 2
            controller._record_result(candidate_primary=0.6035, experiment_id="reset", plan={"experiment_id": "reset"}, accepted=True)
            self.assertEqual(controller.no_significant_improvement, 0)

    def test_three_no_significant_improvement_plans_trigger_convergence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = RealController(log_path=Path(tmpdir) / "converge.jsonl")
            controller.current_best_primary = 0.6000
            controller.best_checkpoint = None
            controller.no_significant_improvement = 0
            controller.planner = type("P", (), {"next_plan": lambda self, max_iterations=None: {"experiment_id": "conv_1", "hypothesis": "h", "strategy": "recovery_demo", "model_config": {}, "seed": 0}})()
            for idx in range(3):
                controller._record_result(candidate_primary=0.6005 + idx * 0.0001, experiment_id=f"conv_{idx}", plan={"experiment_id": f"conv_{idx}"}, accepted=True)
            self.assertTrue(controller.converged)
            self.assertEqual(controller.stop_reason, "convergence_no_improvement_3x")

    def test_convergence_config_uses_three_no_improvement_plans(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "E005_convergence_demo.json"
        controller = RealController(log_path=Path(tempfile.gettempdir()) / "e005_convergence_regression.jsonl", config_path=config_path)
        summary = controller.run_real_loop(max_iterations=3)
        self.assertEqual(summary["iterations_run"], 3)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected"], 2)
        self.assertEqual(summary["best_experiment_id"], "converge_001")
        self.assertAlmostEqual(summary["best_primary"], 0.6020, places=4)
        self.assertTrue(summary["converged"])
        self.assertEqual(summary["stop_reason"], "convergence_no_improvement_3x")

    def test_missing_checkpoint_cannot_replace_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = RealController(log_path=Path(tmpdir) / "missing.jsonl")
            controller.current_best_primary = 0.75
            controller.best_checkpoint = "/tmp/keep.npz"
            controller._record_result(candidate_primary=0.76, experiment_id="missing", plan={"experiment_id": "missing"}, accepted=True, checkpoint_path=Path(tmpdir) / "does_not_exist.npz")
            self.assertEqual(controller.best_checkpoint, "/tmp/keep.npz")

    def test_no_test_leakage_is_rejected(self):
        payload = {"experiment_id": "demo", "split": "validation", "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}, "metadata": {"checkpoint_path": "./artifacts/checkpoints/fake.npz"}, "test_results": {"GAUC": 0.9}}
        with self.assertRaises(ValueError):
            RealRunner._reject_test_leak(payload)

    def test_recovery_demo_command_contains_only_recovery_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = RealRunner(repo_root=repo_root, artifacts_dir=repo_root / "artifacts", checkpoints_dir=repo_root / "artifacts" / "checkpoints")
            plan = {"experiment_id": "recovery_cli", "seed": 0, "strategy": "recovery_demo", "model_config": {"failure_mode": "timeout", "mode": "validation", "sleep_sec": 0.0}}
            captured = {}

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                captured["cmd"] = cmd
                checkpoint_path = Path(cmd[cmd.index("--checkpoint-path") + 1])
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                payload = {"experiment_id": "recovery_cli", "split": "validation", "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}, "metadata": {"checkpoint_path": str(checkpoint_path)}}
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(checkpoint_path, V=np.zeros((2, 2), dtype=np.float32), W=np.zeros(2, dtype=np.float32), b=np.zeros(1, dtype=np.float32))
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                parsed = runner.run(plan)

            self.assertIn("--failure-mode", captured["cmd"])
            self.assertIn("--mode", captured["cmd"])
            self.assertIn("--sleep-sec", captured["cmd"])
            self.assertNotIn("--k", captured["cmd"])
            self.assertNotIn("--lr", captured["cmd"])
            self.assertNotIn("--batch", captured["cmd"])
            self.assertNotIn("--max-epochs", captured["cmd"])
            self.assertNotIn("--patience", captured["cmd"])
            self.assertEqual(parsed["experiment_id"], "recovery_cli")

    def test_recovery_demo_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration keys for strategy 'recovery_demo'"):
            RealRunner(repo_root=Path("."), artifacts_dir=Path("artifacts")).run({
                "experiment_id": "bad_recovery",
                "seed": 0,
                "strategy": "recovery_demo",
                "model_config": {"failure_mode": "none", "foo": "bar"},
            })

    def test_recovery_demo_rejects_primary_and_controller_recomputes_primary_from_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = RealRunner(repo_root=Path(tmpdir), artifacts_dir=Path(tmpdir) / "artifacts", checkpoints_dir=Path(tmpdir) / "artifacts" / "checkpoints")
            with self.assertRaisesRegex(ValueError, "Unknown configuration keys for strategy 'recovery_demo'"):
                runner.run({
                    "experiment_id": "bad_primary",
                    "seed": 0,
                    "strategy": "recovery_demo",
                    "model_config": {"failure_mode": "none", "mode": "validation", "sleep_sec": 0.0, "primary": 0.7000},
                })

            result = {"experiment_id": "recompute_primary", "split": "validation", "metrics": {"GAUC": 0.6020, "nDCG@5": 0.6020}}
            self.assertNotIn("primary", result)
            self.assertNotIn("primary", result["metrics"])

            controller = RealController(log_path=Path(tmpdir) / "recompute.jsonl")
            controller.current_best_primary = 0.6016
            controller.current_best_experiment_id = "E001_fm"
            candidate_primary = 0.5 * (result["metrics"]["GAUC"] + result["metrics"]["nDCG@5"])
            controller._record_result(candidate_primary, "recompute_primary", {"experiment_id": "recompute_primary"}, True)
            self.assertAlmostEqual(controller.current_best_primary, 0.6020, places=4)

    def test_fm_and_bpr_commands_keep_required_model_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            runner = RealRunner(repo_root=repo_root, artifacts_dir=repo_root / "artifacts", checkpoints_dir=repo_root / "artifacts" / "checkpoints")
            capture = []

            def fake_run(cmd, cwd, shell, capture_output, text, check):
                capture.append(cmd)
                result_path = Path(cmd[cmd.index("--result-path") + 1])
                checkpoint_path = Path(cmd[cmd.index("--checkpoint-path") + 1])
                payload = {"experiment_id": cmd[cmd.index("--experiment-id") + 1], "split": "validation", "metrics": {"GAUC": 0.7, "nDCG@5": 0.6}, "metadata": {"checkpoint_path": str(checkpoint_path)}}
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(checkpoint_path, V=np.zeros((2, 2), dtype=np.float32), W=np.zeros(2, dtype=np.float32), b=np.zeros(1, dtype=np.float32))
                return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

            with patch("agent.runner.subprocess.run", side_effect=fake_run):
                runner.run({"experiment_id": "fm_cmdcheck", "seed": 0, "strategy": "fm_validation", "model_config": {"k": 8, "lr": 0.01, "batch": 16, "max_epochs": 3, "patience": 2}})
                runner.run({"experiment_id": "bpr_cmdcheck", "seed": 0, "strategy": "pairwise_bpr_fm", "model_config": {"k": 8, "lr": 0.01, "batch": 16, "max_epochs": 3, "patience": 2, "max_pairs_per_user": 8}})

            fm_cmd = capture[0]
            bpr_cmd = capture[1]
            self.assertIn("--k", fm_cmd)
            self.assertIn("--lr", fm_cmd)
            self.assertIn("--batch", fm_cmd)
            self.assertIn("--max-epochs", fm_cmd)
            self.assertIn("--patience", fm_cmd)
            self.assertIn("--max-pairs-per-user", bpr_cmd)

    def test_arbitrary_configuration_cannot_become_cli_arguments(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration keys"):
            RealRunner(repo_root=Path("."), artifacts_dir=Path("artifacts")).run({
                "experiment_id": "bad_cfg",
                "seed": 0,
                "strategy": "fm_validation",
                "model_config": {"k": 4, "lr": 0.001, "batch": 8, "max_epochs": 2, "patience": 1, "unexpected": "value"},
            })


if __name__ == "__main__":
    unittest.main()
