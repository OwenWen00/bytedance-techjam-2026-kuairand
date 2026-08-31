import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.controller import DryRunController, _resolve_reference_primary
from agent.evaluator import evaluate_agent_metrics


class DryRunLoopTests(unittest.TestCase):
    def test_dry_run_loop_runs_exactly_three_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            summary = controller.run_dry_run_loop(max_iterations=3)

            self.assertEqual(summary["iterations_run"], 3)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected"], 2)

    def test_max_iterations_is_respected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            summary = controller.run_dry_run_loop(max_iterations=2)

            self.assertEqual(summary["iterations_run"], 2)
            self.assertLessEqual(summary["accepted"] + summary["rejected"], 2)

    def test_accept_and_reject_are_both_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            summary = controller.run_dry_run_loop(max_iterations=3)

            self.assertGreaterEqual(summary["accepted"], 1)
            self.assertGreaterEqual(summary["rejected"], 1)

    def test_rejected_experiments_do_not_replace_current_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            summary = controller.run_dry_run_loop(max_iterations=3)

            self.assertEqual(summary["best_experiment_id"], "dry_run_002")
            self.assertAlmostEqual(summary["best_primary"], 0.6450)

    def test_primary_is_calculated_correctly(self):
        metrics = evaluate_agent_metrics({"GAUC": 0.7, "nDCG@5": 0.5})
        self.assertAlmostEqual(metrics["primary"], 0.6)

    def test_malformed_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_agent_metrics({"GAUC": "bad", "nDCG@5": 0.5})

    def test_three_valid_jsonl_records_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            summary = controller.run_dry_run_loop(max_iterations=3)

            self.assertTrue(log_path.exists())
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(records), 3)
            self.assertEqual(summary["iterations_run"], 3)

    def test_every_record_is_marked_as_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            controller.run_dry_run_loop(max_iterations=3)

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for record in records:
                self.assertEqual(record["configuration"]["mode"], "dry_run")
                self.assertTrue(record["metrics"]["validation"]["dry_run"])
                self.assertEqual(record["status"], "completed")
                self.assertIn(record["decision"], {"ACCEPT", "REJECT"})

    def test_no_test_metrics_appear_in_plans_results_or_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            summary = controller.run_dry_run_loop(max_iterations=3)

            log_text = log_path.read_text(encoding="utf-8")
            self.assertNotIn("test", log_text.lower())
            self.assertNotIn("test", json.dumps(summary).lower())

    def test_generated_log_path_uses_tempfile_and_no_real_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "agent_log.jsonl"
            controller = DryRunController(log_path=log_path)
            summary = controller.run_dry_run_loop(max_iterations=3)

            self.assertTrue(Path(summary["log_path"]).exists())
            self.assertTrue(str(summary["log_path"]).startswith(str(tmpdir)))

    def test_parent_chain_is_exact_for_three_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            controller.run_dry_run_loop(max_iterations=3)

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([record["experiment_id"] for record in records], ["dry_run_001", "dry_run_002", "dry_run_003"])
            self.assertEqual([record["parent_id"] for record in records], ["E001_fm", "E001_fm", "dry_run_002"])

    def test_accepted_experiment_is_not_its_own_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "dry_run.jsonl"
            controller = DryRunController(log_path=log_path)
            controller.run_dry_run_loop(max_iterations=3)

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for record in records:
                self.assertNotEqual(record["experiment_id"], record["parent_id"])

    def test_missing_validation_primary_in_e001_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_config = Path(tmpdir) / "E001_fm.json"
            temp_config.write_text(json.dumps({"metrics": {"validation": {}}}), encoding="utf-8")

            with patch("agent.controller.E001_PATH", temp_config):
                with self.assertRaisesRegex(ValueError, "missing the 'primary' value"):
                    _resolve_reference_primary()

    def test_malformed_validation_primary_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_config = Path(tmpdir) / "E001_fm.json"
            temp_config.write_text(json.dumps({"metrics": {"validation": {"primary": "bad"}}}), encoding="utf-8")

            with patch("agent.controller.E001_PATH", temp_config):
                with self.assertRaisesRegex(ValueError, "not numeric"):
                    _resolve_reference_primary()

    def test_non_finite_validation_primary_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_config = Path(tmpdir) / "E001_fm.json"
            temp_config.write_text(json.dumps({"metrics": {"validation": {"primary": float("inf")}}}), encoding="utf-8")

            with patch("agent.controller.E001_PATH", temp_config):
                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    _resolve_reference_primary()


if __name__ == "__main__":
    unittest.main()
