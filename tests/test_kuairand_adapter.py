import json
import tempfile
import unittest
from pathlib import Path

from agent.kuairand import build
from agent.planner import JsonPlannerAdapter
from agent.runner import CommandResult
from agent.tools import RunContext
from models.trial_lab_core import validate_predictions


class FakeRunner:
    def __init__(self, root, variant):
        self.root = Path(root)
        self.variant = variant
        self.argv = None

    def run(self, argv, timeout_seconds, log_dir, env=None):
        del timeout_seconds, env
        self.argv = list(argv)
        output = self.root / argv[argv.index("--output-dir") + 1]
        output.mkdir(parents=True)
        modes = {
            "pointwise_fm": ("pointwise", "official", "random"),
            "pairwise_bpr": ("pairwise", "official", "random"),
            "hard_negative_bpr": ("pairwise", "official", "hard"),
            "history_pairwise": ("pairwise", "history", "random"),
        }
        training, encoder, negative = modes[self.variant]
        summary = {
            "status": "complete",
            "valid": {"GAUC": 0.65, "nDCG@5": 0.55, "primary": 0.60},
            "test": None,
            "config": {
                "training_mode": training,
                "encoder_mode": encoder,
                "negative_strategy": negative,
            },
        }
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        for name in (
            "config.json", "epochs.jsonl", "best_model.npz",
            "validation_predictions.csv",
        ):
            (output / name).write_text("fixture\n", encoding="utf-8")
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stdout = log_path / "stdout.log"
        stderr = log_path / "stderr.log"
        stdout.write_text("ok\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        return CommandResult(
            argv=list(argv), returncode=0, elapsed_seconds=1.25,
            stdout_summary="ok", stderr_summary="",
            stdout_path=str(stdout), stderr_path=str(stderr),
        )


class KuaiRandAdapterTests(unittest.TestCase):
    def test_registry_and_fallback_planner_follow_evidence_order(self):
        registry, planner = build("unused")
        self.assertEqual(
            (
                "run_hard_negative_bpr", "run_history_pairwise",
                "run_pairwise_bpr", "run_pointwise_fm",
            ),
            registry.names(),
        )
        baseline = planner.next_plan("run", 0, [])
        self.assertEqual("run_pointwise_fm", baseline.requested_tool)
        pairwise = planner.next_plan(
            "run", 1,
            [{"run_id": "run-E000", "status": "accepted",
              "single_primary_change": baseline.single_primary_change}],
        )
        self.assertEqual("run_pairwise_bpr", pairwise.requested_tool)
        self.assertEqual(1, len(pairwise.changes))
        self.assertIn("run/E001-active-variant.json", pairwise.changes[0].path)
        history = planner.next_plan(
            "run", 2,
            [
                {"run_id": "run-E000", "status": "accepted",
                 "single_primary_change": baseline.single_primary_change},
                {"run_id": "run-E001", "status": "rejected",
                 "single_primary_change": pairwise.single_primary_change},
            ],
        )
        self.assertEqual("run_history_pairwise", history.requested_tool)

    def test_registry_rejects_test_and_smoke_escape_hatches(self):
        registry, planner = build("unused")
        plan = planner.next_plan("run", 0, [])
        definition = registry.get(plan.requested_tool)
        with self.assertRaises(ValueError):
            definition.validate_params({**plan.params, "score_test": True})
        with self.assertRaises(ValueError):
            definition.validate_params({**plan.params, "smoke": True})

    def test_llm_plan_uses_the_same_controlled_diff_transform(self):
        registry, planner = build("unused")
        del registry
        baseline = planner.next_plan("demo", 0, [])
        history_record = {
            "run_id": "demo-E000", "status": "accepted",
            "single_primary_change": baseline.single_primary_change,
        }
        pairwise_template = next(
            candidate.plan for candidate in planner.candidates
            if candidate.plan.requested_tool == "run_pairwise_bpr"
        )
        raw = json.dumps({"action": "plan", "plan": pairwise_template.to_dict()})
        adapter = JsonPlannerAdapter(
            lambda payload: (raw, 7), plan_transform=planner.prepare_plan,
        )
        plan = adapter.next_plan("demo", 1, [history_record])
        self.assertEqual(1, len(plan.changes))
        self.assertEqual("", plan.changes[0].old_text)
        self.assertIn("E001-active-variant.json", plan.changes[0].path)

    def test_tool_translates_summary_and_redacts_data_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "experiments/logs/demo/E000-attempt1"
            data_dir = str(root.parent / "private-data")
            registry, planner = build(str(root))
            plan = planner.next_plan("demo", 0, [])
            runner = FakeRunner(root, "pointwise_fm")
            output = registry.get(plan.requested_tool).tool.run(
                plan, RunContext(str(root), str(run_dir), runner, data_dir),
            )
            self.assertEqual(0.60, output.primary)
            self.assertIn("<DATA_DIR>", output.command)
            self.assertNotIn(data_dir, output.command)
            self.assertNotIn("--score-test", output.command)
            self.assertNotIn("--smoke", output.command)
            self.assertTrue(all(not Path(path).is_absolute() for path in output.artifacts))

    def test_non_baseline_tool_requires_applied_variant_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, planner = build(str(root))
            baseline = planner.next_plan("demo", 0, [])
            plan = planner.next_plan(
                "demo", 1,
                [{"run_id": "demo-E000", "status": "accepted",
                  "single_primary_change": baseline.single_primary_change}],
            )
            run_dir = root / "experiments/logs/demo/E001-attempt1"
            runner = FakeRunner(root, "pairwise_bpr")
            with self.assertRaises(ValueError):
                registry.get(plan.requested_tool).tool.run(
                    plan, RunContext(str(root), str(run_dir), runner, str(root / "data")),
                )
            marker = root / plan.changes[0].path
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(plan.changes[0].new_text, encoding="utf-8")
            output = registry.get(plan.requested_tool).tool.run(
                plan, RunContext(str(root), str(run_dir), runner, str(root / "data")),
            )
            self.assertEqual(0.60, output.primary)

    def test_prediction_validator_checks_alignment_and_finiteness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            path.write_text(
                "row_id,user_id,video_id,score\n0,u1,v1,0.2\n1,u2,v2,0.3\n",
                encoding="utf-8",
            )
            rows = [(0, "u1", "v1"), (0, "u2", "v2")]
            validate_predictions(path, rows)
            path.write_text(
                "row_id,user_id,video_id,score\n0,u1,v1,nan\n1,u2,v2,0.3\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_predictions(path, rows)


if __name__ == "__main__":
    unittest.main()
