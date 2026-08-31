import json
import tempfile
import unittest
from pathlib import Path

from agent.kuairand import build
from agent.orchestrator import Orchestrator
from agent.planner import JsonPlannerAdapter
from agent.registry import ToolDefinition, ToolRegistry
from agent.runner import CommandResult
from agent.schemas import ToolOutput, config_fingerprint
from agent.selector import ValidationSelector
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
    @staticmethod
    def _record(plan, status, primary):
        return {
            "run_id": "%s-E%03d" % (plan.run_id, plan.iteration),
            "status": status,
            "primary": primary,
            "single_primary_change": plan.single_primary_change,
            "requested_tool": plan.requested_tool,
            "params": dict(plan.params),
            "seed": plan.seed,
            "feature_flags": dict(plan.feature_flags),
            "plan_fingerprint": config_fingerprint(
                plan.requested_tool, plan.params, plan.seed, plan.feature_flags,
            ),
        }

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

    def test_planner_continues_with_unique_one_parameter_neighbors_of_best(self):
        registry, planner = build("unused")
        del registry
        baseline = planner.next_plan("long", 0, [])
        history = [self._record(baseline, "accepted", 0.600)]
        pairwise = planner.next_plan("long", 1, history)
        history.append(self._record(pairwise, "rejected", 0.601))
        history_plan = planner.next_plan("long", 2, history)
        history.append(self._record(history_plan, "accepted", 0.603))

        tuned = planner.next_plan("long", 3, history)
        self.assertEqual("run_history_pairwise", tuned.requested_tool)
        changed = [
            name for name in tuned.params
            if tuned.params[name] != history_plan.params[name]
        ]
        self.assertEqual(1, len(changed))
        self.assertIn("around validation-best", tuned.single_primary_change)
        tuned_fingerprint = config_fingerprint(
            tuned.requested_tool, tuned.params, tuned.seed, tuned.feature_flags,
        )
        self.assertNotIn(tuned_fingerprint, {
            record["plan_fingerprint"] for record in history
        })

        history.append(self._record(tuned, "rejected", 0.602))
        next_tuned = planner.next_plan("long", 4, history)
        self.assertNotEqual(
            tuned_fingerprint,
            config_fingerprint(
                next_tuned.requested_tool, next_tuned.params,
                next_tuned.seed, next_tuned.feature_flags,
            ),
        )

    def test_llm_duplicate_configuration_is_rejected_by_fingerprint(self):
        registry, planner = build("unused")
        del registry
        baseline = planner.next_plan("demo", 0, [])
        with self.assertRaisesRegex(ValueError, "previously executed"):
            planner.prepare_plan(
                baseline,
                [self._record(baseline, "accepted", 0.600)],
            )

    def test_orchestrator_executes_a_fourth_dynamic_experiment(self):
        class ScoredTool:
            def run(self, plan, context):
                del context
                primary = {
                    "run_pointwise_fm": 0.600,
                    "run_pairwise_bpr": 0.601,
                    "run_history_pairwise": 0.603,
                }.get(plan.requested_tool, 0.599)
                if plan.iteration >= 3:
                    primary = 0.602
                return ToolOutput(
                    ["<in-process>", plan.requested_tool],
                    primary + 0.05, primary - 0.05, primary, 0.0,
                )

        with tempfile.TemporaryDirectory() as directory:
            source_registry, planner = build(directory)
            registry = ToolRegistry()
            for name in source_registry.names():
                definition = source_registry.get(name)
                registry.register(ToolDefinition(
                    name, ScoredTool(), definition.param_validators,
                    definition.required_params,
                ))
            state = Orchestrator(
                directory, registry, planner, "dynamic-four",
                selector=ValidationSelector(patience=12),
                max_iterations=4,
            ).run()
            self.assertEqual("max_iterations", state.stop_reason)
            self.assertEqual(4, len(state.history))
            self.assertEqual("run_history_pairwise", state.history[3]["requested_tool"])
            self.assertIn("around validation-best", state.history[3]["single_primary_change"])

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

    def test_fourth_llm_neighborhood_plan_accepts_null_optional_collections(self):
        registry, planner = build("unused")
        del registry
        baseline = planner.next_plan("live-four", 0, [])
        history = [self._record(baseline, "accepted", 0.600)]
        pairwise = planner.next_plan("live-four", 1, history)
        history.append(self._record(pairwise, "rejected", 0.601))
        history_plan = planner.next_plan("live-four", 2, history)
        history.append(self._record(history_plan, "accepted", 0.603))
        tuned = planner.next_plan("live-four", 3, history)

        value = tuned.to_dict()
        value["changes"] = None
        value["editable_paths"] = None
        value["fallback"] = None
        raw = json.dumps({"action": "plan", "plan": value})
        adapter = JsonPlannerAdapter(
            lambda payload: (raw, 11), plan_transform=planner.prepare_plan,
        )
        generated = adapter.next_plan("live-four", 3, history)
        self.assertEqual("run_history_pairwise", generated.requested_tool)
        self.assertEqual(1, len(generated.changes))
        self.assertIn("E003-active-variant.json", generated.changes[0].path)
        self.assertIsNone(adapter.last_error_stage)

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
