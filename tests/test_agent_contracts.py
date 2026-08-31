import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent.patcher import ControlledPatcher
from agent.policy import AgentPolicy, PolicyViolation
from agent.registry import ToolDefinition, ToolRegistry
from agent.schemas import ExperimentPlan, ExperimentResult, SchemaError, ToolOutput
from agent.planner import FallbackPlanner, JsonPlannerAdapter, planner_history_summary


class NoopTool:
    def run(self, plan, context):
        return ToolOutput([], 0.6, 0.5, 0.55, 0.0)


def plan(**changes):
    base = ExperimentPlan(
        run_id="r", iteration=0, parent_run_id=None,
        hypothesis="one hypothesis", rationale="evidence based",
        single_primary_change="one change", experiment_type="unit",
        model_name="external", feature_flags={}, params={"x": 1}, seed=0,
        timeout_minutes=1, expected_cost="low",
        validation_protocol="train then evaluate valid only",
        acceptance_rule="improve valid primary by 0.002",
        editable_paths=[], requested_tool="noop",
    )
    return replace(base, **changes)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry = ToolRegistry()
        self.registry.register(ToolDefinition(
            "noop", NoopTool(), {"x": lambda value: value in (1, 2)}, ("x",),
        ))
        self.policy = AgentPolicy(str(self.root), self.registry)

    def tearDown(self):
        self.tmp.cleanup()

    def test_policy_rejects_frozen_path_traversal_param_and_test_selection(self):
        with self.assertRaises(PolicyViolation):
            self.policy.validate_plan(plan(editable_paths=["evaluate.py"]))
        with self.assertRaises(PolicyViolation):
            self.policy.validate_plan(plan(editable_paths=["agent/../evaluate.py"]))
        with self.assertRaises(ValueError):
            self.policy.validate_plan(plan(params={"x": 999}))
        with self.assertRaises(PolicyViolation):
            self.policy.validate_plan(plan(validation_protocol="--score --split test"))
        with self.assertRaises(ValueError):
            self.policy.validate_plan(plan(params={"x": 1, "split": "test"}))

    def test_failed_result_requires_null_metrics(self):
        result = ExperimentResult(
            "r-E000", 0, "failed", None, None, None, "c", "d", [],
            0.1, None, None, 0.0, 0, 0.0, "", "", "error", None,
            False, None, [], "h", "r", "c", "failed", "noop",
        )
        with self.assertRaises(SchemaError):
            result.validate()

    def test_exact_patch_can_rollback_and_cannot_touch_undeclared_path(self):
        target = self.root / "agent" / "plugin.py"
        target.parent.mkdir(parents=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
        from agent.schemas import FileChange
        changed = plan(
            editable_paths=["agent/plugin.py"],
            changes=[FileChange("agent/plugin.py", "VALUE = 1", "VALUE = 2")],
        )
        self.policy.validate_plan(changed)
        patcher = ControlledPatcher(str(self.root), self.policy)
        patcher.apply(changed)
        self.assertEqual("VALUE = 2\n", target.read_text(encoding="utf-8"))
        patcher.rollback()
        self.assertEqual("VALUE = 1\n", target.read_text(encoding="utf-8"))

    def test_multi_file_patch_is_transactional(self):
        from agent.schemas import FileChange
        first = self.root / "agent" / "one.py"
        second = self.root / "agent" / "two.py"
        first.parent.mkdir(parents=True)
        first.write_text("one\n", encoding="utf-8")
        second.write_text("two\n", encoding="utf-8")
        changed = plan(
            editable_paths=["agent/one.py", "agent/two.py"],
            changes=[
                FileChange("agent/one.py", "one", "changed"),
                FileChange("agent/two.py", "missing", "changed"),
            ],
        )
        patcher = ControlledPatcher(str(self.root), self.policy)
        with self.assertRaises(Exception):
            patcher.apply(changed)
        self.assertEqual("one\n", first.read_text(encoding="utf-8"))

    def test_new_plugin_file_is_removed_on_rollback(self):
        from agent.schemas import FileChange
        target = self.root / "features" / "plugin.py"
        changed = plan(
            editable_paths=["features/plugin.py"],
            changes=[FileChange("features/plugin.py", "", "ENABLED = True\n")],
        )
        self.policy.validate_plan(changed)
        patcher = ControlledPatcher(str(self.root), self.policy)
        patcher.apply(changed)
        self.assertTrue(target.exists())
        patcher.rollback()
        self.assertFalse(target.exists())

    def test_llm_adapter_only_returns_validated_json_plan(self):
        raw = json.dumps({"action": "plan", "plan": plan().to_dict()})
        adapter = JsonPlannerAdapter(lambda payload: (raw, 12))
        generated = adapter.next_plan("live", 3, [])
        self.assertEqual("live", generated.run_id)
        self.assertEqual(3, generated.iteration)
        self.assertEqual(12, adapter.token_usage)
        self.assertEqual(12, adapter.total_token_usage)
        invalid = JsonPlannerAdapter(lambda payload: ("{}", 1))
        with self.assertRaises(Exception):
            invalid.next_plan("live", 0, [])

    def test_llm_adapter_normalizes_null_optional_collections(self):
        value = plan().to_dict()
        value["changes"] = None
        value["editable_paths"] = None
        value["fallback"] = None
        raw = json.dumps({"action": "plan", "plan": value})
        generated = JsonPlannerAdapter(
            lambda payload: (raw, 4),
        ).next_plan("live", 4, [])
        self.assertEqual([], generated.changes)
        self.assertEqual([], generated.editable_paths)
        self.assertEqual({}, generated.fallback)

    def test_llm_shape_errors_are_actionable_not_type_errors(self):
        value = plan().to_dict()
        value["changes"] = [None]
        raw = json.dumps({"action": "plan", "plan": value})
        adapter = JsonPlannerAdapter(lambda payload: (raw, 1))
        with self.assertRaisesRegex(ValueError, r"schema validation.*changes\[0\]"):
            adapter.next_plan("live", 0, [])
        self.assertEqual("schema_validation", adapter.last_error_stage)
        self.assertIn('"changes": [null]', adapter.last_response_excerpt)

    def test_planner_history_excludes_large_execution_evidence(self):
        summary = planner_history_summary([{
            "run_id": "r-E000", "status": "accepted", "primary": 0.6,
            "params": {"x": 1}, "stdout_summary": "large output" * 1000,
            "code_diff_summary": "large diff" * 1000,
            "artifacts": ["private/path"],
        }])
        self.assertEqual("r-E000", summary[0]["run_id"])
        self.assertNotIn("stdout_summary", summary[0])
        self.assertNotIn("code_diff_summary", summary[0])
        self.assertNotIn("artifacts", summary[0])

    def test_llm_stop_envelope_and_deterministic_fallback(self):
        stop = JsonPlannerAdapter(
            lambda payload: ('{"action":"stop","plan":null}', 2)
        )
        self.assertIsNone(stop.next_plan("r", 0, []))

        class BrokenPlanner:
            token_usage = 3

            def next_plan(self, run_id, iteration, history):
                raise RuntimeError("provider down")

        class BackupPlanner:
            token_usage = 0

            def next_plan(self, run_id, iteration, history):
                return plan(run_id=run_id, iteration=iteration)

        fallback = FallbackPlanner(BrokenPlanner(), BackupPlanner())
        generated = fallback.next_plan("r", 0, [])
        self.assertIn("LLM fallback", generated.rationale)
        self.assertIn("RuntimeError", fallback.last_error)
        self.assertIn("provider down", fallback.last_error)
        self.assertEqual("deterministic_fallback", fallback.last_source)

        stop_fallback = FallbackPlanner(stop, BackupPlanner())
        continued = stop_fallback.next_plan("r", 0, [])
        self.assertIsNotNone(continued)
        self.assertIn("LLM requested stop", continued.rationale)
        self.assertEqual(2, stop_fallback.token_usage)
        self.assertIn('"action":"stop"', stop_fallback.last_response_excerpt)

    def test_provider_failure_does_not_reuse_previous_iteration_token_count(self):
        calls = [
            (json.dumps({"action": "plan", "plan": plan().to_dict()}), 7),
            RuntimeError("network unavailable"),
        ]

        def provider(payload):
            del payload
            response = calls.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        primary = JsonPlannerAdapter(provider)
        primary.next_plan("live", 0, [])
        self.assertEqual(7, primary.token_usage)
        with self.assertRaisesRegex(RuntimeError, "network unavailable"):
            primary.next_plan("live", 1, [])
        self.assertEqual(0, primary.token_usage)



if __name__ == "__main__":
    unittest.main()
