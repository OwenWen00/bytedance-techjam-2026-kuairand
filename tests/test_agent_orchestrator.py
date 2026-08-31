import json
import tempfile
import unittest
from pathlib import Path

from agent.orchestrator import Orchestrator
from agent.synthetic import build
from agent.planner import Candidate, DeterministicPlanner
from agent.registry import ToolDefinition, ToolRegistry
from agent.schemas import ExperimentPlan
from agent.selector import ValidationSelector
from agent.state import RunState


class OrchestratorTests(unittest.TestCase):
    def _root(self, path):
        root = Path(path)
        for name in ("data.py", "evaluate.py", "baseline.py", "submit.py", "baseline_scores.json"):
            (root / name).write_text(name + "\n", encoding="utf-8")
        return root

    def test_three_iteration_loop_accepts_recovers_rejects_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            registry, planner = build(str(root))
            orchestrator = Orchestrator(str(root), registry, planner, "demo", max_iterations=3)
            state = orchestrator.run()
            self.assertEqual("STOP", state.phase)
            self.assertEqual("max_iterations", state.stop_reason)
            self.assertEqual([0, 1, 2], state.completed_iterations)
            self.assertEqual(["accepted", "accepted", "rejected"],
                             [item["status"] for item in state.history])
            self.assertEqual(2, state.history[1]["attempt"])
            self.assertIn("fallback", state.history[1]["recovery_action"])
            self.assertEqual(0, state.history[0]["token_usage"])
            self.assertEqual(0.0, state.history[0]["gpu_hours"])
            self.assertEqual({"score": 0.6, "fail_once": False}, state.history[0]["params"])
            self.assertEqual(0, state.history[0]["seed"])
            self.assertEqual(64, len(state.history[0]["plan_fingerprint"]))
            ledger = root / "experiments/logs/demo/experiments.jsonl"
            self.assertEqual(3, len(ledger.read_text(encoding="utf-8").splitlines()))
            resumed = Orchestrator(str(root), registry, planner, "demo", max_iterations=3).run()
            self.assertEqual([0, 1, 2], resumed.completed_iterations)
            self.assertEqual(3, len(ledger.read_text(encoding="utf-8").splitlines()))

    def test_frozen_file_change_stops_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            registry, planner = build(str(root))
            orchestrator = Orchestrator(str(root), registry, planner, "frozen", max_iterations=1)
            (root / "evaluate.py").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(Exception):
                orchestrator.run()

    def test_terminal_failure_is_logged_with_null_metrics(self):
        class FailingTool:
            def run(self, plan, context):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            registry = ToolRegistry()
            registry.register(ToolDefinition("fail", FailingTool(), {}))
            plan = ExperimentPlan(
                run_id="template", iteration=0, parent_run_id=None,
                hypothesis="failure is recorded", rationale="failure fixture",
                single_primary_change="one failure", experiment_type="unit",
                model_name="external", feature_flags={}, params={}, seed=0,
                timeout_minutes=1, expected_cost="low",
                validation_protocol="train and valid only",
                acceptance_rule="valid primary improves by 0.002",
                editable_paths=[], requested_tool="fail",
            )
            planner = DeterministicPlanner([Candidate(plan, 1.0, 0.0)])
            state = Orchestrator(str(root), registry, planner, "failure", max_iterations=1).run()
            record = state.history[0]
            self.assertEqual("failed", record["status"])
            self.assertIsNone(record["GAUC"])
            self.assertIsNone(record["nDCG@5"])
            self.assertIsNone(record["primary"])
            self.assertEqual("execution_error", record["error_class"])

    def test_invalid_planner_plan_is_logged_instead_of_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            registry, _ = build(str(root))
            invalid = ExperimentPlan(
                run_id="template", iteration=0, parent_run_id=None,
                hypothesis="unsafe plan", rationale="fixture",
                single_primary_change="modify evaluator", experiment_type="unit",
                model_name="external", feature_flags={},
                params={"score": 0.6, "fail_once": False}, seed=0,
                timeout_minutes=1, expected_cost="low",
                validation_protocol="train and valid only",
                acceptance_rule="valid primary improves", editable_paths=["evaluate.py"],
                requested_tool="synthetic",
            )
            planner = DeterministicPlanner([Candidate(invalid, 1.0, 0.0)])
            state = Orchestrator(str(root), registry, planner, "invalid", max_iterations=1).run()
            self.assertEqual("failed", state.history[0]["status"])
            self.assertEqual("validation_error", state.history[0]["error_class"])
            self.assertEqual("<not-written>", state.history[0]["config_path"])

    def test_three_non_improvements_trigger_convergence(self):
        from agent.synthetic import _plan
        with tempfile.TemporaryDirectory() as directory:
            root = self._root(directory)
            registry, _ = build(str(root))
            candidates = [
                Candidate(_plan("baseline", 0.600), 4.0, 0.0),
                Candidate(_plan("no gain one", 0.601), 3.0, 0.0),
                Candidate(_plan("no gain two", 0.599), 2.0, 0.0),
                Candidate(_plan("no gain three", 0.6001), 1.0, 0.0),
            ]
            state = Orchestrator(
                str(root), registry, DeterministicPlanner(candidates),
                "converge", max_iterations=10,
            ).run()
            self.assertEqual("converged", state.stop_reason)
            self.assertEqual(4, len(state.history))

    def test_incremental_new_bests_update_anchor_and_cumulative_gain_resets_convergence(self):
        selector = ValidationSelector(epsilon=0.002, patience=3)
        state = RunState("incremental")

        baseline = selector.select(0.6000, state)
        selector.update(state, 0.6000, "E000", baseline)
        self.assertTrue(baseline.accepted)
        self.assertTrue(baseline.significant)

        first = selector.select(0.6009, state)
        selector.update(state, 0.6009, "E001", first)
        self.assertTrue(first.accepted)
        self.assertFalse(first.significant)
        self.assertEqual(0.6009, state.best_primary)
        self.assertEqual(1, state.consecutive_no_improvement)

        second = selector.select(0.6018, state)
        selector.update(state, 0.6018, "E002", second)
        self.assertTrue(second.accepted)
        self.assertFalse(second.significant)

        cumulative = selector.select(0.6021, state)
        selector.update(state, 0.6021, "E003", cumulative)
        self.assertTrue(cumulative.accepted)
        self.assertTrue(cumulative.significant)
        self.assertEqual(0, state.consecutive_no_improvement)
        self.assertEqual(0.6021, state.convergence_reference_primary)


if __name__ == "__main__":
    unittest.main()
