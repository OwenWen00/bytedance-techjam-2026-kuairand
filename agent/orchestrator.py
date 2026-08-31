"""Finite-state autonomous research loop, independent of model implementations."""

from __future__ import annotations

import argparse
import importlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .audit import JsonlLedger, git_diff_summary, git_sha
from .patcher import ControlledPatcher
from .planner import Planner
from .policy import AgentPolicy, FrozenFileViolation
from .recovery import RecoveryPolicy
from .registry import ToolRegistry
from .runner import RunnerError, SafeRunner
from .schemas import ExperimentPlan, ExperimentResult, ToolOutput, config_fingerprint
from .selector import ValidationSelector
from .state import RunState, StateStore
from .tools import RunContext


PHASES = ("INIT", "PLAN", "VALIDATE", "RUN", "EVALUATE", "SELECT", "LOG", "STOP")


def _safe_plan_fingerprint(plan: ExperimentPlan) -> Optional[str]:
    try:
        return config_fingerprint(
            plan.requested_tool, plan.params, plan.seed, plan.feature_flags,
        )
    except Exception:
        return None


class Orchestrator:
    def __init__(self, project_root: str, registry: ToolRegistry, planner: Planner,
                 run_id: str, selector: Optional[ValidationSelector] = None,
                 recovery: Optional[RecoveryPolicy] = None,
                 max_iterations: int = 50, max_wall_seconds: float = 21600.0,
                 data_dir: Optional[str] = None) -> None:
        self.root = Path(project_root).resolve()
        self.registry = registry
        self.planner = planner
        self.run_id = run_id
        self.selector = selector or ValidationSelector()
        self.recovery = recovery or RecoveryPolicy()
        self.max_iterations = min(max(1, max_iterations), 50)
        self.max_wall_seconds = min(max(1.0, max_wall_seconds), 21600.0)
        self.data_dir = str(Path(data_dir).expanduser().resolve()) if data_dir else None
        redactions = {self.data_dir: "<DATA_DIR>"} if self.data_dir else {}
        self.runner = SafeRunner(str(self.root), redactions=redactions)
        self.policy = AgentPolicy(str(self.root), registry)
        self.patcher = ControlledPatcher(str(self.root), self.policy)
        base = self.root / "experiments" / "logs" / run_id
        self.ledger = JsonlLedger(str(base / "experiments.jsonl"))
        self.events = JsonlLedger(str(base / "events.jsonl"))
        self.store = StateStore(str(base / "state.json"))

    def _planner_diagnostics(self) -> Dict[str, Any]:
        source = getattr(self.planner, "last_source", "deterministic")
        if source not in ("deterministic", "llm", "deterministic_fallback"):
            source = "deterministic"
        error = getattr(self.planner, "last_error", None)
        excerpt = getattr(self.planner, "last_response_excerpt", None)
        return {
            "planner_source": source,
            "planner_error": error[:2000] if isinstance(error, str) else None,
            "planner_error_stage": getattr(self.planner, "last_error_stage", None),
            "planner_response_excerpt": (
                excerpt[:2000] if isinstance(excerpt, str) else None
            ),
        }

    def _transition(self, state: RunState, phase: str, **details: Any) -> None:
        if phase not in PHASES:
            raise ValueError("unknown phase: %s" % phase)
        state.phase = phase
        self.store.save(state)
        event = {"run_id": state.run_id, "phase": phase, "iteration": state.next_iteration}
        event.update(details)
        self.events.append(event)

    def _load_or_create(self) -> RunState:
        state = self.store.load()
        if state is None:
            state = RunState(run_id=self.run_id)
            self.store.save(state)
        if state.run_id != self.run_id:
            raise ValueError("state run_id mismatch")
        return state

    def _failure_result(self, plan: ExperimentPlan, config_path: str,
                        parent_sha: Optional[str], error: BaseException,
                        attempt: int, recovery_action: Optional[str],
                        elapsed: float) -> ExperimentResult:
        command = []
        stdout = ""
        stderr = str(error)
        if isinstance(error, RunnerError) and error.result is not None:
            command = error.result.argv
            stdout = error.result.stdout_summary
            stderr = error.result.stderr_summary
        planner_diagnostics = self._planner_diagnostics()
        result = ExperimentResult(
            run_id="%s-E%03d" % (self.run_id, plan.iteration), iteration=plan.iteration,
            status="failed", code_version_id=git_sha(str(self.root)),
            parent_git_sha=parent_sha, result_git_sha=git_sha(str(self.root)),
            config_path=config_path, code_diff_summary=git_diff_summary(str(self.root)),
            command=command, GAUC=None, ndcg_at_5=None, primary=None,
            elapsed_seconds=elapsed, token_usage=getattr(self.planner, "token_usage", 0),
            gpu_hours=0.0, stdout_summary=stdout, stderr_summary=stderr,
            error_class=self.recovery.classify(error), recovery_action=recovery_action,
            human_intervention=False, human_intervention_reason=None, artifacts=[],
            hypothesis=plan.hypothesis, rationale=plan.rationale,
            single_primary_change=plan.single_primary_change,
            decision_rationale="experiment failed after bounded recovery",
            requested_tool=plan.requested_tool, attempt=attempt,
            params=dict(plan.params), seed=plan.seed,
            feature_flags=dict(plan.feature_flags),
            plan_fingerprint=_safe_plan_fingerprint(plan),
            planner_source=planner_diagnostics["planner_source"],
            planner_error=planner_diagnostics["planner_error"],
            planner_response_excerpt=planner_diagnostics["planner_response_excerpt"],
        )
        result.validate()
        return result

    def _execute(self, state: RunState, plan: ExperimentPlan,
                 config_path: str, parent_sha: Optional[str]) -> ExperimentResult:
        attempt = 1
        recovery_action = None
        started = time.monotonic()
        current = plan
        while True:
            run_dir = self.root / "experiments" / "logs" / self.run_id / ("E%03d-attempt%d" % (plan.iteration, attempt))
            try:
                definition = self.registry.get(current.requested_tool)
                output: ToolOutput = definition.tool.run(
                    current,
                    RunContext(str(self.root), str(run_dir), self.runner, self.data_dir),
                )
                output.validate()
                for artifact in output.artifacts:
                    self.policy.normalize_relative(artifact)
                self._transition(state, "EVALUATE", attempt=attempt)
                selection = self.selector.select(output.primary, state)
                self._transition(
                    state, "SELECT", accepted=selection.accepted,
                    significant=selection.significant,
                )
                status = "accepted" if selection.accepted else "rejected"
                planner_diagnostics = self._planner_diagnostics()
                result = ExperimentResult(
                    run_id="%s-E%03d" % (self.run_id, plan.iteration), iteration=plan.iteration,
                    status=status, code_version_id=git_sha(str(self.root)),
                    parent_git_sha=parent_sha, result_git_sha=git_sha(str(self.root)),
                    config_path=config_path, code_diff_summary=git_diff_summary(str(self.root)),
                    command=output.command, GAUC=output.GAUC, ndcg_at_5=output.ndcg_at_5,
                    primary=output.primary, elapsed_seconds=time.monotonic() - started,
                    token_usage=output.token_usage + getattr(self.planner, "token_usage", 0),
                    gpu_hours=output.gpu_hours, stdout_summary=output.stdout_summary,
                    stderr_summary=output.stderr_summary, error_class=None,
                    recovery_action=recovery_action, human_intervention=False,
                    human_intervention_reason=None, artifacts=output.artifacts,
                    hypothesis=plan.hypothesis, rationale=plan.rationale,
                    single_primary_change=plan.single_primary_change,
                    decision_rationale=selection.rationale,
                    requested_tool=current.requested_tool, attempt=attempt,
                    params=dict(current.params), seed=current.seed,
                    feature_flags=dict(current.feature_flags),
                    plan_fingerprint=config_fingerprint(
                        current.requested_tool, current.params,
                        current.seed, current.feature_flags,
                    ),
                    planner_source=planner_diagnostics["planner_source"],
                    planner_error=planner_diagnostics["planner_error"],
                    planner_response_excerpt=planner_diagnostics["planner_response_excerpt"],
                )
                result.validate()
                self.selector.update(state, output.primary, result.run_id, selection)
                if selection.accepted:
                    state.best_artifacts = list(output.artifacts)
                return result
            except Exception as error:
                retry_plan, action = self.recovery.retry(current, attempt)
                if retry_plan is None:
                    return self._failure_result(
                        current, config_path, parent_sha, error, attempt,
                        recovery_action, time.monotonic() - started,
                    )
                self.policy.validate_plan(retry_plan)
                recovery_action = action
                current = retry_plan
                attempt += 1

    def run(self) -> RunState:
        state = self._load_or_create()
        if state.phase == "STOP":
            return state
        while True:
            if len(state.completed_iterations) >= self.max_iterations:
                state.stop_reason = "max_iterations"
                self._transition(state, "STOP", reason=state.stop_reason)
                return state
            if state.elapsed_seconds >= self.max_wall_seconds:
                state.stop_reason = "wall_clock_budget"
                self._transition(state, "STOP", reason=state.stop_reason)
                return state
            if self.selector.converged(state):
                state.stop_reason = "converged"
                self._transition(state, "STOP", reason=state.stop_reason)
                return state

            iteration = state.next_iteration
            self._transition(state, "PLAN")
            plan = self.planner.next_plan(self.run_id, iteration, state.history)
            planner_diagnostics = self._planner_diagnostics()
            self.events.append({
                "run_id": self.run_id,
                "phase": "PLAN",
                "event": "planner_decision",
                "iteration": iteration,
                **planner_diagnostics,
            })
            if plan is None:
                state.stop_reason = "planner_exhausted"
                self._transition(state, "STOP", reason=state.stop_reason)
                return state
            if plan.iteration in state.completed_iterations:
                raise RuntimeError("planner attempted to repeat completed iteration")
            self._transition(state, "VALIDATE")
            parent_sha = git_sha(str(self.root))
            started = time.monotonic()
            try:
                self.policy.validate_plan(plan)
                config_path = self.patcher.write_config_snapshot(plan)
                self.patcher.apply(plan)
            except FrozenFileViolation:
                raise
            except Exception as error:
                self.patcher.rollback()
                result = self._failure_result(
                    plan, "<not-written>", parent_sha, error, 1, None,
                    time.monotonic() - started,
                )
            else:
                self._transition(state, "RUN")
                result = self._execute(state, plan, config_path, parent_sha)
            if result.status == "accepted":
                self.patcher.accept()
            else:
                self.patcher.rollback()
            state.elapsed_seconds += time.monotonic() - started
            self._transition(state, "LOG", status=result.status)
            record = result.to_dict()
            self.ledger.append(record)
            state.history.append(record)
            state.completed_iterations.append(iteration)
            state.next_iteration = iteration + 1
            if result.status == "failed":
                state.consecutive_no_improvement += 1
            self.policy.verify_frozen_files()
            self.store.save(state)


def load_driver(spec: str, project_root: str) -> Tuple[ToolRegistry, Planner]:
    if ":" not in spec:
        raise ValueError("driver must be MODULE:FUNCTION")
    module_name, function_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    registry, planner = factory(project_root)
    return registry, planner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model-agnostic autonomous research loop")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--driver", required=True, help="trusted MODULE:FUNCTION returning (registry, planner)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-wall-seconds", type=float, default=21600.0)
    parser.add_argument("--acceptance-epsilon", type=float, default=0.002)
    parser.add_argument("--convergence-patience", type=int, default=3)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    run_id = args.run_id or ("run-" + uuid.uuid4().hex[:10])
    registry, planner = load_driver(args.driver, args.project_root)
    state = Orchestrator(
        args.project_root, registry, planner, run_id,
        selector=ValidationSelector(
            epsilon=args.acceptance_epsilon,
            patience=args.convergence_patience,
        ),
        max_iterations=args.max_iterations, max_wall_seconds=args.max_wall_seconds,
        data_dir=args.data_dir,
    ).run()
    print(json.dumps({"run_id": state.run_id, "phase": state.phase,
                      "stop_reason": state.stop_reason, "best_primary": state.best_primary},
                     sort_keys=True))


if __name__ == "__main__":
    main()
