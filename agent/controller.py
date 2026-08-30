from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from agent.decision import choose_decision
from agent.evaluator import evaluate_agent_metrics
from agent.logger import ExperimentLogger
from agent.planner import ExperimentPlanner, E004_CONFIG_PATH, REAL_CONFIG_PATH
from agent.runner import DryRunRunner, RealRunner

ROOT = Path(__file__).resolve().parent.parent
E001_PATH = ROOT / "configs" / "E001_fm.json"
LOG_DIR = ROOT / "logs"
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"


def _current_git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Unable to read the current Git HEAD for dry-run logging.")
    return result.stdout.strip()


def _resolve_reference_primary() -> float:
    config_data = json.loads(E001_PATH.read_text(encoding="utf-8"))
    metrics = config_data.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("E001 validation-primary reference is missing the 'metrics' dictionary.")
    validation = metrics.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("E001 validation-primary reference is missing the 'validation' dictionary.")
    primary = validation.get("primary")
    if primary is None:
        raise ValueError("E001 validation-primary reference is missing the 'primary' value.")
    if isinstance(primary, bool):
        raise ValueError("E001 validation-primary reference must not be a boolean.")
    if not isinstance(primary, (int, float)):
        raise ValueError("E001 validation-primary reference is not numeric.")
    value = float(primary)
    if not math.isfinite(value):
        raise ValueError("E001 validation-primary reference is NaN or infinite.")
    return value


class _BestCheckpoint:
    def __init__(self, checkpoint_dir: Path = CHECKPOINT_DIR):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.checkpoint_dir / "best_state.json"

    def write(self, experiment_id: str, primary: float, checkpoint_path: Optional[str | Path] = None) -> Dict[str, Any]:
        payload = {
            "experiment_id": experiment_id,
            "primary": float(primary),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return payload

    def read(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


class DryRunController:
    def __init__(self, log_path: Optional[str | Path] = None):
        self.planner = ExperimentPlanner()
        self.runner = DryRunRunner()
        self.current_best_experiment_id = "E001_fm"
        self.current_best_primary = _resolve_reference_primary()
        self.best_checkpoint = None
        self.checkpoint = _BestCheckpoint()
        self._checkpoint_best_state()
        self.consecutive_no_improvement = 0
        self.max_no_improvement = 3
        self.no_significant_improvement = 0
        self.converged = False
        self.stop_reason = None
        self.log_path = Path(log_path) if log_path is not None else self._build_log_path()
        self.logger = ExperimentLogger(self.log_path)

    def _checkpoint_best_state(self) -> Dict[str, Any]:
        payload = self.checkpoint.write(self.current_best_experiment_id, self.current_best_primary, self.best_checkpoint)
        self.best_checkpoint = payload.get("checkpoint_path")
        return payload

    def _restore_best_state(self) -> Dict[str, Any]:
        payload = self.checkpoint.read()
        if payload is None:
            self.current_best_experiment_id = "E001_fm"
            self.current_best_primary = _resolve_reference_primary()
            self.best_checkpoint = None
            self.no_significant_improvement = 0
            self.converged = False
            self.stop_reason = None
            return self._checkpoint_best_state()
        self.current_best_experiment_id = str(payload.get("experiment_id", "E001_fm"))
        primary_value = payload.get("primary")
        if isinstance(primary_value, bool) or not isinstance(primary_value, (int, float)):
            self.current_best_primary = _resolve_reference_primary()
        else:
            self.current_best_primary = float(primary_value)
        self.best_checkpoint = payload.get("checkpoint_path")
        self.no_significant_improvement = 0
        self.converged = False
        self.stop_reason = None
        return self._checkpoint_best_state()

    def _write_failure_record(self, plan: Dict[str, Any], error: str, command: str = "dry_run: no experiment command executed", recovery: str = "rolled_back_to_best_checkpoint") -> None:
        record = {
            "experiment_id": plan.get("experiment_id", "unknown"),
            "parent_id": self.current_best_experiment_id,
            "hypothesis": plan.get("hypothesis", "unknown"),
            "configuration": {**plan.get("configuration", {}), "mode": "dry_run", "synthetic": True, "metrics_kind": "synthetic_validation"},
            "seed": plan.get("seed", 0),
            "command": command,
            "git_revision": _current_git_revision(),
            "metrics": {"validation": {}},
            "status": "failed",
            "decision": "ERROR",
            "error": str(error),
            "recovery": recovery,
            "wall_clock_sec": 0.0,
            "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
            "manual_interventions": [],
            "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.logger.append(record)

    def _build_log_path(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            unique_name = f"dry_run_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
            candidate = LOG_DIR / unique_name
            if not candidate.exists():
                return candidate

    def run_dry_run_loop(self, max_iterations: int = 3, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if timeout_sec is not None and timeout_sec < 0:
            raise ValueError("timeout_sec must be non-negative when provided.")

        iterations_run = 0
        accepted = 0
        rejected = 0
        deadline = None if timeout_sec is None else time.perf_counter() + float(timeout_sec)

        while iterations_run < max_iterations:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError(f"Dry-run loop exceeded timeout of {timeout_sec} seconds.")
            plan = self.planner.next_plan(max_iterations=max_iterations)
            if plan is None:
                break
            try:
                candidate_metrics = self.runner.run(plan)
                evaluated = evaluate_agent_metrics(candidate_metrics)
            except Exception as exc:
                self._restore_best_state()
                self._write_failure_record(plan, str(exc))
                raise
            candidate_primary = evaluated["primary"]
            previous_best = self.current_best_primary
            previous_best_experiment_id = self.current_best_experiment_id
            decision = choose_decision(candidate_primary, self.current_best_primary)
            improvement = candidate_primary - previous_best
            if decision == "ACCEPT":
                self.current_best_primary = candidate_primary
                self.current_best_experiment_id = plan["experiment_id"]
                accepted += 1
                self.best_checkpoint = None
                self._checkpoint_best_state()
                self.consecutive_no_improvement = 0
                self.no_significant_improvement = 0 if improvement > 0.002 else 1
            else:
                rejected += 1
                if improvement > 0.002:
                    self.consecutive_no_improvement = 0
                    self.no_significant_improvement = 0
                else:
                    self.consecutive_no_improvement += 1
                    self.no_significant_improvement += 1
            if self.no_significant_improvement >= 3:
                self.converged = True
                self.stop_reason = "convergence_no_improvement_3x"
                break
            record = {
                "experiment_id": plan["experiment_id"],
                "parent_id": previous_best_experiment_id,
                "hypothesis": plan["hypothesis"],
                "configuration": {**plan.get("configuration", {}), "mode": "dry_run", "synthetic": True, "metrics_kind": "synthetic_validation", "note": "Synthetic metrics are never real experimental results."},
                "seed": plan.get("seed", 0),
                "command": "dry_run: no experiment command executed",
                "git_revision": _current_git_revision(),
                "metrics": {"validation": {"GAUC": evaluated["GAUC"], "nDCG@5": evaluated["nDCG@5"], "primary": evaluated["primary"], "dry_run": True}, "synthetic": True},
                "status": "completed",
                "decision": decision,
                "error": "no_error",
                "recovery": "no_recovery_required",
                "wall_clock_sec": 0.0,
                "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
                "manual_interventions": [],
                "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self.logger.append(record)
            print(f"Iteration {iterations_run + 1}: {decision} | {plan['experiment_id']} | primary={candidate_primary:.4f}")
            iterations_run += 1

        summary = {
            "iterations_run": iterations_run,
            "accepted": accepted,
            "rejected": rejected,
            "best_experiment_id": self.current_best_experiment_id,
            "best_primary": self.current_best_primary,
            "log_path": str(self.log_path),
            "converged": self.converged,
            "stop_reason": self.stop_reason,
        }
        return summary


class RealController:
    def __init__(self, log_path: Optional[str | Path] = None, config_path: Optional[str | Path] = None):
        target = Path(config_path) if config_path is not None else REAL_CONFIG_PATH
        if str(target).endswith("E004_bpr_fm.json"):
            target = E004_CONFIG_PATH
        self.planner = ExperimentPlanner(config_path=target)
        self.runner = RealRunner()
        self.current_best_experiment_id = "E001_fm"
        self.current_best_primary = _resolve_reference_primary()
        self.best_checkpoint = None
        self.checkpoint = _BestCheckpoint()
        self._checkpoint_best_state()
        self.log_path = Path(log_path) if log_path is not None else self._build_log_path()
        self.logger = ExperimentLogger(self.log_path)
        self.attempts_run = 0
        self.iterations_run = 0
        self.recovery_attempts = 0
        self.no_significant_improvement = 0
        self.converged = False
        self.stop_reason = None

    def _checkpoint_best_state(self) -> Dict[str, Any]:
        payload = self.checkpoint.write(self.current_best_experiment_id, self.current_best_primary, self.best_checkpoint)
        self.best_checkpoint = payload.get("checkpoint_path")
        return payload

    def _restore_best_state(self) -> Dict[str, Any]:
        payload = self.checkpoint.read()
        if payload is None:
            self.current_best_experiment_id = "E001_fm"
            self.current_best_primary = _resolve_reference_primary()
            self.best_checkpoint = None
            self.no_significant_improvement = 0
            self.converged = False
            self.stop_reason = None
            return self._checkpoint_best_state()
        self.current_best_experiment_id = str(payload.get("experiment_id", "E001_fm"))
        primary_value = payload.get("primary")
        if isinstance(primary_value, bool) or not isinstance(primary_value, (int, float)):
            self.current_best_primary = _resolve_reference_primary()
        else:
            self.current_best_primary = float(primary_value)
        self.best_checkpoint = payload.get("checkpoint_path")
        self.no_significant_improvement = 0
        self.converged = False
        self.stop_reason = None
        return self._checkpoint_best_state()

    def _build_log_path(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            unique_name = f"real_validation_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
            candidate = LOG_DIR / unique_name
            if not candidate.exists():
                return candidate

    def _write_failure_record(self, plan: Dict[str, Any], error: str, command: str, recovery: str = "rolled_back_to_best_checkpoint") -> None:
        record = {
            "experiment_id": plan.get("experiment_id", "unknown"),
            "parent_id": self.current_best_experiment_id,
            "hypothesis": plan.get("hypothesis", "unknown"),
            "configuration": {**plan.get("model_config", plan.get("configuration", {})), "mode": "real_validation", "strategy": plan.get("strategy", "unknown")},
            "seed": plan.get("seed", 0),
            "command": command,
            "git_revision": _current_git_revision(),
            "metrics": {"validation": {}},
            "status": "failed",
            "decision": "ERROR",
            "error": str(error),
            "recovery": recovery,
            "wall_clock_sec": 0.0,
            "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
            "manual_interventions": [],
            "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.logger.append(record)

    def _record_result(self, candidate_primary: float, experiment_id: str, plan: Dict[str, Any], accepted: bool, checkpoint_path: Optional[str | Path] = None) -> None:
        candidate_primary = float(candidate_primary)
        checkpoint_value = str(checkpoint_path) if checkpoint_path is not None else None
        if checkpoint_value is not None:
            candidate_path = Path(checkpoint_value)
            if not candidate_path.exists() or candidate_path.suffix != ".npz":
                checkpoint_value = None

        if accepted and candidate_primary > self.current_best_primary:
            improvement = candidate_primary - self.current_best_primary
            self.current_best_primary = candidate_primary
            self.current_best_experiment_id = experiment_id
            if checkpoint_value is not None:
                self.best_checkpoint = checkpoint_value
            self._checkpoint_best_state()
            if improvement > 0.002:
                self.no_significant_improvement = 0
            else:
                self.no_significant_improvement += 1
            if self.no_significant_improvement >= 3:
                self.converged = True
                self.stop_reason = "convergence_no_improvement_3x"
            return

        if candidate_primary > self.current_best_primary:
            delta = candidate_primary - self.current_best_primary
            self.no_significant_improvement = 0 if delta > 0.002 else self.no_significant_improvement + 1
        else:
            self.no_significant_improvement += 1
        if self.no_significant_improvement >= 3:
            self.converged = True
            self.stop_reason = "convergence_no_improvement_3x"

    def run_real_loop(self, max_iterations: int = 1, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        if timeout_sec is not None and timeout_sec < 0:
            raise ValueError("timeout_sec must be non-negative when provided.")

        accepted = 0
        rejected = 0
        deadline = None if timeout_sec is None else time.perf_counter() + float(timeout_sec)
        while self.iterations_run < max_iterations:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError(f"Real validation loop exceeded timeout of {timeout_sec} seconds.")
            plan = self.planner.next_plan(max_iterations=max_iterations)
            if plan is None:
                break
            self.iterations_run += 1
            plan_attempts = 0
            last_exc = None
            last_result = None
            while plan_attempts < 2:
                plan_attempts += 1
                self.attempts_run += 1
                attempt_command = None
                try:
                    result = self.runner.run(plan, timeout=timeout_sec)
                    last_result = result
                    attempt_command = result.get("command")
                    metrics = {"GAUC": result["GAUC"], "nDCG@5": result["nDCG@5"]}
                    candidate_primary = 0.5 * (metrics["GAUC"] + metrics["nDCG@5"])
                    decision = choose_decision(candidate_primary, self.current_best_primary)
                    parent_id = self.current_best_experiment_id
                    record = {
                        "experiment_id": plan["experiment_id"],
                        "parent_id": parent_id,
                        "hypothesis": plan.get("hypothesis", "unknown"),
                        "configuration": {**plan.get("model_config", {}), "mode": "real_validation", "strategy": plan.get("strategy", "unknown"), "split": "validation"},
                        "seed": plan.get("seed", 0),
                        "command": attempt_command or "real_validation: command_not_executed",
                        "git_revision": _current_git_revision(),
                        "metrics": {"validation": {"GAUC": metrics["GAUC"], "nDCG@5": metrics["nDCG@5"], "primary": candidate_primary}},
                        "status": "completed",
                        "decision": decision,
                        "error": "no_error",
                        "recovery": "no_recovery_required",
                        "wall_clock_sec": 0.0,
                        "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
                        "manual_interventions": [],
                        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                    self.logger.append(record)
                    if decision == "ACCEPT":
                        checkpoint_path = result.get("checkpoint_path")
                        self._record_result(candidate_primary, plan["experiment_id"], plan, True, checkpoint_path)
                        accepted += 1
                    else:
                        self._record_result(candidate_primary, plan["experiment_id"], plan, False)
                        rejected += 1
                    break
                except Exception as exc:
                    last_exc = exc
                    if plan_attempts == 1:
                        self.recovery_attempts += 1
                        self._write_failure_record(plan, str(exc), attempt_command or "real_validation: command_not_executed", "retrying_once")
                        continue
                    self._write_failure_record(plan, str(exc), attempt_command or "real_validation: command_not_executed", "rolled_back_to_best_checkpoint")
                    raise
            if last_result is None:
                raise last_exc
            if last_result is not None and self.converged:
                break

        summary = {
            "iterations_run": self.iterations_run,
            "attempts_run": self.attempts_run,
            "recovery_attempts": self.recovery_attempts,
            "accepted": accepted,
            "rejected": rejected,
            "best_experiment_id": self.current_best_experiment_id,
            "best_primary": self.current_best_primary,
            "best_checkpoint": self.best_checkpoint,
            "log_path": str(self.log_path),
            "converged": self.converged,
            "stop_reason": self.stop_reason,
        }
        return summary


def run_agent(
    max_iterations: int = 3,
    log_path: Optional[str | Path] = None,
    mode: str = "dry_run",
    config_path: Optional[str | Path] = None,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    if mode == "real":
        controller = RealController(log_path=log_path, config_path=config_path)
        return controller.run_real_loop(max_iterations=max_iterations, timeout_sec=timeout_sec)
    controller = DryRunController(log_path=log_path)
    return controller.run_dry_run_loop(max_iterations=max_iterations, timeout_sec=timeout_sec)
