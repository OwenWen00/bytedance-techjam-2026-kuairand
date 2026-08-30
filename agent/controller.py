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


def _current_git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
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


class DryRunController:
    def __init__(self, log_path: Optional[str | Path] = None):
        self.planner = ExperimentPlanner()
        self.runner = DryRunRunner()
        self.current_best_experiment_id = "E001_fm"
        self.current_best_primary = _resolve_reference_primary()
        self.log_path = Path(log_path) if log_path is not None else self._build_log_path()
        self.logger = ExperimentLogger(self.log_path)

    def _build_log_path(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            unique_name = f"dry_run_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
            candidate = LOG_DIR / unique_name
            if not candidate.exists():
                return candidate

    def run_dry_run_loop(self, max_iterations: int = 3) -> Dict[str, Any]:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        iterations_run = 0
        accepted = 0
        rejected = 0

        while iterations_run < max_iterations:
            plan = self.planner.next_plan(max_iterations=max_iterations)
            if plan is None:
                break

            record_parent_id = self.current_best_experiment_id
            candidate_metrics = self.runner.run(plan)
            evaluated = evaluate_agent_metrics(candidate_metrics)
            candidate_primary = evaluated["primary"]
            decision = choose_decision(candidate_primary, self.current_best_primary)

            if decision == "ACCEPT":
                self.current_best_primary = candidate_primary
                self.current_best_experiment_id = plan["experiment_id"]
                accepted += 1
            else:
                rejected += 1

            record = {
                "experiment_id": plan["experiment_id"],
                "parent_id": record_parent_id,
                "hypothesis": plan["hypothesis"],
                "configuration": {
                    **plan.get("configuration", {}),
                    "mode": "dry_run",
                    "synthetic": True,
                    "metrics_kind": "synthetic_validation",
                    "note": "Synthetic metrics are never real experimental results.",
                },
                "seed": plan.get("seed", 0),
                "command": "dry_run: no experiment command executed",
                "git_revision": _current_git_revision(),
                "metrics": {
                    "validation": {
                        "GAUC": evaluated["GAUC"],
                        "nDCG@5": evaluated["nDCG@5"],
                        "primary": evaluated["primary"],
                        "dry_run": True,
                    },
                    "synthetic": True,
                },
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
        self.log_path = Path(log_path) if log_path is not None else self._build_log_path()
        self.logger = ExperimentLogger(self.log_path)

    def _build_log_path(self) -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            unique_name = f"real_validation_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
            candidate = LOG_DIR / unique_name
            if not candidate.exists():
                return candidate

    def _write_failure_record(self, plan: Dict[str, Any], error: str, command: str) -> None:
        record = {
            "experiment_id": plan.get("experiment_id", "unknown"),
            "parent_id": self.current_best_experiment_id,
            "hypothesis": plan.get("hypothesis", "unknown"),
            "configuration": {
                **plan.get("model_config", plan.get("configuration", {})),
                "mode": "real_validation",
                "strategy": plan.get("strategy", "unknown"),
            },
            "seed": plan.get("seed", 0),
            "command": command,
            "git_revision": _current_git_revision(),
            "metrics": {"validation": {}},
            "status": "failed",
            "decision": "ERROR",
            "error": str(error),
            "recovery": "not_attempted_e005_out_of_scope",
            "wall_clock_sec": 0.0,
            "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
            "manual_interventions": [],
            "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.logger.append(record)

    def run_real_loop(self, max_iterations: int = 1) -> Dict[str, Any]:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        iterations_run = 0
        accepted = 0
        rejected = 0

        while iterations_run < max_iterations:
            plan = self.planner.next_plan(max_iterations=max_iterations)
            if plan is None:
                break

            record_parent_id = self.current_best_experiment_id
            start = time.perf_counter()
            command = None
            try:
                result = self.runner.run(plan)
                command = result.get("command")
                metrics = {"GAUC": result["GAUC"], "nDCG@5": result["nDCG@5"]}
                candidate_primary = 0.5 * (metrics["GAUC"] + metrics["nDCG@5"])
                decision = choose_decision(candidate_primary, self.current_best_primary)
                if decision == "ACCEPT":
                    self.current_best_primary = candidate_primary
                    self.current_best_experiment_id = plan["experiment_id"]
                    accepted += 1
                else:
                    rejected += 1

                record = {
                    "experiment_id": plan["experiment_id"],
                    "parent_id": record_parent_id,
                    "hypothesis": plan["hypothesis"],
                    "configuration": {
                        **plan.get("model_config", {}),
                        "mode": "real_validation",
                        "strategy": plan.get("strategy", "unknown"),
                        "split": "validation",
                    },
                    "seed": plan.get("seed", 0),
                    "command": command,
                    "git_revision": _current_git_revision(),
                    "metrics": {"validation": {"GAUC": metrics["GAUC"], "nDCG@5": metrics["nDCG@5"], "primary": candidate_primary}},
                    "status": "completed",
                    "decision": decision,
                    "error": "no_error",
                    "recovery": "no_recovery_required",
                    "wall_clock_sec": time.perf_counter() - start,
                    "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
                    "manual_interventions": [],
                    "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                self.logger.append(record)
                print(
                    f"Iteration {iterations_run + 1}: {decision} | {plan['experiment_id']} | "
                    f"primary={candidate_primary:.4f} | prev_best={self.current_best_primary:.4f}"
                )
            except Exception as exc:  # no retry or recovery in E003-B
                self._write_failure_record(plan, str(exc), command or "real_validation: command_not_executed")
                raise

            iterations_run += 1

        summary = {
            "iterations_run": iterations_run,
            "accepted": accepted,
            "rejected": rejected,
            "best_experiment_id": self.current_best_experiment_id,
            "best_primary": self.current_best_primary,
            "log_path": str(self.log_path),
        }
        return summary


def run_agent(
    max_iterations: int = 3,
    log_path: Optional[str | Path] = None,
    mode: str = "dry_run",
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if mode == "real":
        controller = RealController(log_path=log_path, config_path=config_path)
        return controller.run_real_loop(max_iterations=max_iterations)
    controller = DryRunController(log_path=log_path)
    return controller.run_dry_run_loop(max_iterations=max_iterations)
