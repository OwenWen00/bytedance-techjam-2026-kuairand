from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from agent.decision import choose_decision
from agent.evaluator import evaluate_agent_metrics
from agent.logger import ExperimentLogger
from agent.planner import ExperimentPlanner
from agent.runner import DryRunRunner

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


def run_agent(max_iterations: int = 3, log_path: Optional[str | Path] = None) -> Dict[str, Any]:
    controller = DryRunController(log_path=log_path)
    return controller.run_dry_run_loop(max_iterations=max_iterations)
