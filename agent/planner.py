import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "E003_dry_run.json"
REAL_CONFIG_PATH = ROOT / "configs" / "E003_fm_validation.json"
E004_CONFIG_PATH = ROOT / "configs" / "E004_bpr_fm.json"

STRATEGY_MODULES = {
    "fm_validation": "experiments.fm_validation",
    "pairwise_bpr_fm": "experiments.bpr_fm",
    "validation_only_fm_repro": "experiments.fm_validation",
    "validation_only_fm_k8": "experiments.fm_validation",
    "validation_only_fm_k32": "experiments.fm_validation",
    "recovery_demo": "experiments.recovery_demo",
    "fm_history_validation": "experiments.fm_history_validation",
}


def resolve_strategy_module(strategy: str) -> str:
    if not isinstance(strategy, str):
        raise ValueError("Experiment strategy must be a string.")
    if strategy.startswith("synthetic_") or strategy.startswith("dry_run"):
        return "experiments.fm_validation"
    module_name = STRATEGY_MODULES.get(strategy)
    if module_name is None:
        raise ValueError(f"Unknown experiment strategy: {strategy!r}")
    return module_name


class ExperimentPlanner:
    def __init__(self, config_path: str | Path = CONFIG_PATH):
        self.config_path = Path(config_path)
        if not self.config_path.exists() and not self.config_path.is_absolute():
            candidate = ROOT / "configs" / self.config_path.name
            if candidate.exists():
                self.config_path = candidate
        self.plans: List[Dict[str, Any]] = self._load_plans()
        self._index = 0

    def _load_plans(self) -> List[Dict[str, Any]]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Plan config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError("Plan configuration must contain a list of plans.")

        seen_ids = set()
        cleaned: List[Dict[str, Any]] = []
        for index, plan in enumerate(data):
            if not isinstance(plan, dict):
                raise ValueError(f"Plan at index {index} is not a dictionary.")

            experiment_id = plan.get("experiment_id")
            hypothesis = plan.get("hypothesis")
            if not experiment_id or not isinstance(experiment_id, str):
                raise ValueError(f"Plan at index {index} is missing a valid experiment_id.")
            if not hypothesis or not isinstance(hypothesis, str):
                raise ValueError(f"Plan {experiment_id} is missing a valid hypothesis.")
            if "strategy" not in plan:
                raise ValueError(f"Plan {experiment_id} is missing a strategy.")
            strategy = plan.get("strategy")
            if not isinstance(strategy, str):
                raise ValueError(f"Plan {experiment_id} has a non-string strategy.")
            resolve_strategy_module(strategy)
            if experiment_id in seen_ids:
                raise ValueError(f"Duplicate experiment_id detected: {experiment_id}")
            seen_ids.add(experiment_id)
            cleaned.append(plan)

        return cleaned

    def next_plan(self, max_iterations: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if self._index >= len(self.plans):
            return None
        if max_iterations is not None and self._index >= max_iterations:
            return None
        plan = self.plans[self._index]
        self._index += 1
        return plan

    def reset(self) -> None:
        self._index = 0


def load_plans(config_path: str | Path = CONFIG_PATH) -> List[Dict[str, Any]]:
    return ExperimentPlanner(config_path=config_path).plans
