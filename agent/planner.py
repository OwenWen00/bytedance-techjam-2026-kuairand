import json
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "E003_dry_run.json"


class ExperimentPlanner:
    def __init__(self, config_path: str | Path = CONFIG_PATH):
        self.config_path = Path(config_path)
        self.plans: List[Dict[str, Any]] = self._load_plans()
        self._index = 0

    def _load_plans(self) -> List[Dict[str, Any]]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Dry-run plan config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if not isinstance(data, list):
            raise ValueError("Dry-run plan configuration must contain a list of plans.")

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
