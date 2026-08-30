from __future__ import annotations

import json
import math
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
STRATEGY_MODULES = {
    "fm_validation": "experiments.fm_validation",
    "pairwise_bpr_fm": "experiments.bpr_fm",
}


def resolve_strategy_module(strategy: str) -> str:
    if not isinstance(strategy, str):
        raise ValueError("Experiment strategy must be a string.")
    module_name = STRATEGY_MODULES.get(strategy)
    if module_name is None:
        raise ValueError(f"Unknown experiment strategy: {strategy!r}")
    return module_name


def _is_test_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lower = key.lower()
    return lower == "test" or lower.startswith("test_") or lower.startswith("test-")


def _reject_test_leak(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_test_key(str(key)):
                raise ValueError(f"Test leakage detected at '{path}.{key}'")
            if isinstance(child, (dict, list)):
                _reject_test_leak(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                _reject_test_leak(child, f"{path}[{index}]")


def _reject_test_split(value: Any) -> None:
    if isinstance(value, str) and value.lower() == "test":
        raise ValueError("Test split is not allowed in validation-only result payloads.")


def _validate_metric_value(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metric '{name}' is missing, boolean, or non-numeric.")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"Metric '{name}' is NaN or infinite.")
    return fv


class DryRunRunner:
    def run(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Dry-run runner requires a plan dictionary.")

        metrics = plan.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Plan {plan.get('experiment_id', 'unknown')} has malformed metrics.")

        ga = metrics.get("GAUC")
        ndcg = metrics.get("nDCG@5")
        if ga is None or ndcg is None:
            raise ValueError(f"Plan {plan.get('experiment_id', 'unknown')} is missing GAUC or nDCG@5.")

        result = {
            "GAUC": float(ga),
            "nDCG@5": float(ndcg),
            "primary": float(metrics.get("primary", 0.5 * (float(ga) + float(ndcg)))),
            "dry_run": True,
            "validation": {
                "GAUC": float(ga),
                "nDCG@5": float(ndcg),
                "primary": float(metrics.get("primary", 0.5 * (float(ga) + float(ndcg)))),
                "dry_run": True,
            },
        }
        return result


class RealRunner:
    def __init__(self, repo_root: Optional[str | Path] = None, artifacts_dir: Optional[str | Path] = None):
        self.repo_root = Path(repo_root) if repo_root is not None else ROOT
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _result_path(self) -> Path:
        while True:
            candidate = self.artifacts_dir / f"result_{uuid.uuid4().hex}.json"
            if not candidate.exists():
                return candidate

    def run(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Real runner requires a plan dictionary.")

        experiment_id = plan.get("experiment_id")
        if not experiment_id:
            raise ValueError("Plan is missing an experiment_id.")

        model_config = plan.get("model_config") or plan.get("configuration") or {}
        if not isinstance(model_config, dict):
            raise ValueError(f"Plan {experiment_id} has a malformed configuration.")

        strategy = plan.get("strategy", "fm_validation")
        module_name = resolve_strategy_module(strategy)
        seed = plan.get("seed", 0)
        result_path = self._result_path()
        result_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            module_name,
            "--experiment-id",
            str(experiment_id),
            "--seed",
            str(seed),
            "--k",
            str(model_config.get("k", 16)),
            "--lr",
            str(model_config.get("lr", 0.001)),
            "--batch",
            str(model_config.get("batch", 8192)),
            "--max-epochs",
            str(model_config.get("max_epochs", 40)),
            "--patience",
            str(model_config.get("patience", 4)),
            "--result-path",
            str(result_path),
        ]
        if strategy == "pairwise_bpr_fm":
            command.extend(["--max-pairs-per-user", str(model_config.get("max_pairs_per_user", 64))])

        completed = subprocess.run(
            command,
            cwd=str(self.repo_root),
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Real FM validation subprocess failed for {experiment_id} (exit={completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip() or 'no output'}"
            )

        if not result_path.exists():
            fallback = sorted(self.artifacts_dir.glob("result_*.json"))
            if fallback:
                result_path = fallback[-1]
            else:
                raise FileNotFoundError(f"Expected result JSON was not produced for {experiment_id}: {result_path}")

        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Result JSON for {experiment_id} is malformed: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Result payload for {experiment_id} is not a JSON object.")

        _reject_test_leak(payload)
        _reject_test_split(payload.get("split"))

        if payload.get("experiment_id") != experiment_id:
            raise ValueError(
                f"Experiment id mismatch for {experiment_id}: result payload returned {payload.get('experiment_id')}"
            )
        if payload.get("split") != "validation":
            raise ValueError(f"Only validation results are allowed; got split={payload.get('split')!r}")
        if "primary" in payload or any(_is_test_key(str(key)) for key in payload.keys()):
            raise ValueError(f"Result payload for {experiment_id} contains forbidden primary or test field.")

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Result payload for {experiment_id} is missing a metrics dictionary.")
        if any(_is_test_key(str(key)) for key in metrics.keys()):
            raise ValueError(f"Result payload for {experiment_id} includes forbidden test metrics.")

        gauc = _validate_metric_value("GAUC", metrics.get("GAUC"))
        ndcg = _validate_metric_value("nDCG@5", metrics.get("nDCG@5"))

        return {
            "experiment_id": experiment_id,
            "split": "validation",
            "GAUC": gauc,
            "nDCG@5": ndcg,
            "result_path": str(result_path),
            "command": " ".join(command),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "return_code": completed.returncode,
        }


def dry_run_run(plan: Dict[str, Any]) -> Dict[str, Any]:
    return DryRunRunner().run(plan)
