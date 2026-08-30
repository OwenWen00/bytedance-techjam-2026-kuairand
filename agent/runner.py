from __future__ import annotations

import json
import math
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
CHECKPOINTS_DIR = ROOT / "artifacts" / "checkpoints"
STRATEGY_MODULES = {
    "fm_validation": "experiments.fm_validation",
    "pairwise_bpr_fm": "experiments.bpr_fm",
    "validation_only_fm_repro": "experiments.fm_validation",
    "validation_only_fm_k8": "experiments.fm_validation",
    "validation_only_fm_k32": "experiments.fm_validation",
    "recovery_demo": "experiments.recovery_demo",
}

STRATEGY_ALLOWED_KEYS = {
    "fm_validation": {"k", "lr", "batch", "max_epochs", "patience", "fields"},
    "validation_only_fm_repro": {"k", "lr", "batch", "max_epochs", "patience", "fields"},
    "validation_only_fm_k8": {"k", "lr", "batch", "max_epochs", "patience", "fields"},
    "validation_only_fm_k32": {"k", "lr", "batch", "max_epochs", "patience", "fields"},
    "pairwise_bpr_fm": {"k", "lr", "batch", "max_epochs", "patience", "max_pairs_per_user", "fields"},
    "recovery_demo": {"failure_mode", "mode", "sleep_sec", "ga", "ndcg"},
}


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def resolve_strategy_module(strategy: str) -> str:
    if not isinstance(strategy, str):
        raise ValueError("Experiment strategy must be a string.")
    if strategy in STRATEGY_MODULES:
        return STRATEGY_MODULES[strategy]
    if strategy.startswith("synthetic_") or strategy.startswith("dry_run"):
        return "experiments.fm_validation"
    raise ValueError(f"Unknown experiment strategy: {strategy!r}")


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


def _validate_strategy_config(strategy: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    allowed_keys = STRATEGY_ALLOWED_KEYS.get(strategy)
    if allowed_keys is None:
        raise ValueError(f"Unknown experiment strategy: {strategy!r}")
    if not isinstance(model_config, dict):
        raise ValueError(f"Plan for strategy {strategy!r} has a malformed configuration.")
    unknown_keys = sorted(str(key) for key in model_config.keys() if key not in allowed_keys)
    if unknown_keys:
        allowed = ", ".join(sorted(allowed_keys)) or "(none)"
        raise ValueError(
            f"Unknown configuration keys for strategy '{strategy}': {', '.join(unknown_keys)}. "
            f"Allowed keys: {allowed}."
        )
    normalized = {}
    for key in allowed_keys:
        if key in model_config:
            normalized[key] = model_config[key]
    return normalized


def _build_strategy_command(strategy: str, config: Dict[str, Any], experiment_id: str, seed: int, result_path: Path, checkpoint_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        resolve_strategy_module(strategy),
        "--experiment-id",
        str(experiment_id),
        "--seed",
        str(seed),
        "--result-path",
        str(result_path),
        "--checkpoint-path",
        str(checkpoint_path),
    ]

    if strategy == "recovery_demo":
        command.extend(["--mode", str(config.get("mode", "validation"))])
        if "failure_mode" in config:
            command.extend(["--failure-mode", str(config["failure_mode"])])
        command.extend(["--sleep-sec", str(float(config.get("sleep_sec", 0.0)))])
        if "ga" in config:
            command.extend(["--ga", str(config["ga"])])
        if "ndcg" in config:
            command.extend(["--ndcg", str(config["ndcg"])])
        return command

    model_map = {
        "k": "--k",
        "lr": "--lr",
        "batch": "--batch",
        "max_epochs": "--max-epochs",
        "patience": "--patience",
        "max_pairs_per_user": "--max-pairs-per-user",
    }
    for key, flag in model_map.items():
        if key in config:
            command.extend([flag, str(config[key])])
    return command


class RealRunner:
    @staticmethod
    def _reject_test_leak(value: Any, path: str = "root") -> None:
        _reject_test_leak(value, path)

    def __init__(self, repo_root: Optional[str | Path] = None, artifacts_dir: Optional[str | Path] = None, checkpoints_dir: Optional[str | Path] = None):
        self.repo_root = Path(repo_root) if repo_root is not None else ROOT
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_DIR
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _result_path(self) -> Path:
        while True:
            candidate = self.artifacts_dir / f"result_{uuid.uuid4().hex}.json"
            if not candidate.exists():
                return candidate

    def _checkpoint_path(self) -> Path:
        while True:
            candidate = self.checkpoints_dir / f"checkpoint_{uuid.uuid4().hex}.npz"
            if not candidate.exists():
                return candidate

    def run(self, plan: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Real runner requires a plan dictionary.")

        experiment_id = plan.get("experiment_id")
        if not experiment_id:
            raise ValueError("Plan is missing an experiment_id.")

        model_config = plan.get("model_config") or plan.get("configuration") or {}
        if not isinstance(model_config, dict):
            raise ValueError(f"Plan {experiment_id} has a malformed configuration.")

        strategy = plan.get("strategy", "fm_validation")
        model_config = _validate_strategy_config(strategy, model_config)
        module_name = resolve_strategy_module(strategy)
        seed = plan.get("seed", 0)
        result_path = self._result_path()
        checkpoint_path = self._checkpoint_path()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        command = _build_strategy_command(strategy, model_config, experiment_id, seed, result_path, checkpoint_path)

        run_kwargs = {
            "cwd": str(self.repo_root),
            "shell": False,
            "capture_output": True,
            "text": True,
            "check": False,
        }
        if timeout is not None:
            run_kwargs["timeout"] = timeout

        completed = subprocess.run(command, **run_kwargs)

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

        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"Result payload for {experiment_id} is missing a metadata dictionary.")
        if metadata is not None and any(_is_test_key(str(key)) for key in metadata.keys()):
            raise ValueError(f"Result payload for {experiment_id} includes forbidden test metadata fields.")

        resolved_checkpoint = None
        checkpoint_value = None
        if isinstance(metadata, dict):
            checkpoint_value = metadata.get("checkpoint_path")
            if checkpoint_value is not None:
                resolved_checkpoint = Path(checkpoint_value)
                if not resolved_checkpoint.is_absolute():
                    resolved_checkpoint = (self.repo_root / resolved_checkpoint).resolve(strict=False)
                else:
                    resolved_checkpoint = resolved_checkpoint.resolve(strict=False)
                allowed_root = self.checkpoints_dir.resolve(strict=False)
                if not _is_relative_to(resolved_checkpoint, allowed_root):
                    raise ValueError(f"Checkpoint path escapes artifacts directory for {experiment_id}: {checkpoint_value}")
                if resolved_checkpoint != checkpoint_path.resolve(strict=False):
                    raise ValueError(
                        f"Checkpoint path mismatch for {experiment_id}: expected {checkpoint_path} but got {checkpoint_value}"
                    )
                if not resolved_checkpoint.exists():
                    raise FileNotFoundError(f"Checkpoint file is missing for {experiment_id}: {resolved_checkpoint}")
                try:
                    with np.load(resolved_checkpoint, allow_pickle=False) as data:
                        if not data.files:
                            raise ValueError(f"Checkpoint file is empty for {experiment_id}: {resolved_checkpoint}")
                except ValueError as exc:
                    raise ValueError(f"Checkpoint file for {experiment_id} is unreadable: {exc}") from exc

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Result payload for {experiment_id} is missing a metrics dictionary.")
        if any(_is_test_key(str(key)) for key in metrics.keys()):
            raise ValueError(f"Result payload for {experiment_id} includes forbidden test metrics.")

        gauc = _validate_metric_value("GAUC", metrics.get("GAUC"))
        ndcg = _validate_metric_value("nDCG@5", metrics.get("nDCG@5"))

        if resolved_checkpoint is None:
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

        return {
            "experiment_id": experiment_id,
            "split": "validation",
            "GAUC": gauc,
            "nDCG@5": ndcg,
            "result_path": str(result_path),
            "checkpoint_path": str(resolved_checkpoint),
            "checkpoint_size": resolved_checkpoint.stat().st_size,
            "command": " ".join(command),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "return_code": completed.returncode,
        }


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




def dry_run_run(plan: Dict[str, Any]) -> Dict[str, Any]:
    return DryRunRunner().run(plan)
