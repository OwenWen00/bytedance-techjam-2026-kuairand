"""Trusted adapter and evidence-driven plans for the integrated KuaiRand FM lab."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .planner import Candidate, DeterministicPlanner
from .registry import ToolDefinition, ToolRegistry, Validator
from .schemas import ExperimentPlan, FileChange, ToolOutput
from .tools import RunContext


VARIANT_CONFIG = {
    "pointwise_fm": ("pointwise", "official", "random"),
    "pairwise_bpr": ("pairwise", "official", "random"),
    "hard_negative_bpr": ("pairwise", "official", "hard"),
    "history_pairwise": ("pairwise", "history", "random"),
}


def _is_int(value: Any, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _is_number(value: Any, low: float, high: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and low <= float(value) <= high
    )


COMMON_VALIDATORS: Dict[str, Validator] = {
    "k": lambda value: _is_int(value, 2, 64),
    "lr": lambda value: _is_number(value, 1e-5, 0.1),
    "l2": lambda value: _is_number(value, 0.0, 0.1),
    "epochs": lambda value: _is_int(value, 1, 80),
    "batch_size": lambda value: _is_int(value, 256, 262144),
    "patience": lambda value: _is_int(value, 1, 10),
}
PAIRWISE_VALIDATORS: Dict[str, Validator] = {
    **COMMON_VALIDATORS,
    "negative_per_positive": lambda value: _is_int(value, 1, 5),
    "max_pairs_per_epoch": lambda value: _is_int(value, 0, 2_000_000),
}
HARD_NEGATIVE_VALIDATORS: Dict[str, Validator] = {
    **PAIRWISE_VALIDATORS,
    "hard_candidates": lambda value: _is_int(value, 2, 20),
    "hard_negative_warmup": lambda value: _is_int(value, 0, 10),
    "hard_negative_ratio": lambda value: _is_number(value, 0.0, 1.0),
}


class KuaiRandTrialTool:
    """Run one fixed model variant and translate its summary into ToolOutput."""

    requires_data_dir = True

    def __init__(self, variant: str) -> None:
        if variant not in VARIANT_CONFIG:
            raise ValueError("unknown KuaiRand model variant: %s" % variant)
        self.variant = variant

    @staticmethod
    def _relative(root: Path, value: Path) -> str:
        return value.resolve().relative_to(root).as_posix()

    def run(self, plan: ExperimentPlan, context: RunContext) -> ToolOutput:
        if not context.data_dir:
            raise ValueError("KuaiRand model tool requires --data-dir")
        root = Path(context.project_root).resolve()
        run_dir = Path(context.run_dir).resolve()
        relative_run_dir = self._relative(root, run_dir)
        output_relative = relative_run_dir + "/model"
        output_dir = root / output_relative
        if output_dir.exists():
            raise ValueError("model output directory already exists: %s" % output_relative)
        if self.variant != "pointwise_fm":
            marker_changes = [
                change for change in plan.changes
                if change.path.endswith("-active-variant.json")
            ]
            if len(marker_changes) != 1:
                raise ValueError("non-baseline run requires one controlled variant config diff")
            marker_path = root / marker_changes[0].path
            if not marker_path.is_file():
                raise ValueError("controlled variant config diff was not applied")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker != {"variant": self.variant}:
                raise ValueError("controlled variant config does not match the registered tool")

        argv = [
            sys.executable,
            "-m",
            "models.run_trial",
            "--variant",
            self.variant,
            "--data-dir",
            context.data_dir,
            "--starter-dir",
            ".",
            "--output-dir",
            output_relative,
        ]
        for name in sorted(plan.params):
            argv.extend(["--" + name.replace("_", "-"), str(plan.params[name])])
        argv.extend(["--seed", str(plan.seed)])

        result = context.runner.run(argv, plan.timeout_minutes * 60.0, context.run_dir)
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError("model run did not produce summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("test") is not None:
            raise ValueError("research tool produced forbidden test metrics")
        if summary.get("status") != "complete":
            raise ValueError("Agent model run must use the full train/valid split")
        expected = VARIANT_CONFIG[self.variant]
        config = summary.get("config", {})
        actual = (
            config.get("training_mode"),
            config.get("encoder_mode"),
            config.get("negative_strategy"),
        )
        if actual != expected:
            raise ValueError("model summary variant does not match the registered tool")
        valid = summary.get("valid", {})
        metrics = {
            "GAUC": float(valid["GAUC"]),
            "nDCG@5": float(valid["nDCG@5"]),
            "primary": float(valid["primary"]),
        }

        expected_artifacts = (
            "config.json",
            "epochs.jsonl",
            "best_model.npz",
            "validation_predictions.csv",
            "summary.json",
        )
        artifact_paths: List[str] = []
        for name in expected_artifacts:
            path = output_dir / name
            if not path.is_file():
                raise ValueError("model run is missing artifact: %s" % name)
            artifact_paths.append(self._relative(root, path))
        artifact_paths.extend([
            self._relative(root, Path(result.stdout_path)),
            self._relative(root, Path(result.stderr_path)),
        ])
        recorded_argv = ["<DATA_DIR>" if part == context.data_dir else part for part in result.argv]
        output = ToolOutput(
            command=recorded_argv,
            GAUC=metrics["GAUC"],
            ndcg_at_5=metrics["nDCG@5"],
            primary=metrics["primary"],
            elapsed_seconds=result.elapsed_seconds,
            stdout_summary=result.stdout_summary,
            stderr_summary=result.stderr_summary,
            artifacts=artifact_paths,
            token_usage=0,
            gpu_hours=0.0,
        )
        output.validate()
        return output


def _base_params() -> Dict[str, Any]:
    return {
        "k": 16,
        "lr": 0.001,
        "l2": 1e-6,
        "epochs": 40,
        "batch_size": 8192,
        "patience": 4,
    }


def _plan(
    tool: str,
    change: str,
    hypothesis: str,
    rationale: str,
    params: Dict[str, Any],
    feature_flags: Dict[str, bool],
) -> ExperimentPlan:
    return ExperimentPlan(
        run_id="template",
        iteration=0,
        parent_run_id=None,
        hypothesis=hypothesis,
        rationale=rationale,
        single_primary_change=change,
        experiment_type="offline_recommendation_ranking",
        model_name="numpy-factorization-machine",
        feature_flags=feature_flags,
        params=params,
        seed=0,
        timeout_minutes=60.0,
        expected_cost="CPU-only; full train and validation split",
        validation_protocol=(
            "Train on the official train split, select checkpoints on valid primary only, "
            "and never score or expose test to the Agent."
        ),
        acceptance_rule="valid primary improves over the accepted best by more than 0.002",
        editable_paths=[],
        requested_tool=tool,
        expected_signal="GAUC and nDCG@5 produce a higher validation primary",
        fallback={"batch_size": 4096},
    )


def _definition(name: str, variant: str, validators: Dict[str, Validator]) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        tool=KuaiRandTrialTool(variant),
        param_validators=validators,
        required_params=tuple(sorted(validators)),
    )


class KuaiRandPlanner(DeterministicPlanner):
    """Add a run-scoped, reversible config diff to every non-baseline plan."""

    def prepare_plan(self, plan, history):
        if plan is None:
            return None
        if not history and plan.requested_tool != "run_pointwise_fm":
            raise ValueError("the comparable pointwise baseline must run first")
        if plan.requested_tool == "run_pointwise_fm":
            return replace(plan, editable_paths=[], changes=[])
        marker = "experiments/configs/%s/E%03d-active-variant.json" % (
            plan.run_id, plan.iteration,
        )
        variant = {
            "run_pairwise_bpr": "pairwise_bpr",
            "run_hard_negative_bpr": "hard_negative_bpr",
            "run_history_pairwise": "history_pairwise",
        }[plan.requested_tool]
        payload = json.dumps({"variant": variant}, sort_keys=True) + "\n"
        return replace(
            plan,
            editable_paths=[marker],
            changes=[FileChange(marker, "", payload)],
        )

    def next_plan(self, run_id, iteration, history):
        plan = super().next_plan(run_id, iteration, history)
        if plan is None:
            return None
        return self.prepare_plan(plan, history)


def build(project_root: str) -> Tuple[ToolRegistry, DeterministicPlanner]:
    """Build the production registry and the no-key evidence-driven fallback planner."""
    del project_root
    registry = ToolRegistry()
    registry.register(_definition("run_pointwise_fm", "pointwise_fm", COMMON_VALIDATORS))
    registry.register(_definition("run_pairwise_bpr", "pairwise_bpr", PAIRWISE_VALIDATORS))
    registry.register(
        _definition(
            "run_hard_negative_bpr",
            "hard_negative_bpr",
            HARD_NEGATIVE_VALIDATORS,
        )
    )
    registry.register(
        _definition("run_history_pairwise", "history_pairwise", PAIRWISE_VALIDATORS)
    )

    baseline = _plan(
        "run_pointwise_fm",
        "establish the official five-field pointwise FM baseline",
        "The integrated implementation should reproduce the official validation baseline.",
        "A comparable baseline is required before any alternative can be accepted.",
        _base_params(),
        {"pairwise_loss": False, "time_safe_history": False},
    )
    pairwise_params = {
        **_base_params(),
        "negative_per_positive": 1,
        "max_pairs_per_epoch": 0,
    }
    pairwise = _plan(
        "run_pairwise_bpr",
        "replace pointwise logloss with within-user pairwise BPR",
        "A ranking-aligned pairwise loss should improve valid primary over pointwise FM.",
        "This is the lowest-cost isolated change after baseline and has positive three-seed evidence.",
        pairwise_params,
        {"pairwise_loss": True, "time_safe_history": False},
    )
    history_pairwise = _plan(
        "run_history_pairwise",
        "add leakage-safe historical and time fields to pairwise BPR",
        "Past-only user/item statistics and time fields should improve ranking under drift.",
        "Three paired seeds showed a mean +0.002220 primary gain over pointwise FM.",
        dict(pairwise_params),
        {"pairwise_loss": True, "time_safe_history": True},
    )
    planner = KuaiRandPlanner([
        Candidate(baseline, expected_gain=1.0, cost_rank=0.0),
        Candidate(pairwise, expected_gain=0.001507, cost_rank=0.00005),
        Candidate(history_pairwise, expected_gain=0.002220, cost_rank=0.0008),
    ])
    return registry, planner
