"""Deterministic non-model driver used to verify the complete agent loop."""

from __future__ import annotations

from typing import Tuple

from .planner import Candidate, DeterministicPlanner
from .registry import ToolDefinition, ToolRegistry
from .schemas import ExperimentPlan, ToolOutput


class SyntheticTool:
    def run(self, plan: ExperimentPlan, context: object) -> ToolOutput:
        if plan.params.get("fail_once", False):
            raise RuntimeError("synthetic recoverable failure")
        primary = float(plan.params["score"])
        return ToolOutput(
            command=["<in-process>", "synthetic"], GAUC=primary + 0.05,
            ndcg_at_5=primary - 0.05, primary=primary,
            elapsed_seconds=0.0,
            stdout_summary="synthetic validation completed",
            artifacts=[], token_usage=0, gpu_hours=0.0,
        )


def _plan(change: str, score: float, fail_once: bool = False,
          fallback: object = None) -> ExperimentPlan:
    return ExperimentPlan(
        run_id="template", iteration=0, parent_run_id=None,
        hypothesis="The controlled change '%s' will improve validation ranking." % change,
        rationale="Synthetic evidence fixture for the orchestration layer.",
        single_primary_change=change, experiment_type="synthetic",
        model_name="external-model-placeholder", feature_flags={},
        params={"score": score, "fail_once": fail_once}, seed=0,
        timeout_minutes=1.0, expected_cost="negligible",
        validation_protocol="train only; evaluate on valid; never score test",
        acceptance_rule=(
            "any higher primary updates best; cumulative gain > 0.002 resets convergence"
        ), editable_paths=[],
        requested_tool="synthetic", expected_signal="valid primary improves",
        fallback=dict(fallback or {}),
    )


def build(project_root: str) -> Tuple[ToolRegistry, DeterministicPlanner]:
    del project_root
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        "synthetic", SyntheticTool(),
        param_validators={
            "score": lambda value: isinstance(value, (int, float)) and 0.05 <= float(value) <= 0.95,
            "fail_once": lambda value: isinstance(value, bool),
        },
        required_params=("score", "fail_once"),
    ))
    planner = DeterministicPlanner([
        Candidate(_plan("establish controlled baseline", 0.600), 1.00, 0.00),
        Candidate(_plan("exercise deterministic recovery", 0.604, True,
                        {"fail_once": False}), 0.80, 0.05),
        Candidate(_plan("exercise validation rejection", 0.603), 0.60, 0.10),
    ])
    return registry, planner
