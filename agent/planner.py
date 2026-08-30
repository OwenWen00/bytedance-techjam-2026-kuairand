"""Planner protocol and deterministic history-driven implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from .schemas import ExperimentPlan


class Planner(Protocol):
    token_usage: int

    def next_plan(self, run_id: str, iteration: int,
                  history: List[Dict[str, Any]]) -> Optional[ExperimentPlan]:
        ...


@dataclass(frozen=True)
class Candidate:
    plan: ExperimentPlan
    expected_gain: float
    cost_rank: float


class DeterministicPlanner:
    """Selects the best compatible untried candidate from observed evidence."""

    token_usage = 0

    def __init__(self, candidates: Iterable[Candidate]) -> None:
        self.candidates = list(candidates)

    def next_plan(self, run_id: str, iteration: int,
                  history: List[Dict[str, Any]]) -> Optional[ExperimentPlan]:
        tried = {item.get("single_primary_change") for item in history}
        failures = sum(1 for item in history if item.get("status") == "failed")
        no_gain = sum(1 for item in history if item.get("status") == "rejected")
        available = [candidate for candidate in self.candidates
                     if candidate.plan.single_primary_change not in tried]
        if not available:
            return None
        # After instability, favor low cost; after rejections, favor expected gain.
        def score(candidate: Candidate) -> float:
            return candidate.expected_gain * (1.0 + 0.15 * no_gain) - candidate.cost_rank * (1.0 + failures)
        chosen = max(available, key=score)
        rationale = "%s History: %d failed, %d rejected; evidence score %.4f." % (
            chosen.plan.rationale, failures, no_gain, score(chosen),
        )
        return replace(
            chosen.plan, run_id=run_id, iteration=iteration,
            parent_run_id=history[-1].get("run_id") if history else None,
            rationale=rationale,
        )


class JsonPlannerAdapter:
    """Provider-neutral LLM adapter; the provider returns JSON plus token usage.

    The adapter only produces a validated plan. It cannot run commands, edit
    files, or bypass AgentPolicy.
    """

    def __init__(self, provider: Callable[[Dict[str, Any]], Tuple[str, int]],
                 plan_transform: Optional[Callable[[ExperimentPlan, List[Dict[str, Any]]], ExperimentPlan]] = None) -> None:
        self.provider = provider
        self.plan_transform = plan_transform
        self.token_usage = 0
        self.total_token_usage = 0

    def next_plan(self, run_id: str, iteration: int,
                  history: List[Dict[str, Any]]) -> Optional[ExperimentPlan]:
        payload = {
            "run_id": run_id, "iteration": iteration,
            "history": history, "instruction": "Return one ExperimentPlan as JSON or null.",
        }
        raw, tokens = self.provider(payload)
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("planner provider returned invalid token usage")
        self.token_usage = tokens
        self.total_token_usage += tokens
        value = json.loads(raw)
        if value is None:
            return None
        if isinstance(value, dict) and "action" in value:
            action = value.get("action")
            if action == "stop":
                return None
            if action != "plan" or not isinstance(value.get("plan"), dict):
                raise ValueError("LLM planner envelope must contain action=plan and a plan object")
            value = value["plan"]
        if not isinstance(value, dict):
            raise ValueError("LLM planner must return a JSON object")
        value["run_id"] = run_id
        value["iteration"] = iteration
        value["parent_run_id"] = history[-1].get("run_id") if history else None
        plan = ExperimentPlan.from_dict(value)
        if self.plan_transform is not None:
            plan = self.plan_transform(plan, history)
            plan.validate()
        return plan


class FallbackPlanner:
    """Use deterministic planning when the LLM is unavailable or invalid."""

    def __init__(self, primary: Planner, fallback: Planner) -> None:
        self.primary = primary
        self.fallback = fallback
        self.token_usage = 0
        self.total_token_usage = 0
        self.last_error: Optional[str] = None

    def next_plan(self, run_id: str, iteration: int,
                  history: List[Dict[str, Any]]) -> Optional[ExperimentPlan]:
        try:
            plan = self.primary.next_plan(run_id, iteration, history)
            self.token_usage = getattr(self.primary, "token_usage", 0)
            self.total_token_usage += self.token_usage
            self.last_error = None
            return plan
        except Exception as error:
            self.token_usage = getattr(self.primary, "token_usage", 0)
            self.total_token_usage += self.token_usage
            self.last_error = "%s: %s" % (type(error).__name__, error)
            plan = self.fallback.next_plan(run_id, iteration, history)
            if plan is None:
                return None
            return replace(
                plan,
                rationale=plan.rationale + " LLM fallback used after %s." % type(error).__name__,
            )
