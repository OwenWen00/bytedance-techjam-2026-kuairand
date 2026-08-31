"""Planner protocol and deterministic history-driven implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

from .schemas import ExperimentPlan, config_fingerprint


class Planner(Protocol):
    token_usage: int

    def next_plan(self, run_id: str, iteration: int,
                  history: List[Dict[str, Any]]) -> Optional[ExperimentPlan]:
        ...


_PLANNER_HISTORY_FIELDS = (
    "run_id", "iteration", "status", "requested_tool",
    "single_primary_change", "hypothesis", "rationale",
    "GAUC", "nDCG@5", "primary", "params", "seed",
    "feature_flags", "plan_fingerprint", "error_class",
    "recovery_action", "decision_rationale",
)


def planner_history_summary(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return decision evidence without logs, diffs, commands, or artifacts."""
    output: List[Dict[str, Any]] = []
    for record in history:
        compact: Dict[str, Any] = {}
        for name in _PLANNER_HISTORY_FIELDS:
            if name not in record:
                continue
            value = record[name]
            if isinstance(value, str) and len(value) > 600:
                value = value[:597] + "..."
            compact[name] = value
        output.append(compact)
    return output


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
        tried_fingerprints = {
            item.get("plan_fingerprint") for item in history
            if isinstance(item.get("plan_fingerprint"), str)
        }
        for item in history:
            if item.get("plan_fingerprint") or not isinstance(item.get("params"), dict):
                continue
            requested_tool = item.get("requested_tool")
            seed = item.get("seed")
            feature_flags = item.get("feature_flags")
            if (isinstance(requested_tool, str) and isinstance(seed, int)
                    and isinstance(feature_flags, dict)):
                tried_fingerprints.add(config_fingerprint(
                    requested_tool, item["params"], seed, feature_flags,
                ))
        failures = sum(1 for item in history if item.get("status") == "failed")
        no_gain = sum(1 for item in history if item.get("status") == "rejected")
        available = [candidate for candidate in self.candidates
                     if candidate.plan.single_primary_change not in tried
                     and config_fingerprint(
                         candidate.plan.requested_tool, candidate.plan.params,
                         candidate.plan.seed, candidate.plan.feature_flags,
                     ) not in tried_fingerprints]
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
        self.last_response_excerpt: Optional[str] = None
        self.last_error_stage: Optional[str] = None

    def next_plan(self, run_id: str, iteration: int,
                  history: List[Dict[str, Any]]) -> Optional[ExperimentPlan]:
        # A provider failure can happen before it reports usage. Do not carry
        # the previous iteration's token count into this result.
        self.token_usage = 0
        payload = {
            "run_id": run_id, "iteration": iteration,
            "history": planner_history_summary(history),
            "instruction": "Return one ExperimentPlan as JSON or null.",
        }
        self.last_response_excerpt = None
        self.last_error_stage = "provider"
        raw, tokens = self.provider(payload)
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("planner provider returned invalid token usage")
        self.token_usage = tokens
        self.total_token_usage += tokens
        if not isinstance(raw, str):
            raise ValueError("planner provider returned non-text output")
        self.last_response_excerpt = raw[:2000]
        self.last_error_stage = "json_decode"
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("planner JSON decode failed: %s" % error)
        if value is None:
            self.last_error_stage = None
            return None
        if isinstance(value, dict) and "action" in value:
            action = value.get("action")
            if action == "stop":
                self.last_error_stage = None
                return None
            if action != "plan" or not isinstance(value.get("plan"), dict):
                raise ValueError("LLM planner envelope must contain action=plan and a plan object")
            value = value["plan"]
        if not isinstance(value, dict):
            raise ValueError("LLM planner must return a JSON object")
        value["run_id"] = run_id
        value["iteration"] = iteration
        value["parent_run_id"] = history[-1].get("run_id") if history else None
        self.last_error_stage = "schema_validation"
        try:
            plan = ExperimentPlan.from_dict(value)
        except Exception as error:
            raise ValueError("planner schema validation failed: %s" % error)
        if self.plan_transform is not None:
            self.last_error_stage = "plan_transform"
            try:
                plan = self.plan_transform(plan, history)
                plan.validate()
            except Exception as error:
                raise ValueError("planner plan transform failed: %s" % error)
        self.last_error_stage = None
        return plan


class FallbackPlanner:
    """Use deterministic planning when the LLM is unavailable, invalid, or stops early."""

    def __init__(self, primary: Planner, fallback: Planner) -> None:
        self.primary = primary
        self.fallback = fallback
        self.token_usage = 0
        self.total_token_usage = 0
        self.last_error: Optional[str] = None
        self.last_error_stage: Optional[str] = None
        self.last_response_excerpt: Optional[str] = None
        self.last_source = "not_run"

    def next_plan(self, run_id: str, iteration: int,
                  history: List[Dict[str, Any]]) -> Optional[ExperimentPlan]:
        fallback_reason: Optional[str] = None
        try:
            plan = self.primary.next_plan(run_id, iteration, history)
            self.token_usage = getattr(self.primary, "token_usage", 0)
            self.total_token_usage += self.token_usage
            if plan is not None:
                self.last_error = None
                self.last_error_stage = None
                self.last_response_excerpt = getattr(
                    self.primary, "last_response_excerpt", None,
                )
                self.last_source = "llm"
                return plan
            fallback_reason = "LLM requested stop before deterministic search was exhausted"
            self.last_error = fallback_reason
            self.last_error_stage = "early_stop"
            self.last_response_excerpt = getattr(
                self.primary, "last_response_excerpt", None,
            )
        except Exception as error:
            self.token_usage = getattr(self.primary, "token_usage", 0)
            self.total_token_usage += self.token_usage
            self.last_error = "%s: %s" % (type(error).__name__, error)
            self.last_error_stage = getattr(self.primary, "last_error_stage", None)
            self.last_response_excerpt = getattr(
                self.primary, "last_response_excerpt", None,
            )
            fallback_reason = "LLM fallback used after %s" % type(error).__name__
        plan = self.fallback.next_plan(run_id, iteration, history)
        if plan is None:
            self.last_source = "exhausted"
            return None
        self.last_source = "deterministic_fallback"
        return replace(plan, rationale=plan.rationale + " %s." % fallback_reason)
