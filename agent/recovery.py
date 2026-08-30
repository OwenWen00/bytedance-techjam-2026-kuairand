"""Failure classification and at-most-once deterministic retry."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

from .runner import CommandTimeout
from .schemas import ExperimentPlan


class RecoveryPolicy:
    def classify(self, error: BaseException) -> str:
        if isinstance(error, CommandTimeout):
            return "timeout"
        if isinstance(error, (ValueError, TypeError)):
            return "validation_error"
        return "execution_error"

    def retry(self, plan: ExperimentPlan, attempt: int) -> Tuple[Optional[ExperimentPlan], Optional[str]]:
        if attempt != 1 or not plan.fallback:
            return None, None
        params = dict(plan.params)
        params.update(plan.fallback)
        retry_plan = replace(plan, params=params)
        return retry_plan, "applied predeclared fallback parameters once"
