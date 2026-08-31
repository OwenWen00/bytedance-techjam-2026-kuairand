"""Validation-only acceptance and convergence decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .state import RunState


@dataclass(frozen=True)
class Selection:
    accepted: bool
    significant: bool
    rationale: str


class ValidationSelector:
    def __init__(self, epsilon: float = 0.002, patience: int = 3) -> None:
        if (not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool)
                or not math.isfinite(float(epsilon)) or float(epsilon) < 0.0):
            raise ValueError("acceptance epsilon must be a finite non-negative number")
        if not isinstance(patience, int) or isinstance(patience, bool) or not 1 <= patience <= 50:
            raise ValueError("convergence patience must be an integer in [1, 50]")
        self.epsilon = float(epsilon)
        self.patience = patience

    def select(self, primary: float, state: RunState) -> Selection:
        if state.best_primary is None:
            return Selection(
                True, True, "first valid result establishes the baseline and convergence reference",
            )
        best_delta = primary - state.best_primary
        reference = (
            state.convergence_reference_primary
            if state.convergence_reference_primary is not None
            else state.best_primary
        )
        significant_delta = primary - reference
        accepted = best_delta > 0.0
        significant = significant_delta > self.epsilon
        if accepted and significant:
            rationale = (
                "new validation best by %.6f; cumulative gain %.6f exceeds %.6f"
                % (best_delta, significant_delta, self.epsilon)
            )
        elif accepted:
            rationale = (
                "new validation best by %.6f; cumulative gain %.6f does not exceed "
                "convergence epsilon %.6f"
                % (best_delta, significant_delta, self.epsilon)
            )
        else:
            rationale = "validation primary did not beat best (delta %.6f)" % best_delta
        return Selection(accepted, significant, rationale)

    def update(self, state: RunState, primary: float, result_run_id: str,
               selection: Selection) -> None:
        if selection.accepted:
            state.best_primary = primary
            state.best_result_run_id = result_run_id
        if selection.significant:
            state.convergence_reference_primary = primary
            state.consecutive_no_improvement = 0
        else:
            state.consecutive_no_improvement += 1

    def converged(self, state: RunState) -> bool:
        return state.consecutive_no_improvement >= self.patience
