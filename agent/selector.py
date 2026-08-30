"""Validation-only acceptance and convergence decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .state import RunState


@dataclass(frozen=True)
class Selection:
    accepted: bool
    rationale: str


class ValidationSelector:
    def __init__(self, epsilon: float = 0.002, patience: int = 3) -> None:
        self.epsilon = epsilon
        self.patience = patience

    def select(self, primary: float, state: RunState) -> Selection:
        if state.best_primary is None:
            return Selection(True, "first valid result establishes the baseline")
        delta = primary - state.best_primary
        accepted = delta > self.epsilon
        if accepted:
            return Selection(True, "validation primary improved by %.6f (> %.6f)" % (delta, self.epsilon))
        return Selection(False, "validation primary delta %.6f did not exceed %.6f" % (delta, self.epsilon))

    def update(self, state: RunState, primary: float, result_run_id: str,
               selection: Selection) -> None:
        if selection.accepted:
            state.best_primary = primary
            state.best_result_run_id = result_run_id
            state.consecutive_no_improvement = 0
        else:
            state.consecutive_no_improvement += 1

    def converged(self, state: RunState) -> bool:
        return state.consecutive_no_improvement >= self.patience
