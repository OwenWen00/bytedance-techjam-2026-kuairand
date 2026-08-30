"""Atomic, resumable run state."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunState:
    run_id: str
    phase: str = "INIT"
    next_iteration: int = 0
    best_primary: Optional[float] = None
    best_result_run_id: Optional[str] = None
    best_artifacts: List[str] = field(default_factory=list)
    consecutive_no_improvement: int = 0
    elapsed_seconds: float = 0.0
    completed_iterations: List[int] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RunState":
        return cls(**value)


class StateStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[RunState]:
        if not self.path.exists():
            return None
        return RunState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, state: RunState) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(str(temporary), str(self.path))
