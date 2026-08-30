"""Explicit tool registry; the orchestrator never executes unregistered tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Protocol

from .schemas import ExperimentPlan, ToolOutput


class ToolContext(Protocol):
    project_root: str
    run_dir: str
    data_dir: Optional[str]


class ExperimentTool(Protocol):
    def run(self, plan: ExperimentPlan, context: ToolContext) -> ToolOutput:
        ...


Validator = Callable[[Any], bool]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    tool: ExperimentTool
    param_validators: Dict[str, Validator] = field(default_factory=dict)
    required_params: Iterable[str] = field(default_factory=tuple)

    def validate_params(self, params: Dict[str, Any]) -> None:
        unknown = sorted(set(params) - set(self.param_validators))
        if unknown:
            raise ValueError("parameters not allowed for %s: %s" % (self.name, unknown))
        missing = sorted(set(self.required_params) - set(params))
        if missing:
            raise ValueError("required parameters missing for %s: %s" % (self.name, missing))
        for name, value in params.items():
            if not self.param_validators[name](value):
                raise ValueError("invalid value for %s.%s" % (self.name, name))


class ToolRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if not definition.name or definition.name in self._items:
            raise ValueError("tool name must be non-empty and unique")
        self._items[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._items[name]
        except KeyError:
            raise ValueError("unregistered tool: %s" % name)

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._items))
