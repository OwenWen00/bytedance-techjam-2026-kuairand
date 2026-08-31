"""Validated, JSON-serializable contracts used by the research loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from typing import Any, Dict, List, Optional


class SchemaError(ValueError):
    """Raised when persisted or planner-produced data violates a contract."""


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("%s must be a non-empty string" % name)


def config_fingerprint(requested_tool: str, params: Dict[str, Any], seed: int,
                       feature_flags: Dict[str, bool]) -> str:
    """Return a stable identity for the model configuration that is executed."""
    payload = {
        "requested_tool": requested_tool,
        "params": params,
        "seed": seed,
        "feature_flags": feature_flags,
    }
    try:
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SchemaError("experiment configuration must be JSON serializable: %s" % error)
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class FileChange:
    """An exact, reversible text replacement in an allowed file."""

    path: str
    old_text: str
    new_text: str

    def validate(self) -> None:
        _require_text("change.path", self.path)
        if not isinstance(self.old_text, str) or not isinstance(self.new_text, str):
            raise SchemaError("change text values must be strings")
        if self.old_text == self.new_text:
            raise SchemaError("change must alter text")


@dataclass(frozen=True)
class ExperimentPlan:
    run_id: str
    iteration: int
    parent_run_id: Optional[str]
    hypothesis: str
    rationale: str
    single_primary_change: str
    experiment_type: str
    model_name: str
    feature_flags: Dict[str, bool]
    params: Dict[str, Any]
    seed: int
    timeout_minutes: float
    expected_cost: str
    validation_protocol: str
    acceptance_rule: str
    editable_paths: List[str]
    requested_tool: str
    expected_signal: str = "validation primary improves"
    fallback: Dict[str, Any] = field(default_factory=dict)
    changes: List[FileChange] = field(default_factory=list)

    def validate(self) -> None:
        for name in (
            "run_id", "hypothesis", "rationale", "single_primary_change",
            "experiment_type", "model_name", "expected_cost",
            "validation_protocol", "acceptance_rule", "requested_tool",
            "expected_signal",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.iteration, int) or isinstance(self.iteration, bool):
            raise SchemaError("iteration must be an integer")
        if self.iteration < 0:
            raise SchemaError("iteration must be non-negative")
        if self.parent_run_id is not None:
            _require_text("parent_run_id", self.parent_run_id)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SchemaError("seed must be an integer")
        if (not isinstance(self.timeout_minutes, (int, float))
                or isinstance(self.timeout_minutes, bool)):
            raise SchemaError("timeout_minutes must be numeric")
        if not 0 < float(self.timeout_minutes) <= 360:
            raise SchemaError("timeout_minutes must be in (0, 360]")
        if not isinstance(self.feature_flags, dict) or not isinstance(self.params, dict):
            raise SchemaError("feature_flags and params must be mappings")
        if not all(isinstance(name, str) and isinstance(enabled, bool)
                   for name, enabled in self.feature_flags.items()):
            raise SchemaError("feature_flags must map strings to booleans")
        if not isinstance(self.fallback, dict):
            raise SchemaError("fallback must be a mapping")
        if not isinstance(self.editable_paths, list):
            raise SchemaError("editable_paths must be a list")
        if not all(isinstance(path, str) for path in self.editable_paths):
            raise SchemaError("editable_paths must contain strings")
        if not isinstance(self.changes, list):
            raise SchemaError("changes must be a list")
        for change in self.changes:
            if not isinstance(change, FileChange):
                raise SchemaError("changes must contain FileChange objects")
            change.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ExperimentPlan":
        if not isinstance(value, dict):
            raise SchemaError("experiment plan must be a JSON object")
        data = dict(value)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise SchemaError("unknown experiment plan fields: %s" % ", ".join(unknown))

        # Models commonly emit null for optional collections. Normalize those
        # values before constructing the dataclass so malformed model output is
        # reported as SchemaError rather than leaking an unhelpful TypeError.
        for name, empty in (("changes", []), ("editable_paths", []), ("fallback", {})):
            if data.get(name) is None:
                data[name] = empty
        raw_changes = data.get("changes", [])
        if not isinstance(raw_changes, list):
            raise SchemaError("changes must be an array or null")
        changes = []
        for index, item in enumerate(raw_changes):
            if not isinstance(item, dict):
                raise SchemaError("changes[%d] must be an object" % index)
            try:
                change = FileChange(**item)
            except TypeError as error:
                raise SchemaError("invalid changes[%d]: %s" % (index, error))
            change.validate()
            changes.append(change)
        data["changes"] = changes
        try:
            plan = cls(**data)
        except TypeError as error:
            raise SchemaError("invalid experiment plan fields: %s" % error)
        plan.validate()
        return plan


@dataclass
class ExperimentResult:
    run_id: str
    iteration: int
    status: str
    code_version_id: Optional[str]
    parent_git_sha: Optional[str]
    result_git_sha: Optional[str]
    config_path: str
    code_diff_summary: str
    command: List[str]
    GAUC: Optional[float]
    ndcg_at_5: Optional[float]
    primary: Optional[float]
    elapsed_seconds: float
    token_usage: int
    gpu_hours: float
    stdout_summary: str
    stderr_summary: str
    error_class: Optional[str]
    recovery_action: Optional[str]
    human_intervention: bool
    human_intervention_reason: Optional[str]
    artifacts: List[str]
    hypothesis: str
    rationale: str
    single_primary_change: str
    decision_rationale: str
    requested_tool: str
    attempt: int = 1
    params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    plan_fingerprint: Optional[str] = None
    planner_source: str = "deterministic"
    planner_error: Optional[str] = None
    planner_response_excerpt: Optional[str] = None

    def validate(self) -> None:
        if self.status not in ("accepted", "rejected", "failed"):
            raise SchemaError("invalid result status: %s" % self.status)
        metrics = (self.GAUC, self.ndcg_at_5, self.primary)
        if self.status == "failed" and any(value is not None for value in metrics):
            raise SchemaError("failed results must have null metrics")
        if self.status != "failed" and any(value is None for value in metrics):
            raise SchemaError("successful results require all metrics")
        for value in metrics:
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise SchemaError("metrics must be finite values in [0, 1]")
        if self.token_usage < 0 or self.gpu_hours < 0 or self.elapsed_seconds < 0:
            raise SchemaError("resource values must be non-negative")
        if self.human_intervention and not self.human_intervention_reason:
            raise SchemaError("human intervention requires a reason")
        if not isinstance(self.params, dict) or not isinstance(self.feature_flags, dict):
            raise SchemaError("result params and feature_flags must be mappings")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SchemaError("result seed must be an integer")
        if self.plan_fingerprint is not None:
            _require_text("plan_fingerprint", self.plan_fingerprint)
        if self.planner_source not in ("deterministic", "llm", "deterministic_fallback"):
            raise SchemaError("invalid planner_source: %s" % self.planner_source)
        if self.planner_error is not None and not isinstance(self.planner_error, str):
            raise SchemaError("planner_error must be a string or null")
        if (self.planner_response_excerpt is not None
                and not isinstance(self.planner_response_excerpt, str)):
            raise SchemaError("planner_response_excerpt must be a string or null")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["nDCG@5"] = data.pop("ndcg_at_5")
        return data


@dataclass(frozen=True)
class ToolOutput:
    command: List[str]
    GAUC: float
    ndcg_at_5: float
    primary: float
    elapsed_seconds: float
    stdout_summary: str = ""
    stderr_summary: str = ""
    artifacts: List[str] = field(default_factory=list)
    token_usage: int = 0
    gpu_hours: float = 0.0

    def validate(self) -> None:
        for name in ("GAUC", "ndcg_at_5", "primary"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise SchemaError("%s must be in [0, 1]" % name)
