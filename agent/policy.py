"""Safety policy for plans, paths, immutable files, parameters, and splits."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable

from .registry import ToolRegistry
from .schemas import ExperimentPlan


class PolicyViolation(ValueError):
    pass


class FrozenFileViolation(PolicyViolation):
    pass


DEFAULT_EDITABLE = (
    "agent", "models", "features", "experiments/configs", "experiments/logs",
    "experiments/checkpoints", "experiments/predictions", "experiments/submissions",
    "scripts", "docs", "tests",
)
ROOT_EDITABLE = {
    "README.md", ".gitignore", "requirements.txt", ".env.example",
    "THIRD_PARTY_NOTICES.md",
}
FROZEN = ("data.py", "evaluate.py", "baseline.py", "submit.py", "baseline_scores.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AgentPolicy:
    def __init__(self, project_root: str, registry: ToolRegistry,
                 editable_roots: Iterable[str] = DEFAULT_EDITABLE) -> None:
        self.root = Path(project_root).resolve()
        self.registry = registry
        self.editable_roots = tuple(editable_roots)
        self.frozen_hashes = self.hash_frozen_files()

    def hash_frozen_files(self) -> Dict[str, str]:
        return {name: sha256_file(self.root / name) for name in FROZEN if (self.root / name).is_file()}

    def verify_frozen_files(self) -> None:
        current = self.hash_frozen_files()
        if current != self.frozen_hashes:
            changed = sorted(set(current) | set(self.frozen_hashes))
            changed = [name for name in changed if current.get(name) != self.frozen_hashes.get(name)]
            raise FrozenFileViolation("frozen files changed: %s" % changed)

    def normalize_relative(self, value: str, must_exist: bool = False) -> str:
        if not value or os.path.isabs(value):
            raise PolicyViolation("path must be non-empty and relative: %r" % value)
        parts = Path(value).parts
        if ".." in parts:
            raise PolicyViolation("path traversal is not allowed: %s" % value)
        candidate = (self.root / value).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PolicyViolation("path escapes project root: %s" % value)
        if must_exist and not candidate.exists():
            raise PolicyViolation("path does not exist: %s" % value)
        if candidate.exists() and candidate.is_symlink():
            raise PolicyViolation("symbolic links are not editable: %s" % value)
        return candidate.relative_to(self.root).as_posix()

    def is_editable(self, value: str) -> bool:
        relative = self.normalize_relative(value)
        if relative in ROOT_EDITABLE:
            return True
        return any(relative == root or relative.startswith(root + "/") for root in self.editable_roots)

    def validate_plan(self, plan: ExperimentPlan) -> None:
        plan.validate()
        if not 0 <= plan.seed <= 2 ** 31 - 1:
            raise PolicyViolation("seed is outside the controlled range")
        definition = self.registry.get(plan.requested_tool)
        definition.validate_params(plan.params)
        if len(plan.changes) > 0 and not plan.single_primary_change.strip():
            raise PolicyViolation("changes require one named primary change")
        declared = set()
        for path in plan.editable_paths:
            normalized = self.normalize_relative(path)
            if not self.is_editable(normalized):
                raise PolicyViolation("path is outside editable scope: %s" % path)
            declared.add(normalized)
        for change in plan.changes:
            normalized = self.normalize_relative(change.path)
            if normalized not in declared:
                raise PolicyViolation("changed path was not declared: %s" % change.path)
            if not self.is_editable(normalized):
                raise PolicyViolation("changed path is not editable: %s" % change.path)
        protocol = plan.validation_protocol.lower()
        params_json = json.dumps(plan.params, sort_keys=True).lower()
        forbidden = (
            "--score --split test" in protocol
            or bool(re.search(r'"split"\s*:\s*"test"', params_json))
            or "test_labels" in params_json
            or "hidden_test_labels" in params_json
        )
        if forbidden:
            raise PolicyViolation("test-based model selection is forbidden")
        if "valid" not in protocol:
            raise PolicyViolation("validation protocol must explicitly use valid")
        self.verify_frozen_files()
