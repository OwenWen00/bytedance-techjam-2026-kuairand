"""Exact, reversible changes constrained by AgentPolicy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .policy import AgentPolicy
from .schemas import ExperimentPlan


class PatchError(RuntimeError):
    pass


class ControlledPatcher:
    def __init__(self, project_root: str, policy: AgentPolicy) -> None:
        self.root = Path(project_root).resolve()
        self.policy = policy
        self._backups: Dict[str, Optional[bytes]] = {}

    def write_config_snapshot(self, plan: ExperimentPlan) -> str:
        relative = "experiments/configs/%s/E%03d.json" % (plan.run_id, plan.iteration)
        self.policy.normalize_relative(relative)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != plan.to_dict():
                raise PatchError("config snapshot already exists with different content")
            return relative
        path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return relative

    def apply(self, plan: ExperimentPlan) -> None:
        self._backups = {}
        try:
            for change in plan.changes:
                relative = self.policy.normalize_relative(change.path)
                path = self.root / relative
                if change.old_text == "":
                    if path.exists():
                        raise PatchError("new file already exists: %s" % relative)
                    self._backups[relative] = None
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(change.new_text, encoding="utf-8")
                    continue
                if not path.exists():
                    raise PatchError("file does not exist: %s" % relative)
                before = path.read_bytes()
                text = before.decode("utf-8")
                if text.count(change.old_text) != 1:
                    raise PatchError("expected exactly one match in %s" % relative)
                self._backups[relative] = before
                path.write_text(text.replace(change.old_text, change.new_text, 1), encoding="utf-8")
            self.policy.verify_frozen_files()
        except Exception:
            self.rollback()
            raise

    def rollback(self) -> None:
        for relative, content in self._backups.items():
            path = self.root / relative
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(content)
        self._backups = {}
        self.policy.verify_frozen_files()

    def accept(self) -> None:
        self._backups = {}
