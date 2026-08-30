"""Append-only evidence ledger and lightweight Git/file provenance helpers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def git_output(project_root: str, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", project_root] + list(args), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip() or None


def git_sha(project_root: str) -> Optional[str]:
    return git_output(project_root, "rev-parse", "HEAD")


def git_diff_summary(project_root: str) -> str:
    scopes = ("agent", "models", "features", "experiments", "scripts", "docs", "tests")
    tracked = git_output(project_root, "diff", "--stat", "--", *scopes)
    untracked = git_output(
        project_root, "ls-files", "--others", "--exclude-standard", "--", *scopes
    )
    parts = []
    if tracked:
        parts.append(tracked)
    if untracked:
        paths = untracked.splitlines()
        rendered = paths[:20]
        if len(paths) > 20:
            rendered.append("... and %d more" % (len(paths) - 20))
        parts.append("untracked:\n" + "\n".join(rendered))
    return "\n".join(parts) if parts else "no code diff"


class JsonlLedger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        descriptor = os.open(str(self.path), flags, 0o644)
        try:
            os.write(descriptor, payload.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
