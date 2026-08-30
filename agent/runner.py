"""Bounded subprocess execution with process-group cleanup and redacted logs."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


class RunnerError(RuntimeError):
    def __init__(self, message: str, result: Optional["CommandResult"] = None) -> None:
        super().__init__(message)
        self.result = result


class CommandTimeout(RunnerError):
    pass


@dataclass(frozen=True)
class CommandResult:
    argv: List[str]
    returncode: int
    elapsed_seconds: float
    stdout_summary: str
    stderr_summary: str
    stdout_path: str
    stderr_path: str


class SafeRunner:
    def __init__(self, project_root: str, max_summary_chars: int = 8000,
                 redactions: Optional[Dict[str, str]] = None) -> None:
        self.project_root = str(Path(project_root).resolve())
        self.max_summary_chars = max_summary_chars
        self.redactions = {self.project_root: "<PROJECT_ROOT>"}
        self.redactions.update(redactions or {})

    def _redact(self, text: str) -> str:
        for secret, replacement in sorted(self.redactions.items(), key=lambda item: -len(item[0])):
            if secret:
                text = text.replace(secret, replacement)
        text = re.sub(
            r"(?i)(api[_-]?key|token|password|secret)(\s*[=:]\s*)([^\s,;]+)",
            r"\1\2<REDACTED>", text,
        )
        return text

    def _summary(self, text: str) -> str:
        if len(text) > self.max_summary_chars:
            return text[:self.max_summary_chars] + "\n...<TRUNCATED>"
        return text

    def run(self, argv: List[str], timeout_seconds: float, log_dir: str,
            env: Optional[Dict[str, str]] = None) -> CommandResult:
        if not argv or not all(isinstance(part, str) and part for part in argv):
            raise RunnerError("argv must be a non-empty list of strings")
        if timeout_seconds <= 0:
            raise RunnerError("timeout must be positive")
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stdout_file = directory / "stdout.log"
        stderr_file = directory / "stderr.log"
        started = time.monotonic()
        proc = subprocess.Popen(
            argv, cwd=self.project_root, shell=False, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
        elapsed = time.monotonic() - started
        stdout_redacted, stderr_redacted = self._redact(stdout), self._redact(stderr)
        stdout_file.write_text(stdout_redacted, encoding="utf-8")
        stderr_file.write_text(stderr_redacted, encoding="utf-8")
        result = CommandResult(
            argv=list(argv), returncode=proc.returncode, elapsed_seconds=elapsed,
            stdout_summary=self._summary(stdout_redacted),
            stderr_summary=self._summary(stderr_redacted),
            stdout_path=stdout_file.as_posix(), stderr_path=stderr_file.as_posix(),
        )
        if timed_out:
            raise CommandTimeout("command timed out", result)
        if proc.returncode != 0:
            raise RunnerError("command failed with exit code %d" % proc.returncode, result)
        return result
