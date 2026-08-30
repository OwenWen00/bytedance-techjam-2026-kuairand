"""Model-independent experiment tool adapters."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .runner import SafeRunner
from .schemas import ExperimentPlan, ToolOutput


MetricsParser = Callable[[str], Dict[str, float]]
ArgvBuilder = Callable[[ExperimentPlan], List[str]]


def parse_last_json_object(stdout: str) -> Dict[str, float]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            value = json.loads(line)
            return {
                "GAUC": float(value["GAUC"]),
                "nDCG@5": float(value["nDCG@5"]),
                "primary": float(value["primary"]),
            }
    raise ValueError("tool output did not contain a JSON metrics object")


@dataclass(frozen=True)
class RunContext:
    project_root: str
    run_dir: str
    runner: SafeRunner
    data_dir: Optional[str] = None


class SubprocessExperimentTool:
    """Runs a code-defined argv builder; no free-form shell strings are accepted."""

    def __init__(self, argv_builder: ArgvBuilder,
                 parser: MetricsParser = parse_last_json_object,
                 artifact_resolver: Optional[Callable[[ExperimentPlan], List[str]]] = None) -> None:
        self.argv_builder = argv_builder
        self.parser = parser
        self.artifact_resolver = artifact_resolver or (lambda plan: [])

    def run(self, plan: ExperimentPlan, context: RunContext) -> ToolOutput:
        result = context.runner.run(
            self.argv_builder(plan), plan.timeout_minutes * 60.0, context.run_dir,
        )
        metrics = self.parser(result.stdout_summary)
        output = ToolOutput(
            command=result.argv, GAUC=metrics["GAUC"],
            ndcg_at_5=metrics["nDCG@5"], primary=metrics["primary"],
            elapsed_seconds=result.elapsed_seconds,
            stdout_summary=result.stdout_summary,
            stderr_summary=result.stderr_summary,
            artifacts=self.artifact_resolver(plan),
        )
        output.validate()
        return output


def python_module_argv(module: str, fixed_args: Optional[List[str]] = None) -> ArgvBuilder:
    """Build a safe adapter for modules that accept --param-name value arguments."""

    prefix = [sys.executable, "-m", module] + list(fixed_args or [])

    def build(plan: ExperimentPlan) -> List[str]:
        argv = list(prefix)
        for name in sorted(plan.params):
            value = plan.params[name]
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        argv.extend(["--seed", str(plan.seed)])
        return argv

    return build
