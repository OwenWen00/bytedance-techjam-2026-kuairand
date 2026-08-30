"""User-facing CLI with first-run LLM onboarding."""

from __future__ import annotations

import argparse
import getpass
import json
import uuid
from pathlib import Path
from typing import Callable, List, Optional

from .config import (
    ConfigError, default_config_path, load_config, load_or_prompt,
    prompt_for_config,
)
from .llm import PlannerLLMProvider, build_client
from .orchestrator import Orchestrator, load_driver
from .planner import FallbackPlanner, JsonPlannerAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kuairand-agent",
        description="Launch the controlled autonomous ML research agent",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--driver", default="agent.kuairand:build",
                        help="trusted MODULE:FUNCTION returning (registry, planner)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-wall-seconds", type=float, default=21600.0)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=None, help="local LLM config path")
    parser.add_argument("--configure", action="store_true",
                        help="replace local LLM configuration, then exit")
    parser.add_argument("--show-config", action="store_true",
                        help="show provider/model and a masked API key, then exit")
    parser.add_argument("--offline", action="store_true",
                        help="use the deterministic planner without API configuration")
    parser.add_argument("--non-interactive", action="store_true",
                        help="fail instead of prompting when API configuration is missing")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    return parser


def _candidate_dicts(planner: object) -> List[dict]:
    candidates = getattr(planner, "candidates", [])
    output = []
    for candidate in candidates:
        plan = getattr(candidate, "plan", None)
        if plan is not None and hasattr(plan, "to_dict"):
            output.append(plan.to_dict())
    return output


def _require_external_config_path(config_path: Path, project_root: str) -> None:
    project = Path(project_root).resolve()
    candidate = config_path.resolve(strict=False)
    try:
        candidate.relative_to(project)
    except ValueError:
        return
    raise ConfigError("LLM config must be outside the project directory")


def run_cli(
    argv: Optional[List[str]] = None,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    output_fn: Callable[[str], None] = print,
) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser() if args.config else default_config_path()
    try:
        if args.configure or args.show_config or not args.offline:
            _require_external_config_path(config_path, args.project_root)
        if args.configure:
            config = prompt_for_config(config_path, input_fn, secret_fn, output_fn)
            output_fn(json.dumps(config.masked_dict(), sort_keys=True))
            return 0
        if args.show_config:
            config = load_config(config_path)
            if config is None:
                raise ConfigError("no LLM API configuration found")
            output_fn(json.dumps(config.masked_dict(), sort_keys=True))
            return 0

        registry, deterministic_planner = load_driver(args.driver, args.project_root)
        requires_data = any(
            getattr(registry.get(name).tool, "requires_data_dir", False)
            for name in registry.names()
        )
        if requires_data and not args.data_dir:
            raise ConfigError("--data-dir is required by the selected model driver")
        planner = deterministic_planner
        if args.offline:
            output_fn("使用离线确定性 planner；不会调用外部 LLM API。")
        else:
            config = load_or_prompt(
                config_path, allow_prompt=not args.non_interactive,
                input_fn=input_fn, secret_fn=secret_fn, output_fn=output_fn,
            )
            client = build_client(config, timeout_seconds=args.llm_timeout)
            provider = PlannerLLMProvider(
                client, registry.names(), _candidate_dicts(deterministic_planner),
            )
            plan_transform = getattr(deterministic_planner, "prepare_plan", None)
            llm_planner = JsonPlannerAdapter(provider, plan_transform=plan_transform)
            planner = FallbackPlanner(llm_planner, deterministic_planner)
            output_fn("LLM planner: %s / %s" % (config.provider, config.model))

        run_id = args.run_id or ("run-" + uuid.uuid4().hex[:10])
        state = Orchestrator(
            args.project_root, registry, planner, run_id,
            max_iterations=args.max_iterations,
            max_wall_seconds=args.max_wall_seconds,
            data_dir=args.data_dir,
        ).run()
        output_fn(json.dumps({
            "run_id": state.run_id,
            "phase": state.phase,
            "stop_reason": state.stop_reason,
            "best_primary": state.best_primary,
        }, sort_keys=True))
        return 0
    except (ConfigError, ValueError, ImportError, AttributeError) as error:
        output_fn("配置或启动错误: %s" % error)
        return 2


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
