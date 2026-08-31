"""Standard-library LLM adapters for OpenAI Responses and DeepSeek Chat."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from .config import LLMConfig


class LLMAPIError(RuntimeError):
    pass


Transport = Callable[..., Any]


def _usage_total(response: Mapping[str, Any]) -> int:
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int) and total >= 0:
        return total
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    return int(input_tokens or 0) + int(output_tokens or 0)


class JsonHTTPClient:
    def __init__(self, config: LLMConfig, transport: Transport = urllib.request.urlopen,
                 timeout_seconds: float = 60.0, retries: int = 1) -> None:
        config.validate()
        self.config = config
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self.retries = max(0, min(retries, 2))

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
                "User-Agent": "kuairand-research-agent/1.0",
            },
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with self.transport(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise LLMAPIError("LLM API returned a non-object response")
                return value
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                detail = detail.replace(self.config.api_key, "<REDACTED>")
                retryable = error.code == 429 or 500 <= error.code < 600
                if retryable and attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise LLMAPIError("LLM API HTTP %d: %s" % (error.code, detail))
            except urllib.error.URLError as error:
                if attempt < self.retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise LLMAPIError("LLM API connection failed: %s" % error.reason)
            except (UnicodeError, ValueError) as error:
                raise LLMAPIError("LLM API returned invalid JSON: %s" % error)
        raise LLMAPIError("LLM API request failed")


class OpenAIResponsesClient(JsonHTTPClient):
    def complete_json(self, system_prompt: str, user_prompt: str,
                      response_schema: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
        response_format: Dict[str, Any]
        if response_schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "name": "experiment_plan_response",
                "schema": response_schema,
                # Local validation remains authoritative and provides better
                # diagnostics across providers. Non-strict schema mode avoids
                # rejecting provider-supported dynamic parameter dictionaries.
                "strict": False,
            }
        response = self._post("/responses", {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "text": {"format": response_format},
            "max_output_tokens": 4000,
            "store": False,
        })
        text = response.get("output_text")
        if not isinstance(text, str) or not text.strip():
            chunks: List[str] = []
            for item in response.get("output", []):
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        value = content.get("text")
                        if isinstance(value, str):
                            chunks.append(value)
            text = "".join(chunks)
        if not isinstance(text, str) or not text.strip():
            raise LLMAPIError("OpenAI response did not contain output text")
        return text, _usage_total(response)


class DeepSeekChatClient(JsonHTTPClient):
    def complete_json(self, system_prompt: str, user_prompt: str,
                      response_schema: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
        del response_schema
        response = self._post("/chat/completions", {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
            "temperature": 0.1,
            "max_tokens": 4000,
        })
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMAPIError("DeepSeek response did not contain message content")
        if not isinstance(text, str) or not text.strip():
            raise LLMAPIError("DeepSeek returned empty message content")
        return text, _usage_total(response)


def build_client(config: LLMConfig, transport: Transport = urllib.request.urlopen,
                 timeout_seconds: float = 60.0) -> JsonHTTPClient:
    if config.provider == "openai":
        return OpenAIResponsesClient(config, transport, timeout_seconds)
    if config.provider == "deepseek":
        return DeepSeekChatClient(config, transport, timeout_seconds)
    raise ValueError("unsupported provider: %s" % config.provider)


PLAN_SYSTEM_PROMPT = """You are the planning component of a controlled ML research agent.
Return exactly one JSON object and no prose. Never output shell commands or secrets.
Choose only a registered tool and only declared editable paths. Use train and valid
for model selection; never use test labels. Propose one primary change per iteration.
The candidate_plans are canonical starting points, not the complete experiment queue.
After canonical comparisons, propose an untried one-parameter neighbor of the accepted
validation best using search_context. Do not stop merely because candidate_plans were
already tried; iteration, wall-clock, convergence, and deterministic exhaustion are
enforced outside you.
The response envelope must be either {\"action\":\"stop\",\"plan\":null} or
{\"action\":\"plan\",\"plan\":<ExperimentPlan object>}."""


PLAN_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["plan", "stop"]},
        "plan": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "iteration": {"type": "integer"},
                        "parent_run_id": {"type": ["string", "null"]},
                        "hypothesis": {"type": "string"},
                        "rationale": {"type": "string"},
                        "single_primary_change": {"type": "string"},
                        "experiment_type": {"type": "string"},
                        "model_name": {"type": "string"},
                        "feature_flags": {"type": "object"},
                        "params": {"type": "object"},
                        "seed": {"type": "integer"},
                        "timeout_minutes": {"type": "number"},
                        "expected_cost": {"type": "string"},
                        "validation_protocol": {"type": "string"},
                        "acceptance_rule": {"type": "string"},
                        "editable_paths": {
                            "anyOf": [
                                {"type": "null"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                        "requested_tool": {"type": "string"},
                        "expected_signal": {"type": "string"},
                        "fallback": {
                            "anyOf": [{"type": "null"}, {"type": "object"}]
                        },
                        "changes": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "path": {"type": "string"},
                                            "old_text": {"type": "string"},
                                            "new_text": {"type": "string"},
                                        },
                                        "required": ["path", "old_text", "new_text"],
                                        "additionalProperties": False,
                                    },
                                },
                            ]
                        },
                    },
                    "required": [
                        "run_id", "iteration", "parent_run_id", "hypothesis",
                        "rationale", "single_primary_change", "experiment_type",
                        "model_name", "feature_flags", "params", "seed",
                        "timeout_minutes", "expected_cost", "validation_protocol",
                        "acceptance_rule", "editable_paths", "requested_tool",
                        "expected_signal", "fallback", "changes",
                    ],
                    "additionalProperties": False,
                },
            ]
        },
    },
    "required": ["action", "plan"],
    "additionalProperties": False,
}


class PlannerLLMProvider:
    """Turns orchestration history and driver candidates into a JSON plan request."""

    def __init__(self, client: JsonHTTPClient, tool_names: Iterable[str],
                 candidate_plans: Optional[Iterable[Dict[str, Any]]] = None,
                 search_context: Optional[Dict[str, Any]] = None) -> None:
        self.client = client
        self.tool_names = list(tool_names)
        self.candidate_plans = list(candidate_plans or [])
        self.search_context = dict(search_context or {})

    def __call__(self, payload: Dict[str, Any]) -> Tuple[str, int]:
        request = {
            "task": payload,
            "registered_tools": self.tool_names,
            "candidate_plans": self.candidate_plans,
            "search_context": self.search_context,
            "required_plan_fields": [
                "run_id", "iteration", "parent_run_id", "hypothesis", "rationale",
                "single_primary_change", "experiment_type", "model_name",
                "feature_flags", "params", "seed", "timeout_minutes",
                "expected_cost", "validation_protocol", "acceptance_rule",
                "editable_paths", "requested_tool", "expected_signal", "fallback",
                "changes",
            ],
        }
        return self.client.complete_json(
            PLAN_SYSTEM_PROMPT,
            json.dumps(request, ensure_ascii=False, sort_keys=True),
            response_schema=PLAN_RESPONSE_SCHEMA,
        )
