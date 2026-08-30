"""Local LLM configuration and first-run onboarding."""

from __future__ import annotations

import getpass
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional


SUPPORTED_PROVIDERS = ("deepseek", "openai")
PROVIDER_DEFAULTS = {
    "deepseek": {
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "model": "gpt-5.4",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)

    def validate(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ConfigError("unsupported provider: %s" % self.provider)
        for name in ("model", "base_url", "api_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError("%s must be a non-empty string" % name)
        if not self.base_url.startswith("https://"):
            raise ConfigError("base_url must use HTTPS")

    def masked_dict(self) -> Dict[str, str]:
        suffix = self.api_key[-4:] if len(self.api_key) >= 4 else "****"
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "****" + suffix,
        }


def default_config_path(env: Optional[Mapping[str, str]] = None) -> Path:
    values = env if env is not None else os.environ
    base = values.get("XDG_CONFIG_HOME")
    if base:
        return Path(base).expanduser() / "kuairand-agent" / "llm.json"
    return Path.home() / ".config" / "kuairand-agent" / "llm.json"


def _from_mapping(data: Mapping[str, str], env: Mapping[str, str]) -> LLMConfig:
    provider = str(data.get("provider", "")).lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError("provider must be one of: %s" % ", ".join(SUPPORTED_PROVIDERS))
    defaults = PROVIDER_DEFAULTS[provider]
    env_key = env.get(defaults["api_key_env"], "").strip()
    config = LLMConfig(
        provider=provider,
        model=str(env.get("KUAI_AGENT_LLM_MODEL", data.get("model", defaults["model"]))).strip(),
        base_url=str(data.get("base_url", defaults["base_url"])).rstrip("/"),
        api_key=env_key or str(data.get("api_key", "")).strip(),
    )
    config.validate()
    return config


def load_config(path: Path, env: Optional[Mapping[str, str]] = None) -> Optional[LLMConfig]:
    values = env if env is not None else os.environ
    provider_override = values.get("KUAI_AGENT_LLM_PROVIDER", "").lower().strip()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ConfigError("cannot read LLM config: %s" % error)
        if not isinstance(raw, dict):
            raise ConfigError("LLM config must contain a JSON object")
        if raw.get("api_key") and os.name == "posix":
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o077:
                raise ConfigError(
                    "LLM config contains an API key but permissions are %03o; expected 600" % mode
                )
        configured_provider = str(raw.get("provider", "")).lower().strip()
        if provider_override:
            raw["provider"] = provider_override
            # A key stored for another provider must never be reused.
            if provider_override != configured_provider:
                raw["api_key"] = ""
        provider = str(raw.get("provider", "")).lower().strip()
        if provider in SUPPORTED_PROVIDERS:
            env_name = PROVIDER_DEFAULTS[provider]["api_key_env"]
            if not values.get(env_name, "").strip() and not str(raw.get("api_key", "")).strip():
                return None
        return _from_mapping(raw, values)
    if provider_override:
        env_name = PROVIDER_DEFAULTS.get(provider_override, {}).get("api_key_env")
        if env_name and not values.get(env_name, "").strip():
            return None
        return _from_mapping({"provider": provider_override}, values)
    for provider in SUPPORTED_PROVIDERS:
        key = values.get(PROVIDER_DEFAULTS[provider]["api_key_env"], "").strip()
        if key:
            return _from_mapping({"provider": provider}, values)
    return None


def save_config(config: LLMConfig, path: Path) -> None:
    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    payload = json.dumps(asdict(config), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".llm-", dir=str(path.parent))
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, str(path))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def prompt_for_config(
    path: Path,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    output_fn: Callable[[str], None] = print,
) -> LLMConfig:
    output_fn("未检测到可用的 LLM API 配置。API key 不会写入项目或实验日志。")
    while True:
        choice = input_fn("选择服务商 [1] DeepSeek  [2] OpenAI/ChatGPT: ").strip().lower()
        if choice in ("1", "deepseek"):
            provider = "deepseek"
            break
        if choice in ("2", "openai", "chatgpt"):
            provider = "openai"
            break
        output_fn("请输入 1 或 2。")
    defaults = PROVIDER_DEFAULTS[provider]
    while True:
        api_key = secret_fn("请输入 %s API key: " % provider).strip()
        if api_key:
            break
        output_fn("API key 不能为空。")
    model = input_fn("模型名称 [%s]: " % defaults["model"]).strip() or defaults["model"]
    config = LLMConfig(provider, model, defaults["base_url"], api_key)
    config.validate()
    save_choice = input_fn("保存到本机配置 %s？[Y/n]: " % path).strip().lower()
    if save_choice not in ("n", "no"):
        save_config(config, path)
        output_fn("配置已保存；文件权限已限制为仅当前用户可读写。")
    else:
        output_fn("本次仅在内存中使用 API key。")
    return config


def load_or_prompt(
    path: Path,
    allow_prompt: bool = True,
    env: Optional[Mapping[str, str]] = None,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    output_fn: Callable[[str], None] = print,
) -> LLMConfig:
    config = load_config(path, env)
    if config is not None:
        return config
    if not allow_prompt:
        raise ConfigError(
            "LLM API config is missing; run interactively or set "
            "OPENAI_API_KEY / DEEPSEEK_API_KEY"
        )
    return prompt_for_config(path, input_fn, secret_fn, output_fn)
