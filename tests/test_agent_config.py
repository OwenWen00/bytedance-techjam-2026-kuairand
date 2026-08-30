import json
import stat
import tempfile
import unittest
from pathlib import Path

from agent.config import (
    LLMConfig, load_config, prompt_for_config, save_config,
)


class ConfigTests(unittest.TestCase):
    def test_save_load_and_mask_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config" / "llm.json"
            config = LLMConfig("openai", "gpt-test", "https://api.openai.com/v1", "secret-key-1234")
            save_config(config, path)
            self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))
            loaded = load_config(path, env={})
            self.assertEqual("secret-key-1234", loaded.api_key)
            self.assertNotIn("secret-key-1234", json.dumps(loaded.masked_dict()))

    def test_environment_is_valid_configuration_without_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                Path(directory) / "missing.json",
                env={
                    "KUAI_AGENT_LLM_PROVIDER": "deepseek",
                    "DEEPSEEK_API_KEY": "env-secret",
                    "KUAI_AGENT_LLM_MODEL": "deepseek-test",
                },
            )
            self.assertEqual("deepseek", config.provider)
            self.assertEqual("deepseek-test", config.model)
            self.assertEqual("env-secret", config.api_key)

    def test_same_provider_override_keeps_stored_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            save_config(
                LLMConfig("openai", "gpt-test", "https://api.openai.com/v1", "stored-key"),
                path,
            )
            config = load_config(path, env={"KUAI_AGENT_LLM_PROVIDER": "openai"})
            self.assertEqual("stored-key", config.api_key)

    def test_first_run_prompts_and_can_keep_key_in_memory_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            answers = iter(["2", "", "n"])
            output = []
            config = prompt_for_config(
                path, input_fn=lambda prompt: next(answers),
                secret_fn=lambda prompt: "prompt-secret",
                output_fn=output.append,
            )
            self.assertEqual("openai", config.provider)
            self.assertFalse(path.exists())
            self.assertNotIn("prompt-secret", "\n".join(output))

    def test_insecure_key_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            path.write_text(json.dumps({
                "provider": "openai", "model": "gpt-test",
                "base_url": "https://api.openai.com/v1", "api_key": "secret",
            }), encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaises(Exception):
                load_config(path, env={})

    def test_incomplete_file_is_treated_as_missing_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            path.write_text(json.dumps({"provider": "openai"}), encoding="utf-8")
            self.assertIsNone(load_config(path, env={}))


if __name__ == "__main__":
    unittest.main()
