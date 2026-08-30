import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from agent.cli import run_cli
from agent.config import LLMConfig, save_config


class CLITests(unittest.TestCase):
    def test_offline_cli_runs_without_api_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = []
            result = run_cli([
                "--offline", "--project-root", directory,
                "--driver", "agent.synthetic:build",
                "--run-id", "cli-offline", "--max-iterations", "1",
            ], input_fn=lambda prompt: self.fail("unexpected prompt"),
               secret_fn=lambda prompt: self.fail("unexpected secret prompt"),
               output_fn=output.append)
            self.assertEqual(0, result)
            self.assertTrue((Path(directory) / "experiments/logs/cli-offline/state.json").exists())
            self.assertIn("offline", "\n".join(output).lower())

    def test_configured_llm_planner_is_connected_to_cli(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, system_prompt, user_prompt):
                self.calls += 1
                request = json.loads(user_prompt)
                selected = request["candidate_plans"][0]
                return json.dumps({"action": "plan", "plan": selected}), 9

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as config_directory:
            root = Path(directory)
            config_path = Path(config_directory) / "local-config.json"
            save_config(
                LLMConfig("openai", "gpt-test", "https://api.openai.com/v1", "fake-secret"),
                config_path,
            )
            client = FakeClient()
            output = []
            with patch("agent.cli.build_client", return_value=client):
                result = run_cli([
                    "--project-root", directory, "--config", str(config_path),
                    "--driver", "agent.synthetic:build", "--run-id", "cli-llm",
                    "--max-iterations", "1", "--non-interactive",
                ], output_fn=output.append)
            self.assertEqual(0, result)
            self.assertEqual(1, client.calls)
            ledger = root / "experiments/logs/cli-llm/experiments.jsonl"
            record = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(9, record["token_usage"])
            self.assertNotIn("fake-secret", ledger.read_text(encoding="utf-8"))

    def test_cli_rejects_api_config_inside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            output = []
            result = run_cli([
                "--project-root", directory,
                "--config", str(Path(directory) / "secret.json"),
                "--show-config",
            ], output_fn=output.append)
            self.assertEqual(2, result)
            self.assertIn("outside", "\n".join(output))

    def test_normal_first_run_prompts_for_provider_and_key(self):
        class FakeClient:
            def complete_json(self, system_prompt, user_prompt):
                selected = json.loads(user_prompt)["candidate_plans"][0]
                return json.dumps({"action": "plan", "plan": selected}), 5

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as config_directory:
            config_path = Path(config_directory) / "missing.json"
            answers = iter(["1", "", "n"])
            output = []
            with patch("agent.cli.build_client", return_value=FakeClient()):
                result = run_cli([
                    "--project-root", directory, "--config", str(config_path),
                    "--driver", "agent.synthetic:build", "--run-id", "first-run",
                    "--max-iterations", "1",
                ], input_fn=lambda prompt: next(answers),
                   secret_fn=lambda prompt: "first-run-secret",
                   output_fn=output.append)
            self.assertEqual(0, result)
            self.assertFalse(config_path.exists())
            self.assertIn("deepseek", "\n".join(output).lower())
            self.assertNotIn("first-run-secret", "\n".join(output))


if __name__ == "__main__":
    unittest.main()
