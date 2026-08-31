import json
import unittest

from agent.config import LLMConfig
from agent.llm import DeepSeekChatClient, OpenAIResponsesClient, PlannerLLMProvider


class FakeResponse:
    def __init__(self, value):
        self.payload = json.dumps(value).encode("utf-8")

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return FakeResponse(self.response)


class LLMClientTests(unittest.TestCase):
    def test_openai_responses_payload_and_usage(self):
        transport = RecordingTransport({
            "output_text": '{"action":"stop","plan":null}',
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        })
        config = LLMConfig("openai", "gpt-test", "https://api.openai.com/v1", "openai-secret")
        text, tokens = OpenAIResponsesClient(config, transport=transport).complete_json("system", "user")
        request, timeout = transport.requests[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("https://api.openai.com/v1/responses", request.full_url)
        self.assertEqual("gpt-test", payload["model"])
        self.assertEqual({"type": "json_object"}, payload["text"]["format"])
        self.assertFalse(payload["store"])
        self.assertEqual(10, tokens)
        self.assertIn('"stop"', text)
        self.assertNotIn("openai-secret", request.data.decode("utf-8"))

    def test_openai_nested_output_fallback(self):
        transport = RecordingTransport({
            "output": [{"content": [{"type": "output_text", "text": "{}"}]}],
            "usage": {"total_tokens": 4},
        })
        config = LLMConfig("openai", "gpt-test", "https://api.openai.com/v1", "key")
        text, tokens = OpenAIResponsesClient(config, transport=transport).complete_json("s", "u")
        self.assertEqual("{}", text)
        self.assertEqual(4, tokens)

    def test_openai_uses_structured_output_schema_when_provided(self):
        transport = RecordingTransport({
            "output_text": '{"action":"stop","plan":null}',
            "usage": {"total_tokens": 2},
        })
        config = LLMConfig("openai", "gpt-test", "https://api.openai.com/v1", "key")
        schema = {
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        }
        OpenAIResponsesClient(config, transport=transport).complete_json(
            "s", "u", response_schema=schema,
        )
        payload = json.loads(transport.requests[0][0].data.decode("utf-8"))
        response_format = payload["text"]["format"]
        self.assertEqual("json_schema", response_format["type"])
        self.assertEqual(schema, response_format["schema"])
        self.assertFalse(response_format["strict"])

    def test_deepseek_chat_payload_and_usage(self):
        transport = RecordingTransport({
            "choices": [{"message": {"content": '{"action":"stop","plan":null}'}}],
            "usage": {"total_tokens": 21},
        })
        config = LLMConfig("deepseek", "deepseek-test", "https://api.deepseek.com", "deepseek-secret")
        text, tokens = DeepSeekChatClient(config, transport=transport).complete_json("system", "user")
        request, timeout = transport.requests[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("https://api.deepseek.com/chat/completions", request.full_url)
        self.assertEqual("deepseek-test", payload["model"])
        self.assertEqual({"type": "json_object"}, payload["response_format"])
        self.assertEqual(21, tokens)
        self.assertIn('"stop"', text)

    def test_planner_provider_passes_tools_and_candidates_without_key(self):
        class Client:
            def complete_json(self, system, user, response_schema=None):
                self.system, self.user = system, user
                self.response_schema = response_schema
                return '{"action":"stop","plan":null}', 1

        client = Client()
        provider = PlannerLLMProvider(
            client, ["train"], [{"requested_tool": "train"}],
            search_context={"search_space": {"train": {"lr": [0.01]}}},
        )
        _, tokens = provider({"history": []})
        body = json.loads(client.user)
        self.assertEqual(["train"], body["registered_tools"])
        self.assertEqual([0.01], body["search_context"]["search_space"]["train"]["lr"])
        self.assertIn("Do not stop merely", client.system)
        self.assertEqual("object", client.response_schema["type"])
        self.assertEqual(1, tokens)


if __name__ == "__main__":
    unittest.main()
