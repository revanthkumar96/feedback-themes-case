import unittest

from feedback_themes.groq import GroqClient, GroqError


class GroqClientTests(unittest.TestCase):
    def test_sends_strict_schema_without_exposing_key_in_payload(self) -> None:
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured.update(
                url=url, headers=headers, payload=payload, timeout=timeout
            )
            return {
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": {"content": '{"results":[]}'}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            }

        client = GroqClient("secret-key", post_json=fake_post)
        schema = {"type": "object"}
        completion = client.classify("classify these", schema)

        self.assertEqual(
            "https://api.groq.com/openai/v1/chat/completions", captured["url"]
        )
        response_format = captured["payload"]["response_format"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertIs(schema, response_format["json_schema"]["schema"])
        self.assertEqual("low", captured["payload"]["reasoning_effort"])
        self.assertNotIn("secret-key", str(captured["payload"]))
        self.assertEqual(100, completion.usage["input_tokens"])

    def test_rejects_response_without_content(self) -> None:
        client = GroqClient(
            "secret-key",
            post_json=lambda *_: {"choices": [], "usage": {}},
        )
        with self.assertRaises(GroqError):
            client.classify("prompt", {"type": "object"})
