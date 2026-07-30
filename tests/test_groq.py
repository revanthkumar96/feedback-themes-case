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
        self.assertEqual("medium", captured["payload"]["reasoning_effort"])
        self.assertEqual(4096, captured["payload"]["max_completion_tokens"])
        self.assertNotIn("secret-key", str(captured["payload"]))
        self.assertEqual(100, completion.usage["input_tokens"])

    def test_rejects_response_without_content(self) -> None:
        client = GroqClient(
            "secret-key",
            post_json=lambda *_: {"choices": [], "usage": {}},
        )
        with self.assertRaises(GroqError):
            client.classify("prompt", {"type": "object"})

    def test_allows_explicit_completion_budget(self) -> None:
        client = GroqClient(
            "secret-key",
            max_completion_tokens=5000,
            post_json=lambda *_: {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            },
        )
        self.assertEqual(5000, client.max_completion_tokens)

    def test_retries_only_rate_limits_using_server_delay(self) -> None:
        calls = 0
        waits = []

        def rate_limited_once(*_):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise GroqError(
                    "rate limited",
                    status_code=429,
                    retry_after_seconds=2.0,
                )
            return {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }

        client = GroqClient(
            "secret-key",
            post_json=rate_limited_once,
            sleep=waits.append,
        )
        client.classify("prompt", {"type": "object"})
        self.assertEqual(2, calls)
        self.assertEqual([2.5], waits)
        self.assertEqual(1, client.rate_limit_retry_count)
