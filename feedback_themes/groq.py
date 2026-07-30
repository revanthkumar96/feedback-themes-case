from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class GroqError(RuntimeError):
    """Raised when Groq cannot return a usable completion."""


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    usage: dict[str, int]


PostJson = Callable[
    [str, dict[str, str], dict[str, Any], float],
    dict[str, Any],
]


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:1000]
        raise GroqError(f"Groq returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise GroqError(f"Could not reach Groq: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise GroqError("Groq returned a non-JSON response") from error


class GroqClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "openai/gpt-oss-20b",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 60.0,
        post_json: PostJson = _post_json,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._post_json = post_json

    def classify(self, prompt: str, schema: dict[str, Any]) -> Completion:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You classify bank customer feedback against a fixed "
                        "taxonomy. Follow the supplied rules and return only the "
                        "requested structured result."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": "low",
            "max_completion_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "review_theme_assignments",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        response = self._post_json(
            f"{self._base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "feedback-themes-slice1/0.1",
            },
            payload,
            self._timeout_seconds,
        )
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GroqError("Groq response did not contain message content") from error
        if not isinstance(content, str) or not content.strip():
            raise GroqError("Groq returned empty message content")

        raw_usage = response.get("usage") or {}
        usage = {
            "input_tokens": int(raw_usage.get("prompt_tokens") or 0),
            "output_tokens": int(raw_usage.get("completion_tokens") or 0),
            "total_tokens": int(raw_usage.get("total_tokens") or 0),
        }
        returned_model = response.get("model")
        return Completion(
            content=content,
            model=returned_model if isinstance(returned_model, str) else self.model,
            usage=usage,
        )
