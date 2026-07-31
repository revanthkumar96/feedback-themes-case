from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


class GroqError(RuntimeError):
    """Raised when Groq cannot return a usable completion."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.error_code = error_code


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    usage: dict[str, int]


PostJson = Callable[
    [str, dict[str, str], dict[str, Any], float],
    dict[str, Any],
]
Sleep = Callable[[float], None]


def _retry_after_seconds_from_text(text: str) -> float | None:
    """Parse provider guidance like 'try again in 240ms', '7.66s', '1m3.5s'."""
    match = re.search(
        r"try again in (?:([0-9]+)m(?!s))?([0-9.]+)\s*(ms|s)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    value = float(match.group(2))
    if match.group(3).lower() == "ms":
        value /= 1000.0
    return minutes * 60.0 + value


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
        raw_body = error.read().decode("utf-8", errors="replace")
        body = raw_body[:1000]
        retry_after: float | None = None
        error_code: str | None = None
        try:
            error_payload = json.loads(raw_body)
            candidate_code = (error_payload.get("error") or {}).get("code")
            if isinstance(candidate_code, str):
                error_code = candidate_code
        except json.JSONDecodeError:
            pass
        header_value = error.headers.get("Retry-After")
        if header_value:
            try:
                retry_after = float(header_value)
            except ValueError:
                pass
        if retry_after is None:
            retry_after = _retry_after_seconds_from_text(body)
        raise GroqError(
            f"Groq returned HTTP {error.code}: {body}",
            status_code=error.code,
            retry_after_seconds=retry_after,
            error_code=error_code,
        ) from error
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
        reasoning_effort: str = "medium",
        max_completion_tokens: int | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: float = 60.0,
        post_json: PostJson = _post_json,
        sleep: Sleep = time.sleep,
        max_rate_limit_retries: int = 6,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be low, medium, or high")
        if max_completion_tokens is not None and max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive")
        if max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries cannot be negative")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens or (
            2048 if reasoning_effort == "low" else 4096
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._post_json = post_json
        self._sleep = sleep
        self._max_rate_limit_retries = max_rate_limit_retries
        self.rate_limit_retry_count = 0

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
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "review_theme_assignments",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        for attempt in range(self._max_rate_limit_retries + 1):
            try:
                response = self._post_json(
                    f"{self._base_url}/chat/completions",
                    {
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "feedback-themes/0.1",
                    },
                    payload,
                    self._timeout_seconds,
                )
                break
            except GroqError as error:
                if (
                    error.status_code != 429
                    or attempt == self._max_rate_limit_retries
                ):
                    raise
                wait_seconds = min(
                    max((error.retry_after_seconds or 1.0) + 0.5, 0.5),
                    60.0,
                )
                self.rate_limit_retry_count += 1
                self._sleep(wait_seconds)
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
