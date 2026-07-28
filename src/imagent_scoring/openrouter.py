from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Vision backends served through OpenRouter: the checklist answerer (S2b) and the
# blind pairwise judge (S4).
#
# Stdlib-only, so these are testable with a mocked opener and carry no supply
# chain into a scored run. Local model backends (detector, OCR, preference) land
# separately; these two are the ones that work without a GPU.

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_VQA_MODEL = "google/gemini-2.5-flash"
DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 3

# The VQA answerer wants the single most likely answer, so it runs greedy.
DEFAULT_VQA_TEMPERATURE = 0.0
# The judge does NOT. Three votes at temperature 0 are three copies of one
# answer, and the majority rule would buy nothing while paying triple. Sampling
# is what makes the votes independent enough for a majority to mean something.
DEFAULT_JUDGE_TEMPERATURE = 0.7

_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot produce a usable answer."""


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name, strict=False)[0] or "image/png"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class OpenRouterClient:
    """Minimal chat-completions client with retry on transient failures."""

    def __init__(
        self,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = 1.0,
        referer: str = "https://tryimagent.com",
        title: str = "imagent scoring",
        opener=urllib.request.urlopen,
        sleep=time.sleep,
    ) -> None:
        self.api_key_env = api_key_env
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.referer = referer
        self.title = title
        self._opener = opener
        self._sleep = sleep

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise OpenRouterError(f"missing API key env var: {self.api_key_env}")

        body = json.dumps(payload).encode("utf-8")
        delay = self.retry_backoff_seconds

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.referer,
                    "X-Title": self.title,
                },
            )
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code in _RETRYABLE and attempt < self.max_retries:
                    self._sleep(delay)
                    delay *= 2
                    continue
                detail = error.read().decode("utf-8", errors="replace")[:400]
                # The key is only ever in the request headers, never echoed here.
                raise OpenRouterError(f"OpenRouter HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt < self.max_retries:
                    self._sleep(delay)
                    delay *= 2
                    continue
                raise OpenRouterError(f"OpenRouter request failed: {error}") from error
            except json.JSONDecodeError as error:
                raise OpenRouterError("OpenRouter response was not valid JSON") from error

        raise OpenRouterError("OpenRouter request failed after retries")


def message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("OpenRouter response carried no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise OpenRouterError("OpenRouter response carried no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise OpenRouterError("OpenRouter response carried no text content")
    return content.strip()


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise OpenRouterError(f"expected a JSON object, got: {content[:200]}") from error
    if not isinstance(parsed, dict):
        raise OpenRouterError("expected a JSON object")
    return parsed


class OpenRouterVqa:
    """Answers one frozen checklist question about one image.

    Deliberately one question per call. Batching would be cheaper, but the
    checklist is dependency-ordered: a child question is only asked once its
    parent passed, and asking "is the hat red?" when there is no hat produces a
    meaningless answer we would then have to pay for and discard.
    """

    def __init__(
        self,
        *,
        client: OpenRouterClient | None = None,
        model: str = DEFAULT_VQA_MODEL,
        temperature: float = DEFAULT_VQA_TEMPERATURE,
    ) -> None:
        self.client = client or OpenRouterClient(title="imagent vqa")
        self.model = model
        self.temperature = temperature
        self._data_urls: dict[str, str] = {}

    def _data_url(self, path: Path) -> str:
        # One image is asked several questions, so encode it once per run rather
        # than once per question.
        key = str(path)
        if key not in self._data_urls:
            self._data_urls[key] = image_data_url(path)
        return self._data_urls[key]

    def answer(self, path: Path, question: str) -> str:
        response = self.client.complete(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": 8,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Answer this question about the image with exactly one "
                                    'word: "yes" or "no". Do not explain.\n\n'
                                    f"Question: {question}"
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": self._data_url(path)}},
                        ],
                    }
                ],
            }
        )
        # The caller normalises: anything that is not a clear yes counts as no,
        # so a hedging answer never earns a candidate a point.
        return message_text(response)


class OpenRouterPairwiseJudge:
    """The blind head-to-head judge.

    It is told nothing about which agent produced which image — the slot
    assignment happens above this layer, in `judge_problem`. It receives the
    objective fact-check results so it judges composition and craft rather than
    re-guessing facts a detector already verified.
    """

    def __init__(
        self,
        *,
        client: OpenRouterClient | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = DEFAULT_JUDGE_TEMPERATURE,
        max_tokens: int = 300,
    ) -> None:
        self.client = client or OpenRouterClient(title="imagent judge")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rationales: list[str] = []

    def compare(
        self,
        *,
        prompt: str,
        image_a: Path,
        image_b: Path,
        notes_a: str = "",
        notes_b: str = "",
    ) -> str:
        instructions = [
            "You are judging two images generated from the same request.",
            "Decide which image is better overall: composition, clarity, craft, and",
            "how well it serves the request. Automated checks have already verified",
            "objective requirements, so do not re-count objects or re-read text.",
            "",
            f"Request: {prompt}",
        ]
        if notes_a or notes_b:
            instructions += [
                "",
                f"Automated check results for image A: {notes_a or 'none'}",
                f"Automated check results for image B: {notes_b or 'none'}",
            ]
        instructions += [
            "",
            'Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "one sentence"}',
            'Use "tie" only when the two are genuinely indistinguishable in quality.',
        ]

        response = self.client.complete(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "\n".join(instructions)},
                            {"type": "text", "text": "Image A:"},
                            {"type": "image_url", "image_url": {"url": image_data_url(image_a)}},
                            {"type": "text", "text": "Image B:"},
                            {"type": "image_url", "image_url": {"url": image_data_url(image_b)}},
                        ],
                    }
                ],
            }
        )

        parsed = parse_json_object(message_text(response))
        self.rationales.append(str(parsed.get("reason", ""))[:300])

        winner = str(parsed.get("winner", "")).strip().casefold()
        if winner in {"a", "image a"}:
            return "A"
        if winner in {"b", "image b"}:
            return "B"
        # Anything else is a tie. An unparseable verdict must never hand over a
        # crown.
        return "tie"
