"""Minimal candidate agent used by the engine's own tests.

The benchmark must be testable without the competition repository checked out,
so this fixture implements the submission contract and nothing else. It calls
OpenRouter through the same code path a real agent would, which lets the tests
exercise the runner end to end against a mocked transport.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


class FixtureAgent:
    def setup(self, config: dict[str, Any], workdir: Path) -> None:
        agent_config = config.get("agent") if isinstance(config, dict) else {}
        agent_config = agent_config if isinstance(agent_config, dict) else {}
        backend = agent_config.get("image_backend")
        self.backend = backend if isinstance(backend, dict) else {}
        self.workdir = Path(workdir)

    def generate(self, case: dict[str, Any]) -> dict[str, Any]:
        api_key = os.environ.get(str(self.backend.get("api_key_env", "OPENROUTER_API_KEY")))
        if not api_key:
            raise RuntimeError("missing OpenRouter API key")

        model = str(self.backend.get("model", "google/gemini-3.1-flash-image"))
        payload = {"model": model, "prompt": str(case.get("prompt", "")), "n": 1}
        request = urllib.request.Request(
            str(self.backend.get("endpoint", "https://openrouter.ai/api/v1/images")),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))

        item = body["data"][0]
        return {
            "image_bytes": base64.b64decode(item["b64_json"]),
            "media_type": str(item.get("media_type", "image/png")),
            "trace": {"agent": "fixture", "model": model, "prompt": payload["prompt"]},
            "metadata": {"model": model, "cost_usd": float(body.get("usage", {}).get("cost", 0.0) or 0.0)},
        }
