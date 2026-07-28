from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BenchmarkConfig


DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    "runtime": {"max_feedback_rounds": 1},
    "agent": {
        "image_backend": {
            "mode": "live",
            "provider": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "endpoint": "https://openrouter.ai/api/v1/images",
            "model": "google/gemini-3.1-flash-image",
            "resolution": "1K",
            "aspect_ratio": "1:1",
            "output_format": "png",
            "send_seed": False,
            "send_output_format": False,
            "timeout_seconds": 240,
            "referer": "https://tryimagent.com",
            "title": "imagent benchmark",
        },
        "verifier": {
            "provider": "openrouter_vision",
            "api_key_env": "OPENROUTER_API_KEY",
            "model": "google/gemini-2.5-flash",
            "timeout_seconds": 180,
            "referer": "https://tryimagent.com",
            "title": "imagent verifier",
        },
    },
    "evaluation": {
        "image_judge": {
            "provider": "openrouter_vision",
            "api_key_env": "OPENROUTER_API_KEY",
            "model": "google/gemini-2.5-flash",
            "rubric_version": "openrouter-vision-v1",
            "temperature": 0,
            "max_tokens": 1200,
            "timeout_seconds": 180,
            "referer": "https://tryimagent.com",
            "title": "imagent benchmark judge",
        }
    },
}

DEFAULT_POLICY: dict[str, Any] = {
    "minimum_score": 85.0,
    "max_failed_cases": 0,
    "max_cost_usd": 999.0,
    "max_latency_p95_ms": 30000.0,
}


def load_config(path: str | Path | None = None) -> BenchmarkConfig:
    if path is None:
        raw: dict[str, Any] = {}
    else:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"benchmark config must be a JSON object: {config_path}")

    agent_config = _deep_merge(DEFAULT_AGENT_CONFIG, dict(raw.get("agent_config", {}) or {}))
    policy = _deep_merge(DEFAULT_POLICY, dict(raw.get("policy", {}) or {}))
    execution = dict(raw.get("execution", {}) or {})

    return BenchmarkConfig(
        benchmark_version=str(raw.get("benchmark_version", "openrouter-vision-v1")),
        dataset_version=str(raw.get("dataset_version", "openrouter-vision-v1.0.0")),
        suite=str(raw.get("suite", "openrouter_vision_v1")),
        agent_config=agent_config,
        policy=policy,
        execution=execution,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
