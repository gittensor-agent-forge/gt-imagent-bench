from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent_loader import load_agent
from .artifacts import persist_agent_result


DEFAULT_MODEL = "google/gemini-3.1-flash-image"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/images"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("prompt", nargs="+", help='Prompt text. Example: imagent-bench try "A benchmark badge."')
    parser.add_argument("--repository", default=".", help="Agent repository to load (default: current directory)")
    parser.add_argument("--output-dir", default=None, help="Where to write artifacts. Defaults to results/<UTC datetime>")
    parser.add_argument("--capability", default="plan", help="Case capability label")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter image model")
    parser.add_argument("--resolution", default="1K", help="OpenRouter image resolution")
    parser.add_argument("--aspect-ratio", default="1:1", help="OpenRouter image aspect ratio")


def run_once(args: argparse.Namespace) -> int:
    """Run a repository's agent against one ad-hoc prompt.

    This is a development convenience, not a scored run. It exercises the same
    load-and-persist path the benchmark uses, so a candidate that works here
    works under `imagent-bench run`.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"results/{timestamp}").expanduser().resolve()
    repository = Path(args.repository).expanduser().resolve()

    agent, manifest = load_agent(repository)
    agent.setup(_config(args), repository)

    case: dict[str, Any] = {
        "id": timestamp,
        "run_id": timestamp,
        "prompt": " ".join(args.prompt).strip(),
        "capability": args.capability,
        "allowed_tools": [args.capability],
    }
    if not case["prompt"]:
        raise SystemExit('prompt is required, e.g. imagent-bench try "A benchmark badge."')

    image_path, trace_path, metadata = persist_agent_result(
        agent.generate(case), run_id=timestamp, case_output_dir=output_dir
    )
    print(
        json.dumps(
            {
                "agent": manifest.get("id") or manifest.get("name") or "agent",
                "image_path": str(image_path),
                "trace_path": str(trace_path),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "agent": {
            "image_backend": {
                "provider": "openrouter",
                "api_key_env": "OPENROUTER_API_KEY",
                "endpoint": DEFAULT_ENDPOINT,
                "model": args.model,
                "resolution": args.resolution,
                "aspect_ratio": args.aspect_ratio,
                "output_format": "png",
                "send_output_format": False,
                "send_seed": False,
                "timeout_seconds": 240,
                "referer": "https://tryimagent.com",
                "title": "imagent-bench try",
            }
        }
    }
