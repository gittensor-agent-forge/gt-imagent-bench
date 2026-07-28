from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Any


_IMAGE_EXTENSION_BY_MEDIA_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
}

# A candidate could return an arbitrarily large payload. Cap it so one bad
# submission cannot fill the disk of whatever is running the benchmark.
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class AgentResultError(RuntimeError):
    """Raised when an agent returns a result the benchmark cannot persist."""


def persist_agent_result(result: Any, *, run_id: str, case_output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Validate an agent result and write its artifacts.

    The benchmark owns persistence: the agent hands back bytes and a trace, and
    the engine decides where they go and what they are called. A candidate can
    therefore neither choose its own artifact paths nor overwrite anything
    outside the directory allocated to its case.
    """
    if not isinstance(result, dict):
        raise AgentResultError(f"agent.generate must return a dict, got {type(result).__name__}")

    image_bytes = result.get("image_bytes")
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise AgentResultError("agent result is missing bytes under 'image_bytes'")
    if not image_bytes:
        raise AgentResultError("agent returned an empty image")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise AgentResultError(f"agent image exceeds {MAX_IMAGE_BYTES} bytes")

    media_type = str(result.get("media_type") or "image/png")
    trace = result.get("trace")
    trace = trace if isinstance(trace, dict) else {}
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    safe_id = safe_run_id(run_id)
    images_dir = case_output_dir / "images"
    traces_dir = case_output_dir / "traces"
    images_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{safe_id}{extension_for_media_type(media_type)}"
    trace_path = traces_dir / f"{safe_id}.json"
    image_path.write_bytes(bytes(image_bytes))

    try:
        serialized = json.dumps({"run_id": safe_id, "media_type": media_type, **trace}, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise AgentResultError(f"agent trace is not JSON-serializable: {exc}") from exc
    trace_path.write_text(serialized + "\n", encoding="utf-8")

    return image_path, trace_path, metadata


def safe_run_id(value: Any) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise AgentResultError("run_id must be filename safe")
    return text


def extension_for_media_type(media_type: str) -> str:
    normalized = media_type.split(";", 1)[0].strip().lower()
    if normalized in _IMAGE_EXTENSION_BY_MEDIA_TYPE:
        return _IMAGE_EXTENSION_BY_MEDIA_TYPE[normalized]
    guessed = mimetypes.guess_extension(normalized, strict=False)
    return ".jpg" if guessed == ".jpe" else guessed or ".png"
