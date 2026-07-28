from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any


class AgentLoadError(RuntimeError):
    """Raised when a candidate repository cannot be loaded as an agent."""


def resolve_commit_sha(repository: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def prepare_repository(repository: str | Path, checkout_dir: Path | None = None, commit: str | None = None) -> Path:
    repository_text = str(repository)
    local_path = Path(repository_text).expanduser()
    if local_path.exists():
        return local_path.resolve()

    if checkout_dir is None:
        raise AgentLoadError(f"repository path does not exist: {repository}")

    checkout_dir.mkdir(parents=True, exist_ok=True)
    destination = checkout_dir / "candidate"
    subprocess.run(["git", "clone", "--depth", "1", repository_text, str(destination)], check=True)
    if commit:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination, check=True)
        subprocess.run(["git", "checkout", commit], cwd=destination, check=True)
    return destination.resolve()


def load_agent(repository: Path) -> tuple[Any, dict[str, Any]]:
    manifest = _load_manifest(repository)
    entrypoint = str(manifest.get("entrypoint", "")).strip()
    if ":" not in entrypoint:
        raise AgentLoadError("agent manifest entrypoint must use 'module:attribute'")
    module_name, attribute = entrypoint.split(":", 1)
    module_name = module_name.strip()
    attribute = attribute.strip()
    if not module_name or not attribute:
        raise AgentLoadError("agent manifest entrypoint must include module and attribute")

    _clear_candidate_modules(module_name)
    repository_path = str(repository)
    # Avoid unbounded sys.path growth across repeated loads; keep the entry so
    # candidate agents can still perform lazy imports later.
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    try:
        module = importlib.import_module(module_name)
        agent_cls = getattr(module, attribute)
        return agent_cls(), manifest
    except Exception as exc:  # noqa: BLE001
        raise AgentLoadError(f"failed to load agent entrypoint {entrypoint}: {exc}") from exc


def _load_manifest(repository: Path) -> dict[str, Any]:
    for relative in ("agent/agent.yaml", "agent/agent.json"):
        path = repository / relative
        if path.exists():
            if path.suffix == ".json":
                import json

                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise AgentLoadError(f"agent manifest must be an object: {path}")
                return data
            return _parse_simple_yaml(path)
    raise AgentLoadError("candidate repository is missing agent/agent.yaml")


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith(" "):
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        data[key.strip()] = _coerce_scalar(value.strip())
    return data


def _coerce_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value == "":
        return {}
    return value.strip("'\"")


def _clear_candidate_modules(root_module: str) -> None:
    root = root_module.split(".", 1)[0]
    for name in list(sys.modules):
        if name == root or name.startswith(root + "."):
            del sys.modules[name]
