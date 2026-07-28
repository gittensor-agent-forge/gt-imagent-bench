from __future__ import annotations

import sys
from pathlib import Path

import pytest

from imagent_bench.agent_loader import load_agent


def _candidate_repository() -> Path:
    # A self-contained fixture, so the engine's tests never depend on the
    # competition repository being checked out beside it.
    return Path(__file__).resolve().parent / "fixtures" / "candidate"


def test_load_agent_does_not_duplicate_sys_path_entries() -> None:
    repository = _candidate_repository()
    if not (repository / "agent" / "agent.py").exists():
        pytest.skip("candidate imagent repository not available")

    repository_path = str(repository)
    before = sys.path.count(repository_path)

    load_agent(repository)
    after_first = sys.path.count(repository_path)

    load_agent(repository)
    after_second = sys.path.count(repository_path)

    # A single load adds at most one entry, and a repeated load must never
    # add a duplicate regardless of any pre-existing sys.path state.
    assert after_first <= before + 1
    assert after_second == after_first
