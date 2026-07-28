from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _safe_int(value: Any) -> int:
    # A non-integer seed should not abort the whole run; fall back to 0.
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_version: str
    dataset_version: str
    suite: str
    agent_config: dict[str, Any]
    policy: dict[str, Any]
    execution: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    prompt: str
    capability: str
    seed: int
    allowed_tools: list[str]
    expected: dict[str, Any] = field(default_factory=dict)
    assets: list[str] = field(default_factory=list)
    search_snapshots: list[str] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BenchmarkCase":
        raw_id = record.get("id") or record.get("run_id")
        if raw_id is None:
            raise ValueError("benchmark case is missing id")
        return cls(
            id=str(raw_id),
            prompt=str(record["prompt"]),
            capability=str(record.get("capability", "plan")),
            seed=_safe_int(record.get("seed", 0)),
            allowed_tools=[str(value) for value in record.get("allowed_tools", [])],
            expected=dict(record.get("expected", {}) or {}),
            assets=[str(value) for value in record.get("assets", []) or []],
            search_snapshots=[str(value) for value in record.get("search_snapshots", []) or []],
            memory=dict(record.get("memory", {}) or {}),
        )

    def to_agent_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "run_id": self.id,
            "prompt": self.prompt,
            "capability": self.capability,
            "seed": self.seed,
            "allowed_tools": self.allowed_tools,
            "expected": self.expected,
        }
        if self.assets:
            payload["assets"] = self.assets
        if self.search_snapshots:
            payload["search_snapshots"] = self.search_snapshots
        if self.memory:
            payload["memory"] = self.memory
        return payload


@dataclass(frozen=True)
class Artifact:
    type: str
    path: str
    sha256: str
    media_type: str | None = None


@dataclass(frozen=True)
class CaseResult:
    id: str
    prompt: str
    capability: str
    status: str
    score: float
    latency_ms: float
    cost_usd: float
    artifacts: list[Artifact]
    dimensions: dict[str, float] | None = None
    judge: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class PolicyResult:
    passed: bool
    reasons: list[str]
    thresholds: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: str
    run_id: str
    repository: str
    commit_sha: str
    pull_request: dict[str, Any] | None
    contributor: dict[str, Any] | None
    benchmark_version: str
    dataset_version: str
    status: str
    overall_score: float
    metrics: dict[str, Any]
    cases: list[CaseResult]
    artifacts: list[Artifact]
    logs: list[Artifact]
    configuration: dict[str, Any]
    policy: PolicyResult
    ranking: dict[str, Any] | None
    started_at: str
    completed_at: str
    execution_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
