from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent_loader import load_agent, prepare_repository, resolve_commit_sha
from .artifacts import persist_agent_result
from .config import load_config
from .models import Artifact, BenchmarkResult, CaseResult
from .policy import evaluate_policy
from .reporting import artifact_for, write_markdown_summary, write_report
from .scoring import evaluate_case
from .suite import load_cases, suite_dir


class BenchmarkRunError(RuntimeError):
    """Raised when the benchmark cannot complete."""


def run(
    repository: str | Path,
    commit: str | None = None,
    config: str | Path | None = None,
    output_dir: str | Path = "benchmark-output",
    baseline_score: float | None = None,
    baseline_commit: str | None = None,
) -> BenchmarkResult:
    started = time.perf_counter()
    started_at = _utc_now()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logs_dir = output_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "benchmark.log"

    benchmark_config = load_config(config)
    checkout_dir = output_path / "checkout"
    candidate_repo = prepare_repository(repository, checkout_dir=checkout_dir, commit=commit)
    commit_sha = resolve_commit_sha(candidate_repo)
    _validate_local_repository_commit(repository, commit, commit_sha)
    repository_identity = _repository_identity(repository, candidate_repo)
    pull_request_metadata, contributor_metadata = _github_report_metadata()

    _append_log(log_path, f"repository={repository_identity}")
    _append_log(log_path, f"repository_path={candidate_repo}")
    _append_log(log_path, f"commit={commit_sha}")
    _append_log(log_path, f"suite={benchmark_config.suite}")

    try:
        agent, manifest = load_agent(candidate_repo)
        agent.setup(benchmark_config.agent_config, suite_dir(benchmark_config.suite))
    except Exception as exc:  # noqa: BLE001
        raise BenchmarkRunError(f"failed to prepare agent: {exc}") from exc

    cases = load_cases(benchmark_config.suite)
    judge_config = _judge_config(benchmark_config.agent_config)
    case_results: list[CaseResult] = []
    top_level_artifacts: list[Artifact] = []

    for case in cases:
        case_output_dir = output_path / "cases" / case.id
        case_output_dir.mkdir(parents=True, exist_ok=True)
        _append_log(log_path, f"running case={case.id}")
        case_started = time.perf_counter()
        try:
            output = agent.generate(case.to_agent_payload())
            image_path, trace_path, metadata = persist_agent_result(
                output, run_id=case.id, case_output_dir=case_output_dir
            )
            # The engine times the call. A candidate does not get to report its
            # own latency.
            latency_ms = round((time.perf_counter() - case_started) * 1000.0, 3)
            score, judge_result = evaluate_case(
                image_path,
                prompt=case.prompt,
                judge_config=judge_config,
            )
            status = _case_status(
                judge_result=judge_result,
                expected=case.expected,
                policy_minimum=float(benchmark_config.policy.get("minimum_score", 0.0) or 0.0),
            )
            cost_usd = float(metadata.get("cost_usd", 0.0) or 0.0) + float(judge_result.get("cost_usd", 0.0) or 0.0)
            artifacts = [
                artifact_for(image_path, output_path, "image"),
                artifact_for(trace_path, output_path, "trace"),
            ]
            top_level_artifacts.extend(artifacts)
            case_results.append(
                CaseResult(
                    id=case.id,
                    prompt=case.prompt,
                    capability=case.capability,
                    status=status,
                    score=score,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    artifacts=artifacts,
                    dimensions=judge_result.get("dimensions") if judge_result else None,
                    judge=judge_result.get("judge") if judge_result else None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _append_log(log_path, f"case={case.id} error={exc}")
            case_results.append(
                CaseResult(
                    id=case.id,
                    prompt=case.prompt,
                    capability=case.capability,
                    status="error",
                    score=0.0,
                    latency_ms=round((time.perf_counter() - case_started) * 1000.0, 3),
                    cost_usd=0.0,
                    artifacts=[],
                    error=str(exc),
                )
            )

    metrics = _metrics(case_results)
    ranking = _ranking(
        benchmark_config.policy,
        candidate_score=metrics["overall_score"],
        baseline_score=baseline_score if baseline_score is not None else _env_float("IMAGENT_BASELINE_SCORE"),
        baseline_commit=baseline_commit or os.environ.get("IMAGENT_BASELINE_COMMIT"),
    )
    policy_result = evaluate_policy(
        benchmark_config.policy,
        overall_score=metrics["overall_score"],
        failed_case_count=metrics["failed_case_count"],
        cost_usd=metrics["cost_usd"],
        latency_p95_ms=metrics["latency_p95_ms"],
        ranking=ranking,
    )
    status = "pass" if policy_result.passed else "fail"
    completed_at = _utc_now()

    log_artifact = artifact_for(log_path, output_path, "log")
    result = BenchmarkResult(
        schema_version="1.0",
        run_id=str(uuid.uuid4()),
        repository=repository_identity,
        commit_sha=commit_sha,
        pull_request=pull_request_metadata,
        contributor=contributor_metadata,
        benchmark_version=benchmark_config.benchmark_version,
        dataset_version=benchmark_config.dataset_version,
        status=status,
        overall_score=metrics["overall_score"],
        metrics=metrics,
        cases=case_results,
        artifacts=top_level_artifacts,
        logs=[log_artifact],
        configuration={
            "agent_manifest": manifest,
            "agent_config": benchmark_config.agent_config,
            "execution": benchmark_config.execution,
        },
        policy=policy_result,
        ranking=ranking,
        started_at=started_at,
        completed_at=completed_at,
        execution_time_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )
    write_report(result, output_path)
    write_markdown_summary(result, output_path)
    return result


def _judge_config(agent_config: dict[str, Any]) -> dict[str, Any]:
    evaluation = agent_config.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    judge_config = evaluation.get("image_judge")
    return judge_config if isinstance(judge_config, dict) else {}


def _case_status(*, judge_result: dict[str, Any], expected: dict[str, Any], policy_minimum: float) -> str:
    # No judge means smoke mode: producing a valid image is the whole bar, and
    # persist_agent_result already rejected anything unusable.
    if not judge_result:
        return "pass"

    # A judged case without its own floor falls back to the policy minimum.
    # Defaulting to "pass" here would let a case score 3/100 and still count as
    # a success, which is how the old official suite reported 5/5 passes.
    minimum_score = _case_minimum_score(expected)
    if minimum_score is None:
        minimum_score = policy_minimum
    return "pass" if float(judge_result.get("overall_score", 0.0) or 0.0) >= minimum_score else "fail"


def _case_minimum_score(expected: dict[str, Any]) -> float | None:
    raw_value = expected.get("minimum_score")
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _metrics(cases: list[CaseResult]) -> dict[str, Any]:
    scores = [case.score for case in cases]
    latencies = [case.latency_ms for case in cases]
    cost_usd = round(sum(case.cost_usd for case in cases), 6)
    failed = sum(1 for case in cases if case.status != "pass")
    return {
        "overall_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "case_count": len(cases),
        "failed_case_count": failed,
        "latency_ms": {
            "min": round(min(latencies), 3) if latencies else 0.0,
            "max": round(max(latencies), 3) if latencies else 0.0,
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        },
        "latency_p95_ms": _percentile(latencies, 95),
        "cost_usd": cost_usd,
    }


def _ranking(
    policy: dict[str, Any],
    *,
    candidate_score: float,
    baseline_score: float | None,
    baseline_commit: str | None,
) -> dict[str, Any]:
    thresholds = _improvement_thresholds(policy)
    minimum_merge_improvement = float(policy.get("minimum_merge_improvement", thresholds.get("minor", 0.0)))
    if baseline_score is None:
        return {
            "baseline_score": None,
            "baseline_commit": baseline_commit,
            "candidate_score": candidate_score,
            "delta": None,
            "label": "baseline-unavailable",
            "merge_eligible": False,
            "minimum_merge_improvement": minimum_merge_improvement,
            "thresholds": thresholds,
        }

    delta = round(candidate_score - baseline_score, 6)
    # Only a negative delta is a regression. A positive or zero delta that is too
    # small to clear the smallest improvement threshold is a non-regression.
    label = "score-regression" if delta < 0 else "no-significant-change"
    for name, threshold in sorted(thresholds.items(), key=lambda item: item[1]):
        if delta >= threshold:
            label = f"improvement-{name}"
    return {
        "baseline_score": baseline_score,
        "baseline_commit": baseline_commit,
        "candidate_score": candidate_score,
        "delta": delta,
        "label": label,
        "merge_eligible": delta >= minimum_merge_improvement,
        "minimum_merge_improvement": minimum_merge_improvement,
        "thresholds": thresholds,
    }


def _improvement_thresholds(policy: dict[str, Any]) -> dict[str, float]:
    raw = policy.get("improvement_thresholds")
    if not isinstance(raw, dict):
        raw = {"minor": 1.0, "strong": 3.0, "major": 7.0}
    thresholds: dict[str, float] = {}
    for name, value in raw.items():
        try:
            thresholds[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return thresholds or {"minor": 1.0, "strong": 3.0, "major": 7.0}


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    quantiles = statistics.quantiles(ordered, n=100, method="inclusive")
    # quantiles has 99 cut points (indices 0..98); clamp so p<=0 and p=100 stay in range.
    index = min(max(percentile - 1, 0), len(quantiles) - 1)
    return round(float(quantiles[index]), 3)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _utc_now(), "message": message}) + "\n")


def _validate_local_repository_commit(repository: str | Path, requested_commit: str | None, actual_commit: str) -> None:
    if not requested_commit:
        return
    local_candidate = Path(str(repository)).expanduser()
    if not local_candidate.exists():
        return
    if _commit_matches(actual_commit, requested_commit):
        return
    raise BenchmarkRunError(
        "local repository HEAD does not match --commit; checkout the desired commit first or use a git URL repository"
    )


def _commit_matches(actual_commit: str, requested_commit: str) -> bool:
    actual = actual_commit.strip()
    requested = requested_commit.strip()
    if not actual or not requested:
        return False
    return actual == requested or actual.startswith(requested) or requested.startswith(actual)


def _repository_identity(repository: str | Path, candidate_repo: Path) -> str:
    requested = str(repository).strip()
    local_candidate = Path(requested).expanduser()
    if local_candidate.exists():
        remote_url = _git_origin_url(candidate_repo)
        if remote_url:
            return _normalize_repository_identifier(remote_url)
    normalized = _normalize_repository_identifier(requested)
    return normalized or str(candidate_repo)


def _git_origin_url(repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    remote_url = completed.stdout.strip()
    return remote_url or None


def _normalize_repository_identifier(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = _strip_url_credentials(text)
    for pattern in (
        r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
        r"^ssh://git@github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
        r"^git@github\.com:(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?$",
        r"^github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
    ):
        match = re.match(pattern, text)
        if match:
            return f"{match.group('owner')}/{match.group('repo')}"
    return text.removesuffix(".git")


def _strip_url_credentials(text: str) -> str:
    # Remove embedded userinfo (e.g. tokens or user:pass) from a URL authority so
    # credentials never leak into the normalized identifier or the report. Only
    # scheme-based URLs carry userinfo before an "@" in the authority; the SCP
    # form (git@github.com:owner/repo.git) has no "//" and is left untouched.
    match = re.match(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<rest>.*)$", text)
    if not match:
        return text
    rest = match.group("rest")
    authority, sep, tail = rest.partition("/")
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    return f"{match.group('scheme')}{authority}{sep}{tail}"


def _github_report_metadata() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    payload = _github_event_payload()
    if not payload:
        return None, None

    pull_request = payload.get("pull_request")
    if isinstance(pull_request, dict):
        contributor = _contributor_metadata(pull_request.get("user"))
        return _pull_request_metadata(pull_request, payload), contributor

    contributor = _contributor_metadata(payload.get("sender"))
    return None, contributor


def _github_event_payload() -> dict[str, Any] | None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return None
    path = Path(event_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _pull_request_metadata(pull_request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    number = pull_request.get("number", payload.get("number"))
    try:
        number_value = int(number)
    except (TypeError, ValueError):
        return None

    state = str(pull_request.get("state", "")).strip().lower()
    if pull_request.get("merged_at"):
        state = "merged"
    elif state not in {"open", "closed"}:
        state = "closed"

    return {
        "number": number_value,
        "title": str(pull_request.get("title", "")).strip(),
        "state": state,
        "html_url": _optional_string(pull_request.get("html_url")),
        "merged_at": _optional_string(pull_request.get("merged_at")),
        "closed_at": _optional_string(pull_request.get("closed_at")),
    }


def _contributor_metadata(raw_user: Any) -> dict[str, Any] | None:
    if not isinstance(raw_user, dict):
        return None
    login = _optional_string(raw_user.get("login"))
    if not login:
        return None
    return {
        "login": login,
        "name": _optional_string(raw_user.get("name")),
        "avatar_url": _optional_string(raw_user.get("avatar_url")),
        "html_url": _optional_string(raw_user.get("html_url")),
    }


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
