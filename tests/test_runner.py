from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from imagent_bench.runner import (
    BenchmarkRunError,
    _case_status,
    _normalize_repository_identifier,
    _percentile,
    _ranking,
    run,
)


def _candidate_repository() -> Path:
    # A self-contained fixture, so the engine's tests never depend on the
    # competition repository being checked out beside it.
    return Path(__file__).resolve().parent / "fixtures" / "candidate"


@pytest.fixture
def openrouter_http_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_urlopen(request: Any, *args: Any, **kwargs: Any) -> "_FakeResponse":
        url = str(getattr(request, "full_url", ""))
        if url.endswith("/images"):
            return _FakeResponse(
                {
                    "created": 1783296000,
                    "data": [
                        {
                            "b64_json": base64.b64encode(b"fake-png-bytes").decode("ascii"),
                            "media_type": "image/png",
                        }
                    ],
                    "usage": {"cost": 0.001},
                }
            )
        return _FakeResponse(
            {
                "model": "google/gemini-2.5-flash",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "scores": {
                                        "prompt_alignment": 100,
                                        "visual_quality": 100,
                                        "aesthetics": 100,
                                        "text_accuracy": 100,
                                        "layout_and_composition": 100,
                                        "realism": 100,
                                    },
                                    "overall_score": 100,
                                    "rationale": "offline OpenRouter fixture",
                                }
                            )
                        }
                    }
                ],
                "usage": {"cost": 0.001},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_runner_executes_local_imagent_and_writes_report(openrouter_http_mock: None, tmp_path: Path) -> None:
    repository = _candidate_repository()
    config = Path(__file__).resolve().parents[1] / "configs" / "openrouter-vision-benchmark.json"

    result = run(repository=repository, config=config, output_dir=tmp_path)

    report_path = tmp_path / "benchmark-report.json"
    summary_path = tmp_path / "benchmark-summary.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert result.overall_score == 100.0
    assert report["schema_version"] == "1.0"
    # The identity is derived from the candidate's git origin, so it depends on
    # which repository the fixture happens to sit in. Assert it resolved to
    # something; _normalize_repository_identifier is tested directly below.
    assert isinstance(report["repository"], str) and report["repository"]
    assert report["metrics"]["case_count"] == 3
    assert report["policy"]["passed"] is True
    assert report["ranking"]["baseline_score"] is None
    assert summary_path.exists()


def test_runner_marks_merge_eligible_when_score_improves_baseline(
    openrouter_http_mock: None, tmp_path: Path
) -> None:
    repository = _candidate_repository()
    config = Path(__file__).resolve().parents[1] / "configs" / "openrouter-vision-benchmark.json"

    result = run(
        repository=repository,
        config=config,
        output_dir=tmp_path,
        baseline_score=95.0,
        baseline_commit="baseline123",
    )

    report = json.loads((tmp_path / "benchmark-report.json").read_text(encoding="utf-8"))

    assert result.status == "pass"
    assert report["ranking"]["baseline_score"] == 95.0
    assert report["ranking"]["baseline_commit"] == "baseline123"
    assert report["ranking"]["delta"] == 5.0
    assert report["ranking"]["label"] == "improvement-strong"
    assert report["ranking"]["merge_eligible"] is True


def test_runner_marks_judged_case_fail_when_score_is_below_case_minimum() -> None:
    status = _case_status(
        judge_result={"overall_score": 74.9},
        expected={"minimum_score": 75.0},
        policy_minimum=0.0,
    )

    assert status == "fail"


def test_runner_falls_back_to_policy_minimum_when_case_declares_none() -> None:
    # A judged case with no floor of its own must not pass by default; that is
    # how the retired official suite reported every case as a pass.
    assert (
        _case_status(judge_result={"overall_score": 40.0}, expected={}, policy_minimum=75.0)
        == "fail"
    )
    assert (
        _case_status(judge_result={"overall_score": 90.0}, expected={}, policy_minimum=75.0)
        == "pass"
    )


def test_runner_reads_pull_request_metadata_from_github_event(
    monkeypatch, openrouter_http_mock: None, tmp_path: Path  # noqa: ANN001
) -> None:
    repository = _candidate_repository()
    config = Path(__file__).resolve().parents[1] / "configs" / "openrouter-vision-benchmark.json"
    event_path = tmp_path / "github-event.json"
    event_path.write_text(
        json.dumps(
            {
                "number": 77,
                "pull_request": {
                    "number": 77,
                    "title": "feat: benchmark metadata",
                    "state": "open",
                    "html_url": "https://github.com/gittensor-agent-forge/gt-imagent/pull/77",
                    "merged_at": None,
                    "closed_at": None,
                    "user": {
                        "login": "mitchelltop",
                        "avatar_url": "https://avatars.example.test/mitchelltop",
                        "html_url": "https://github.com/mitchelltop",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    result = run(repository=repository, config=config, output_dir=tmp_path / "report")
    report = json.loads((tmp_path / "report" / "benchmark-report.json").read_text(encoding="utf-8"))

    assert result.pull_request == {
        "number": 77,
        "title": "feat: benchmark metadata",
        "state": "open",
        "html_url": "https://github.com/gittensor-agent-forge/gt-imagent/pull/77",
        "merged_at": None,
        "closed_at": None,
    }
    assert result.contributor == {
        "login": "mitchelltop",
        "name": None,
        "avatar_url": "https://avatars.example.test/mitchelltop",
        "html_url": "https://github.com/mitchelltop",
    }
    assert report["pull_request"]["number"] == 77
    assert report["contributor"]["login"] == "mitchelltop"


def test_normalize_repository_identifier_handles_common_github_urls() -> None:
    assert _normalize_repository_identifier("https://github.com/gittensor-agent-forge/gt-imagent.git") == "gittensor-agent-forge/gt-imagent"
    assert _normalize_repository_identifier("git@github.com:imagent-ai/imagent-bench.git") == "imagent-ai/imagent-bench"


def test_normalize_repository_identifier_strips_embedded_credentials() -> None:
    token_url = "https://ghp_secretToken123@github.com/o/r.git"
    userpass_url = "https://u:p@github.com/o/r.git"

    assert _normalize_repository_identifier(token_url) == "o/r"
    assert _normalize_repository_identifier(userpass_url) == "o/r"
    # No credential substring may survive into the normalized identifier.
    assert "ghp_secretToken123" not in _normalize_repository_identifier(token_url)
    assert "@" not in _normalize_repository_identifier(token_url)
    assert "u:p" not in _normalize_repository_identifier(userpass_url)
    # SCP form must remain unaffected by the credential stripping.
    assert _normalize_repository_identifier("git@github.com:o/r.git") == "o/r"


def test_percentile_handles_p100_and_p95_without_indexerror() -> None:
    values = [float(v) for v in range(1, 101)]

    # p=100 previously raised IndexError; it must now clamp to the top cut point.
    p100 = _percentile(values, 100)
    p95 = _percentile(values, 95)

    assert isinstance(p100, float)
    assert p95 <= p100 <= 100.0
    assert p95 == pytest.approx(95.0, abs=1.0)


def test_ranking_labels_small_positive_delta_as_non_regression() -> None:
    ranking = _ranking({}, candidate_score=95.5, baseline_score=95.0, baseline_commit="abc")

    assert ranking["delta"] == 0.5
    assert ranking["label"] == "no-significant-change"
    assert ranking["merge_eligible"] is False


def test_ranking_labels_negative_delta_as_regression() -> None:
    ranking = _ranking({}, candidate_score=94.0, baseline_score=95.0, baseline_commit="abc")

    assert ranking["delta"] == -1.0
    assert ranking["label"] == "score-regression"


def test_runner_rejects_local_repository_commit_mismatch(tmp_path: Path) -> None:
    repository = _candidate_repository()
    config = Path(__file__).resolve().parents[1] / "configs" / "openrouter-vision-benchmark.json"

    with pytest.raises(BenchmarkRunError, match="local repository HEAD does not match --commit"):
        run(repository=repository, commit="deadbeef", config=config, output_dir=tmp_path)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")
