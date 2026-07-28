from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagent_scoring import Detection, parse_answer_key, score_facts
from imagent_scoring.facts import MissingBackendError
from imagent_scoring.openrouter import OpenRouterClient, OpenRouterObjectVerifier


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _verifier(*answers: str, capture: list | None = None) -> OpenRouterObjectVerifier:
    queue = list(answers)

    def opener(request, timeout):
        if capture is not None:
            capture.append(json.loads(request.data))
        return _Response(_reply(queue.pop(0) if queue else "0"))

    return OpenRouterObjectVerifier(client=OpenRouterClient(opener=opener, sleep=lambda _: None))


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


KEY = {
    "version": "1.0",
    "problem_id": "geneval-0001",
    "source": "geneval",
    "prompt": "a photo of three yellow bananas and a blue cup",
    "requirements": {
        "objects": [
            {"name": "banana", "count": 3, "color": "yellow"},
            {"name": "cup", "count": 1, "color": "blue"},
        ]
    },
}


# --- the verifier itself -----------------------------------------------------


def test_a_count_is_read_out_of_the_answer(image: Path) -> None:
    assert _verifier("3").count(image, "banana") == 3
    assert _verifier("There are 2 cups.").count(image, "cup") == 2


def test_an_unparseable_count_is_zero_not_a_pass(image: Path) -> None:
    # "We could not establish it" must never be read as success.
    assert _verifier("I am not sure").count(image, "banana") == 0


def test_a_colour_is_normalised(image: Path) -> None:
    assert _verifier("Yellow.").colour(image, "banana") == "yellow"
    assert _verifier("none").colour(image, "banana") == ""
    assert _verifier("unknown").colour(image, "cup") == ""


def test_a_relation_is_asked_in_plain_english(image: Path) -> None:
    captured: list = []
    verifier = _verifier("yes", capture=captured)

    assert verifier.relation(image, "cup", "left_of", "banana") is True
    assert "to the left of" in captured[0]["messages"][0]["content"][0]["text"]


def test_the_image_is_encoded_once_across_questions(image: Path, monkeypatch) -> None:
    reads = {"n": 0}
    original = Path.read_bytes

    def counting(self):
        if self.name == "candidate.png":
            reads["n"] += 1
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    verifier = _verifier("3", "yellow", "yes")
    verifier.count(image, "banana")
    verifier.colour(image, "banana")
    verifier.relation(image, "cup", "left_of", "banana")

    assert reads["n"] == 1


# --- grading through the verifier -------------------------------------------


def test_grading_works_with_no_detector(image: Path) -> None:
    # 3 bananas, yellow; 1 cup, but green.
    verifier = _verifier("3", "yellow", "1", "green")

    report = score_facts(image, parse_answer_key(KEY), verifier=verifier)

    assert report.fact_score == pytest.approx(0.75)
    failed = [check for check in report.checks if not check.passed]
    assert failed[0].label == "colour: cup is blue"


def test_verifier_results_are_marked_non_deterministic(image: Path) -> None:
    # A detector's answer can be re-derived by anyone; this one cannot, and a
    # report must never let a reader confuse the two.
    report = score_facts(image, parse_answer_key(KEY), verifier=_verifier("3", "yellow", "1", "blue"))

    assert all("non-deterministic" in check.detail for check in report.checks)


def test_a_detector_wins_when_both_are_supplied(image: Path) -> None:
    class _Detector:
        def detect(self, path):
            return [
                Detection("banana", (0, 0, 10, 10), color="yellow"),
                Detection("banana", (20, 0, 30, 10), color="yellow"),
                Detection("banana", (40, 0, 50, 10), color="yellow"),
                Detection("cup", (60, 0, 70, 10), color="blue"),
            ]

    report = score_facts(
        image, parse_answer_key(KEY), detector=_Detector(), verifier=_verifier("0", "none")
    )

    assert report.fact_score == pytest.approx(1.0)
    assert not any("non-deterministic" in check.detail for check in report.checks)


def test_neither_backend_is_still_an_error(image: Path) -> None:
    with pytest.raises(MissingBackendError, match="neither a detector nor an object verifier"):
        score_facts(image, parse_answer_key(KEY))


def test_a_verifier_failure_fails_the_check_rather_than_the_run(image: Path) -> None:
    class _Broken:
        def count(self, path, name):
            raise RuntimeError("provider down")

        def colour(self, path, name):
            return ""

        def relation(self, path, subject, relation, obj):
            return False

    report = score_facts(image, parse_answer_key(KEY), verifier=_Broken())

    assert report.fact_score == 0.0
    assert all("verifier error" in check.detail for check in report.checks)
