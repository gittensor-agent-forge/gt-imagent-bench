from __future__ import annotations

from pathlib import Path

import pytest

from imagent_scoring import Detection, TextSpan, parse_answer_key, score_facts
from imagent_scoring.facts import MissingBackendError, aggregate
from imagent_scoring.models import CheckResult


class _StubOcr:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self, path: Path) -> list[TextSpan]:
        return [TextSpan(text=self.text)]


class _StubDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, path: Path) -> list[Detection]:
        self.calls += 1
        return self.detections


class _StubLoader:
    def size(self, path: Path) -> tuple[int, int]:
        return (512, 512)


class _StubVqa:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def answer(self, path: Path, question: str) -> str:
        return self.answers.get(question, "no")


def _box(x: float, y: float) -> tuple[float, float, float, float]:
    return (x - 10, y - 10, x + 10, y + 10)


def test_the_worked_example_from_the_build_plan(tmp_path: Path) -> None:
    """three yellow bananas and a blue cup, where the cup came out green."""
    key = parse_answer_key(
        {
            "version": "1.0",
            "problem_id": "geneval-0001",
            "source": "geneval",
            "task": "color_binding",
            "prompt": "a photo of three yellow bananas and a blue cup",
            "requirements": {
                "objects": [
                    {"name": "banana", "count": 3, "color": "yellow"},
                    {"name": "cup", "count": 1, "color": "blue"},
                ]
            },
        }
    )
    detector = _StubDetector(
        [
            Detection(label="banana", box=_box(10, 10), color="yellow"),
            Detection(label="banana", box=_box(40, 10), color="yellow"),
            Detection(label="banana", box=_box(70, 10), color="yellow"),
            Detection(label="cup", box=_box(200, 10), color="green"),
        ]
    )

    report = score_facts(tmp_path / "image.png", key, detector=detector)

    assert report.fact_score == pytest.approx(0.75)
    assert report.passed_count == 3
    assert len(report.checks) == 4
    failed = [check for check in report.checks if not check.passed]
    assert failed[0].label == "colour: cup is blue"
    assert "0 of 1 cup are blue" in failed[0].detail


def test_one_detection_pass_serves_both_objects_and_relations(tmp_path: Path) -> None:
    key = parse_answer_key(
        {
            "version": "1.0",
            "problem_id": "geneval-0002",
            "source": "geneval",
            "prompt": "a cup to the left of a banana",
            "requirements": {
                "objects": [{"name": "cup"}, {"name": "banana"}],
                "relations": [{"subject": "cup", "relation": "left_of", "object": "banana"}],
            },
        }
    )
    detector = _StubDetector(
        [
            Detection(label="cup", box=_box(50, 250)),
            Detection(label="banana", box=_box(450, 250)),
        ]
    )

    report = score_facts(tmp_path / "i.png", key, detector=detector, loader=_StubLoader())

    assert report.fact_score == pytest.approx(1.0)
    # The detector is expensive; it must run once per image, not once per check.
    assert detector.calls == 1


def test_text_and_checklist_are_combined_into_one_score(tmp_path: Path) -> None:
    key = parse_answer_key(
        {
            "version": "1.0",
            "problem_id": "text-0003",
            "source": "text_rendering",
            "prompt": "a badge that says PASS",
            "requirements": {
                "text": ["PASS"],
                "questions": [
                    {"id": "q1", "text": "Is there a badge?"},
                    {"id": "q2", "text": "Is the badge centred?"},
                ],
            },
        }
    )

    report = score_facts(
        tmp_path / "i.png",
        key,
        ocr=_StubOcr("PASS"),
        vqa=_StubVqa({"Is there a badge?": "yes", "Is the badge centred?": "no"}),
    )

    assert report.fact_score == pytest.approx(2 / 3)
    assert [check.kind for check in report.checks] == ["text", "question", "question"]


def test_a_missing_backend_is_an_error_not_a_free_pass(tmp_path: Path) -> None:
    key = parse_answer_key(
        {
            "version": "1.0",
            "problem_id": "text-0004",
            "source": "text_rendering",
            "prompt": "a badge that says PASS",
            "requirements": {"text": ["PASS"]},
        }
    )

    with pytest.raises(MissingBackendError, match="no OCR engine"):
        score_facts(tmp_path / "i.png", key)


def test_aggregate_is_weighted() -> None:
    checks = [
        CheckResult(kind="text", label="a", passed=True, score=1.0, weight=3.0),
        CheckResult(kind="text", label="b", passed=False, score=0.0, weight=1.0),
    ]

    assert aggregate(checks) == pytest.approx(0.75)
    assert aggregate([]) == 0.0


def test_the_report_serialises_for_publication(tmp_path: Path) -> None:
    key = parse_answer_key(
        {
            "version": "1.0",
            "problem_id": "text-0005",
            "source": "text_rendering",
            "prompt": "a badge that says PASS",
            "requirements": {"text": ["PASS"]},
        }
    )

    payload = score_facts(tmp_path / "i.png", key, ocr=_StubOcr("PASS")).to_dict()

    assert payload["problem_id"] == "text-0005"
    assert payload["fact_score"] == 1.0
    assert payload["passed"] == 1
    assert payload["checks"][0]["detail"] == "found exactly"
