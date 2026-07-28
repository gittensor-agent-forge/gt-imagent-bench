from __future__ import annotations

from pathlib import Path

import pytest

from imagent_scoring.checklist import check_questions, order_questions
from imagent_scoring.models import (
    ChecklistQuestion,
    Detection,
    ObjectRequirement,
    RelationRequirement,
    TextRequirement,
    TextSpan,
)
from imagent_scoring.object_check import check_objects, check_relations
from imagent_scoring.text_check import check_text, normalize, similarity


# --- text ------------------------------------------------------------------


def test_required_text_is_found_regardless_of_case_and_punctuation() -> None:
    results = check_text(
        [TextRequirement(value="Signal Review Board")],
        [TextSpan(text="SIGNAL REVIEW BOARD!")],
    )

    assert results[0].passed
    assert results[0].score == 1.0


def test_a_misspelling_earns_partial_credit_but_never_passes() -> None:
    results = check_text(
        [TextRequirement(value="Signal Review Board")],
        [TextSpan(text="Signal Reveiw Board")],
    )

    assert not results[0].passed
    assert 0.8 <= results[0].score < 1.0
    assert "near miss" in results[0].detail


def test_missing_text_scores_zero() -> None:
    results = check_text(
        [TextRequirement(value="Signal Review Board")],
        [TextSpan(text="completely unrelated caption")],
    )

    assert not results[0].passed
    assert results[0].score == 0.0


def test_exact_match_mode_rejects_extra_text() -> None:
    requirement = TextRequirement(value="PASS", match="exact")

    assert check_text([requirement], [TextSpan(text="PASS")])[0].passed
    assert not check_text([requirement], [TextSpan(text="PASS OR FAIL")])[0].passed


def test_normalisation_folds_accents_and_whitespace() -> None:
    assert normalize("  Café   RÉSUMÉ!  ") == "cafe resume"
    assert similarity("abc", "abc") == 1.0
    assert similarity("abc", "abd") == pytest.approx(2 / 3)


# --- objects ---------------------------------------------------------------


def _box(x: float, y: float) -> tuple[float, float, float, float]:
    return (x - 10, y - 10, x + 10, y + 10)


def test_object_count_must_match_exactly() -> None:
    detections = [Detection(label="banana", box=_box(10, 10)) for _ in range(3)]

    results = check_objects([ObjectRequirement(name="banana", count=3)], detections)

    assert results[0].passed
    assert results[0].score == 1.0


def test_a_miscount_earns_partial_credit_but_never_passes() -> None:
    detections = [Detection(label="banana", box=_box(10, 10)) for _ in range(2)]

    results = check_objects([ObjectRequirement(name="banana", count=3)], detections)

    assert not results[0].passed
    assert 0.0 < results[0].score < 1.0
    assert "expected 3, found 2" in results[0].detail


def test_low_confidence_detections_are_ignored() -> None:
    detections = [Detection(label="cup", box=_box(10, 10), confidence=0.2)]

    results = check_objects([ObjectRequirement(name="cup")], detections)

    assert not results[0].passed


def test_colour_is_checked_from_the_detection_not_guessed() -> None:
    detections = [
        Detection(label="cup", box=_box(10, 10), color="blue"),
        Detection(label="banana", box=_box(50, 10), color="green"),
    ]

    results = check_objects(
        [
            ObjectRequirement(name="cup", count=1, color="blue"),
            ObjectRequirement(name="banana", count=1, color="yellow"),
        ],
        detections,
    )
    colour_checks = [check for check in results if check.kind == "object" and "colour" in check.label]

    assert colour_checks[0].passed
    assert not colour_checks[1].passed
    assert "are yellow" in colour_checks[1].detail


# --- relations -------------------------------------------------------------


def test_spatial_relation_is_decided_by_box_geometry() -> None:
    detections = [
        Detection(label="cup", box=_box(20, 100)),
        Detection(label="banana", box=_box(180, 100)),
    ]

    results = check_relations(
        [RelationRequirement(subject="cup", relation="left_of", object="banana")],
        detections,
        image_width=200,
        image_height=200,
    )

    assert results[0].passed


def test_a_relation_within_the_margin_is_not_satisfied() -> None:
    detections = [
        Detection(label="cup", box=_box(99, 100)),
        Detection(label="banana", box=_box(101, 100)),
    ]

    results = check_relations(
        [RelationRequirement(subject="cup", relation="left_of", object="banana")],
        detections,
        image_width=200,
        image_height=200,
    )

    assert not results[0].passed


def test_a_relation_with_a_missing_object_fails_and_says_which() -> None:
    results = check_relations(
        [RelationRequirement(subject="cup", relation="above", object="banana")],
        [Detection(label="cup", box=_box(20, 20))],
        image_width=200,
        image_height=200,
    )

    assert not results[0].passed
    assert "banana was not detected" in results[0].detail


# --- checklist -------------------------------------------------------------


class _StubVqa:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def answer(self, path: Path, question: str) -> str:
        self.asked.append(question)
        return self.answers.get(question, "no")


def test_a_child_question_is_not_asked_when_its_parent_fails(tmp_path: Path) -> None:
    questions = [
        ChecklistQuestion(id="q1", text="Is there a hat?"),
        ChecklistQuestion(id="q2", text="Is the hat red?", depends_on="q1"),
    ]
    vqa = _StubVqa({"Is there a hat?": "no", "Is the hat red?": "yes"})

    results = check_questions(questions, tmp_path / "image.png", vqa=vqa)

    assert not results[0].passed
    assert not results[1].passed
    assert results[1].skipped
    # The meaningless question was never paid for.
    assert "Is the hat red?" not in vqa.asked


def test_a_child_question_is_asked_when_its_parent_passes(tmp_path: Path) -> None:
    questions = [
        ChecklistQuestion(id="q1", text="Is there a hat?"),
        ChecklistQuestion(id="q2", text="Is the hat red?", depends_on="q1"),
    ]
    vqa = _StubVqa({"Is there a hat?": "yes", "Is the hat red?": "Yes, clearly."})

    results = check_questions(questions, tmp_path / "image.png", vqa=vqa)

    assert results[0].passed
    assert results[1].passed


def test_an_unclear_vqa_answer_counts_as_no(tmp_path: Path) -> None:
    questions = [ChecklistQuestion(id="q1", text="Is there a hat?")]
    vqa = _StubVqa({"Is there a hat?": "it is hard to say"})

    results = check_questions(questions, tmp_path / "image.png", vqa=vqa)

    assert not results[0].passed


def test_a_vqa_failure_fails_the_check_instead_of_crashing(tmp_path: Path) -> None:
    class _Broken:
        def answer(self, path: Path, question: str) -> str:
            raise RuntimeError("provider down")

    results = check_questions(
        [ChecklistQuestion(id="q1", text="Is there a hat?")], tmp_path / "i.png", vqa=_Broken()
    )

    assert not results[0].passed
    assert "vqa error" in results[0].detail


def test_questions_are_ordered_parents_first() -> None:
    questions = [
        ChecklistQuestion(id="child", text="c", depends_on="parent"),
        ChecklistQuestion(id="parent", text="p"),
    ]

    assert [question.id for question in order_questions(questions)] == ["parent", "child"]


def test_a_dependency_cycle_is_rejected() -> None:
    questions = [
        ChecklistQuestion(id="a", text="a", depends_on="b"),
        ChecklistQuestion(id="b", text="b", depends_on="a"),
    ]

    with pytest.raises(ValueError, match="cycle"):
        order_questions(questions)


def test_a_missing_parent_is_rejected() -> None:
    questions = [ChecklistQuestion(id="a", text="a", depends_on="ghost")]

    with pytest.raises(ValueError, match="unknown"):
        order_questions(questions)
