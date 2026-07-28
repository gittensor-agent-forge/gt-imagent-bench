from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .backends import VqaEngine
from .models import CheckResult, ChecklistQuestion


# The TIFA / DSG method: a set of atomic yes/no questions is frozen with the
# problem, and a VQA model answers them against the image. Questions are written
# once by the generator, never at run time, so the answer key cannot drift
# between two candidates answering the same problem.


def _normalize_answer(value: str) -> str:
    text = value.strip().casefold()
    if text.startswith("yes") or text in {"y", "true", "1"}:
        return "yes"
    if text.startswith("no") or text in {"n", "false", "0"}:
        return "no"
    # Anything a VQA model says that is not a clear yes counts as no. A vague
    # answer must never earn a candidate a point.
    return "no"


def order_questions(questions: Sequence[ChecklistQuestion]) -> list[ChecklistQuestion]:
    """Parents before children. Raises on a missing parent or a dependency cycle."""
    by_id = {question.id: question for question in questions}
    for question in questions:
        if question.depends_on is not None and question.depends_on not in by_id:
            raise ValueError(f"question {question.id!r} depends on unknown {question.depends_on!r}")

    ordered: list[ChecklistQuestion] = []
    placed: set[str] = set()
    remaining = list(questions)
    while remaining:
        progressed = False
        for question in list(remaining):
            if question.depends_on is None or question.depends_on in placed:
                ordered.append(question)
                placed.add(question.id)
                remaining.remove(question)
                progressed = True
        if not progressed:
            cycle = ", ".join(question.id for question in remaining)
            raise ValueError(f"dependency cycle among questions: {cycle}")
    return ordered


def check_questions(
    questions: Sequence[ChecklistQuestion], path: Path, *, vqa: VqaEngine
) -> list[CheckResult]:
    """Answer each question, skipping children whose parent failed.

    Skipping is not leniency: a child whose parent failed is recorded as failed
    without being asked. Asking "is the hat red?" when there is no hat produces a
    meaningless answer, and paying for that answer is worse than useless.
    """
    results: dict[str, CheckResult] = {}

    for question in order_questions(questions):
        parent = results.get(question.depends_on) if question.depends_on else None
        if parent is not None and not parent.passed:
            results[question.id] = CheckResult(
                kind="question",
                label=f"{question.id}: {question.text}",
                passed=False,
                score=0.0,
                weight=question.weight,
                detail=f"not asked: depends on {question.depends_on}, which failed",
                skipped=True,
            )
            continue

        try:
            raw = vqa.answer(path, question.text)
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed check, not a crash
            results[question.id] = CheckResult(
                kind="question",
                label=f"{question.id}: {question.text}",
                passed=False,
                score=0.0,
                weight=question.weight,
                detail=f"vqa error: {exc}",
            )
            continue

        answer = _normalize_answer(raw)
        passed = answer == question.expect
        results[question.id] = CheckResult(
            kind="question",
            label=f"{question.id}: {question.text}",
            passed=passed,
            score=1.0 if passed else 0.0,
            weight=question.weight,
            detail=f"expected {question.expect}, answered {answer}",
        )

    # Report in the order the generator wrote them, not dependency order.
    return [results[question.id] for question in questions]
