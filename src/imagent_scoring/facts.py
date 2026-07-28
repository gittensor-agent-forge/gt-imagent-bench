from __future__ import annotations

from pathlib import Path

from .backends import ImageLoader, ObjectDetector, OcrEngine, VqaEngine
from .checklist import check_questions
from .models import AnswerKey, CheckResult, FactReport
from .object_check import check_objects, check_relations
from .text_check import check_text


# S2. Runs every check the answer key declares and folds them into one number.
# This is the objective half of the score, and the half that decides most
# matches. Every value here is recomputable by anyone holding the published image
# and the published answer key.


class MissingBackendError(RuntimeError):
    """Raised when an answer key needs a backend that was not supplied."""


def score_facts(
    path: str | Path,
    key: AnswerKey,
    *,
    loader: ImageLoader | None = None,
    ocr: OcrEngine | None = None,
    detector: ObjectDetector | None = None,
    vqa: VqaEngine | None = None,
) -> FactReport:
    """Grade one image against one answer key.

    A missing backend is an error, never a skipped check: silently dropping a
    requirement would inflate the score of every candidate that happens to be
    graded on a misconfigured machine.
    """
    image_path = Path(path)
    checks: list[CheckResult] = []

    if key.text:
        if ocr is None:
            raise MissingBackendError(
                f"answer key {key.problem_id!r} declares text requirements but no OCR engine was supplied"
            )
        checks.extend(check_text(key.text, ocr.read(image_path)))

    if key.objects or key.relations:
        if detector is None:
            raise MissingBackendError(
                f"answer key {key.problem_id!r} declares object requirements but no detector was supplied"
            )
        # One detection pass feeds both the object checks and the relation checks.
        detections = detector.detect(image_path)

        if key.objects:
            checks.extend(check_objects(key.objects, detections))

        if key.relations:
            if loader is None:
                raise MissingBackendError(
                    f"answer key {key.problem_id!r} declares relations but no image loader was supplied"
                )
            width, height = loader.size(image_path)
            checks.extend(
                check_relations(
                    key.relations, detections, image_width=width, image_height=height
                )
            )

    if key.questions:
        if vqa is None:
            raise MissingBackendError(
                f"answer key {key.problem_id!r} declares checklist questions but no VQA engine was supplied"
            )
        checks.extend(check_questions(key.questions, image_path, vqa=vqa))

    return FactReport(
        problem_id=key.problem_id,
        fact_score=aggregate(checks),
        checks=tuple(checks),
    )


def aggregate(checks: list[CheckResult]) -> float:
    """Weighted mean of the per-check scores, in [0, 1]."""
    total_weight = sum(check.weight for check in checks)
    if total_weight <= 0:
        return 0.0
    return sum(check.score * check.weight for check in checks) / total_weight
