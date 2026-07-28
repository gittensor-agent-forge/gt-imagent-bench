from __future__ import annotations

from collections.abc import Sequence

from .models import CheckResult, Detection, ObjectRequirement, RelationRequirement


# Object presence, counting, colour and position, following the GenEval approach:
# a detector reports what is in the image and the checks are plain arithmetic over
# the boxes. Nothing here is a judgement call.


DEFAULT_MIN_CONFIDENCE = 0.5
# Two centres closer than this fraction of the image's size are not meaningfully
# left/right or above/below each other, so the relation is scored as unmet rather
# than decided by a pixel.
RELATION_MARGIN = 0.05


def _matching(
    detections: Sequence[Detection], label: str, min_confidence: float
) -> list[Detection]:
    wanted = label.strip().casefold()
    return [
        detection
        for detection in detections
        if detection.label.strip().casefold() == wanted
        and detection.confidence >= min_confidence
    ]


def check_objects(
    requirements: Sequence[ObjectRequirement],
    detections: Sequence[Detection],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    for requirement in requirements:
        found = _matching(detections, requirement.name, min_confidence)

        if requirement.count is None:
            passed = len(found) >= 1
            results.append(
                CheckResult(
                    kind="object",
                    label=f"object: {requirement.name}",
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    weight=requirement.weight,
                    detail=f"found {len(found)}",
                )
            )
        else:
            passed = len(found) == requirement.count
            # Partial credit scales with how close the count was, so 2 of 3 beats
            # 0 of 3. A miscount can never reach 1.0, because the divisor is at
            # least the requested count.
            miscount = abs(len(found) - requirement.count)
            score = 1.0 if passed else max(0.0, 1.0 - miscount / max(1, requirement.count))
            results.append(
                CheckResult(
                    kind="object",
                    label=f"object: {requirement.count}x {requirement.name}",
                    passed=passed,
                    score=score,
                    weight=requirement.weight,
                    detail=f"expected {requirement.count}, found {len(found)}",
                )
            )

        if requirement.color is None:
            continue

        wanted_color = requirement.color.strip().casefold()
        expected = requirement.count if requirement.count is not None else max(1, len(found))
        correct = [
            detection
            for detection in found
            if (detection.color or "").strip().casefold() == wanted_color
        ]
        colour_passed = len(correct) >= expected and expected > 0
        results.append(
            CheckResult(
                kind="object",
                label=f"colour: {requirement.name} is {requirement.color}",
                passed=colour_passed,
                score=1.0 if colour_passed else (len(correct) / expected if expected else 0.0),
                weight=requirement.weight,
                detail=(
                    f"{len(correct)} of {expected} {requirement.name} are {requirement.color}"
                    if found
                    else f"no {requirement.name} detected to check colour"
                ),
            )
        )

    return results


def check_relations(
    requirements: Sequence[RelationRequirement],
    detections: Sequence[Detection],
    *,
    image_width: float,
    image_height: float,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    margin: float = RELATION_MARGIN,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    for requirement in requirements:
        subjects = _matching(detections, requirement.subject, min_confidence)
        objects = _matching(detections, requirement.object, min_confidence)
        label = f"relation: {requirement.subject} {requirement.relation} {requirement.object}"

        if not subjects or not objects:
            missing = requirement.subject if not subjects else requirement.object
            results.append(
                CheckResult(
                    kind="relation",
                    label=label,
                    passed=False,
                    score=0.0,
                    weight=requirement.weight,
                    detail=f"{missing} was not detected",
                )
            )
            continue

        # Any detected pair satisfying the relation counts: the prompt asks for a
        # cup left of a banana, not for every cup to be left of every banana.
        passed = any(
            _relation_holds(
                requirement.relation,
                subject.center,
                obj.center,
                image_width=image_width,
                image_height=image_height,
                margin=margin,
            )
            for subject in subjects
            for obj in objects
        )
        results.append(
            CheckResult(
                kind="relation",
                label=label,
                passed=passed,
                score=1.0 if passed else 0.0,
                weight=requirement.weight,
                detail="satisfied" if passed else "no detected pair satisfies the relation",
            )
        )

    return results


def _relation_holds(
    relation: str,
    subject_center: tuple[float, float],
    object_center: tuple[float, float],
    *,
    image_width: float,
    image_height: float,
    margin: float,
) -> bool:
    subject_x, subject_y = subject_center
    object_x, object_y = object_center
    horizontal_margin = image_width * margin
    vertical_margin = image_height * margin

    if relation == "left_of":
        return subject_x < object_x - horizontal_margin
    if relation == "right_of":
        return subject_x > object_x + horizontal_margin
    if relation == "above":
        return subject_y < object_y - vertical_margin
    if relation == "below":
        return subject_y > object_y + vertical_margin
    raise ValueError(f"unknown relation: {relation}")


# --- the no-detector path ----------------------------------------------------
#
# Same checks, same scoring, asked of a vision model instead of read off boxes.
# Every result is flagged non-deterministic, because it is: a detector's answer
# can be reproduced by anyone holding the image, and this one cannot.


def verify_objects(requirements, path, *, verifier) -> list[CheckResult]:
    results: list[CheckResult] = []

    for requirement in requirements:
        try:
            found = max(0, int(verifier.count(path, requirement.name)))
        except Exception as error:  # noqa: BLE001 - a failed check is a failed check
            results.append(
                CheckResult(
                    kind="object",
                    label=f"object: {requirement.name}",
                    passed=False,
                    score=0.0,
                    weight=requirement.weight,
                    detail=f"verifier error: {error}"[:200],
                )
            )
            continue

        if requirement.count is None:
            passed = found >= 1
            score = 1.0 if passed else 0.0
            label = f"object: {requirement.name}"
        else:
            passed = found == requirement.count
            miscount = abs(found - requirement.count)
            score = 1.0 if passed else max(0.0, 1.0 - miscount / max(1, requirement.count))
            label = f"object: {requirement.count}x {requirement.name}"

        results.append(
            CheckResult(
                kind="object",
                label=label,
                passed=passed,
                score=score,
                weight=requirement.weight,
                detail=f"counted {found} (non-deterministic: vision model)",
            )
        )

        if requirement.color is None:
            continue

        try:
            seen = str(verifier.colour(path, requirement.name)).strip().casefold()
        except Exception as error:  # noqa: BLE001
            seen = ""
        wanted = requirement.color.strip().casefold()
        colour_passed = bool(seen) and seen == wanted
        results.append(
            CheckResult(
                kind="object",
                label=f"colour: {requirement.name} is {requirement.color}",
                passed=colour_passed,
                score=1.0 if colour_passed else 0.0,
                weight=requirement.weight,
                detail=f"saw {seen or 'nothing'} (non-deterministic: vision model)",
            )
        )

    return results


def verify_relations(requirements, path, *, verifier) -> list[CheckResult]:
    results: list[CheckResult] = []
    for requirement in requirements:
        label = f"relation: {requirement.subject} {requirement.relation} {requirement.object}"
        try:
            holds = bool(
                verifier.relation(path, requirement.subject, requirement.relation, requirement.object)
            )
            detail = "satisfied" if holds else "not satisfied"
        except Exception as error:  # noqa: BLE001
            holds, detail = False, f"verifier error: {error}"[:200]
        results.append(
            CheckResult(
                kind="relation",
                label=label,
                passed=holds,
                score=1.0 if holds else 0.0,
                weight=requirement.weight,
                detail=f"{detail} (non-deterministic: vision model)",
            )
        )
    return results
