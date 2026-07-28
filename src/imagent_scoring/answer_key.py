from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    AnswerKey,
    ChecklistQuestion,
    ObjectRequirement,
    RelationRequirement,
    TextRequirement,
)


# Answer keys are machine-written by a benchmark generator. Parsing is strict on
# purpose: a malformed key must fail loudly, because a key that silently parses
# as empty would grade every image as perfect.


SCHEMA_VERSION = "1.0"
VALID_RELATIONS = {"left_of", "right_of", "above", "below"}
VALID_MATCHES = {"exact", "contains"}


class AnswerKeyError(ValueError):
    """Raised when an answer key cannot be parsed."""


def load_answer_key(path: str | Path) -> AnswerKey:
    file_path = Path(path)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnswerKeyError(f"{file_path}: not valid JSON: {exc}") from exc
    return parse_answer_key(raw)


def parse_answer_key(raw: Any) -> AnswerKey:
    if not isinstance(raw, dict):
        raise AnswerKeyError("answer key must be a JSON object")

    version = str(raw.get("version", ""))
    if version != SCHEMA_VERSION:
        raise AnswerKeyError(f"unsupported answer key version: {version!r}")

    problem_id = _required_string(raw, "problem_id")
    prompt = _required_string(raw, "prompt")
    source = _required_string(raw, "source")

    requirements = raw.get("requirements", {})
    if not isinstance(requirements, dict):
        raise AnswerKeyError("'requirements' must be an object")

    key = AnswerKey(
        version=version,
        problem_id=problem_id,
        prompt=prompt,
        source=source,
        task=str(raw.get("task", "")),
        text=tuple(_parse_text(item) for item in _list(requirements, "text")),
        objects=tuple(_parse_object(item) for item in _list(requirements, "objects")),
        relations=tuple(_parse_relation(item) for item in _list(requirements, "relations")),
        questions=tuple(_parse_question(item) for item in _list(requirements, "questions")),
    )

    if key.is_empty():
        raise AnswerKeyError(f"answer key {problem_id!r} declares no requirements")

    identifiers = [question.id for question in key.questions]
    if len(set(identifiers)) != len(identifiers):
        raise AnswerKeyError(f"answer key {problem_id!r} has duplicate question ids")

    return key


def _required_string(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AnswerKeyError(f"'{field}' must be a non-empty string")
    return value.strip()


def _list(raw: dict[str, Any], field: str) -> list[Any]:
    value = raw.get(field, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnswerKeyError(f"'{field}' must be an array")
    return value


def _weight(raw: dict[str, Any]) -> float:
    value = raw.get("weight", 1.0)
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise AnswerKeyError(f"weight must be a number, got {value!r}") from exc
    if weight <= 0:
        raise AnswerKeyError(f"weight must be positive, got {weight}")
    return weight


def _parse_text(raw: Any) -> TextRequirement:
    if isinstance(raw, str):
        return TextRequirement(value=raw)
    if not isinstance(raw, dict):
        raise AnswerKeyError("text requirement must be a string or an object")

    value = _required_string(raw, "value")
    match = str(raw.get("match", "contains"))
    if match not in VALID_MATCHES:
        raise AnswerKeyError(f"text match must be one of {sorted(VALID_MATCHES)}, got {match!r}")

    threshold = float(raw.get("partial_threshold", 0.8))
    if not 0.0 <= threshold <= 1.0:
        raise AnswerKeyError(f"partial_threshold must be within [0, 1], got {threshold}")

    return TextRequirement(
        value=value, match=match, weight=_weight(raw), partial_threshold=threshold  # type: ignore[arg-type]
    )


def _parse_object(raw: Any) -> ObjectRequirement:
    if not isinstance(raw, dict):
        raise AnswerKeyError("object requirement must be an object")

    name = _required_string(raw, "name")
    count = raw.get("count")
    if count is not None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise AnswerKeyError(f"object count must be a non-negative integer, got {count!r}")

    color = raw.get("color")
    if color is not None and (not isinstance(color, str) or not color.strip()):
        raise AnswerKeyError("object color must be a non-empty string when present")

    return ObjectRequirement(
        name=name,
        count=count,
        color=color.strip() if isinstance(color, str) else None,
        weight=_weight(raw),
    )


def _parse_relation(raw: Any) -> RelationRequirement:
    if not isinstance(raw, dict):
        raise AnswerKeyError("relation requirement must be an object")

    relation = _required_string(raw, "relation")
    if relation not in VALID_RELATIONS:
        raise AnswerKeyError(f"relation must be one of {sorted(VALID_RELATIONS)}, got {relation!r}")

    return RelationRequirement(
        subject=_required_string(raw, "subject"),
        relation=relation,  # type: ignore[arg-type]
        object=_required_string(raw, "object"),
        weight=_weight(raw),
    )


def _parse_question(raw: Any) -> ChecklistQuestion:
    if not isinstance(raw, dict):
        raise AnswerKeyError("question must be an object")

    expect = str(raw.get("expect", "yes")).strip().casefold()
    if expect not in {"yes", "no"}:
        raise AnswerKeyError(f"question expect must be 'yes' or 'no', got {expect!r}")

    depends_on = raw.get("depends_on")
    if depends_on is not None and (not isinstance(depends_on, str) or not depends_on.strip()):
        raise AnswerKeyError("depends_on must be a non-empty string when present")

    return ChecklistQuestion(
        id=_required_string(raw, "id"),
        text=_required_string(raw, "text"),
        expect=expect,  # type: ignore[arg-type]
        depends_on=depends_on.strip() if isinstance(depends_on, str) else None,
        weight=_weight(raw),
    )
