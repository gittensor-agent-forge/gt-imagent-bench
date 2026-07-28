from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagent_scoring import load_answer_key, parse_answer_key
from imagent_scoring.answer_key import AnswerKeyError


def _key(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": "1.0",
        "problem_id": "geneval-0001",
        "source": "geneval",
        "task": "counting",
        "prompt": "a photo of three yellow bananas and a blue cup",
        "requirements": {
            "objects": [
                {"name": "banana", "count": 3, "color": "yellow"},
                {"name": "cup", "count": 1, "color": "blue"},
            ]
        },
    }
    payload.update(overrides)
    return payload


def test_a_generated_key_parses(tmp_path: Path) -> None:
    path = tmp_path / "key.json"
    path.write_text(json.dumps(_key()), encoding="utf-8")

    key = load_answer_key(path)

    assert key.problem_id == "geneval-0001"
    assert key.objects[0].name == "banana"
    assert key.objects[0].count == 3
    assert key.objects[0].color == "yellow"


def test_a_bare_string_is_accepted_as_a_text_requirement() -> None:
    key = parse_answer_key(_key(requirements={"text": ["PASS"]}))

    assert key.text[0].value == "PASS"
    assert key.text[0].match == "contains"


def test_an_unsupported_version_is_rejected() -> None:
    with pytest.raises(AnswerKeyError, match="version"):
        parse_answer_key(_key(version="2.0"))


def test_a_key_with_no_requirements_is_rejected() -> None:
    # This is the dangerous case: an empty key would grade every image as perfect.
    with pytest.raises(AnswerKeyError, match="no requirements"):
        parse_answer_key(_key(requirements={}))


def test_duplicate_question_ids_are_rejected() -> None:
    payload = _key(
        requirements={
            "questions": [
                {"id": "q1", "text": "one"},
                {"id": "q1", "text": "two"},
            ]
        }
    )

    with pytest.raises(AnswerKeyError, match="duplicate question ids"):
        parse_answer_key(payload)


def test_an_unknown_relation_is_rejected() -> None:
    payload = _key(
        requirements={"relations": [{"subject": "cup", "relation": "beside", "object": "banana"}]}
    )

    with pytest.raises(AnswerKeyError, match="relation must be one of"):
        parse_answer_key(payload)


def test_a_negative_count_is_rejected() -> None:
    with pytest.raises(AnswerKeyError, match="non-negative integer"):
        parse_answer_key(_key(requirements={"objects": [{"name": "cup", "count": -1}]}))


def test_a_non_positive_weight_is_rejected() -> None:
    with pytest.raises(AnswerKeyError, match="weight must be positive"):
        parse_answer_key(_key(requirements={"objects": [{"name": "cup", "weight": 0}]}))


def test_malformed_json_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(AnswerKeyError, match="not valid JSON"):
        load_answer_key(path)
