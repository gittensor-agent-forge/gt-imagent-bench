from __future__ import annotations

import re

import pytest

from imagent_bench.problems import (
    GENERATOR_VERSION,
    TASK_CYCLE,
    generate_problems,
    new_challenge_id,
)
from imagent_scoring import parse_answer_key

SEED = bytes.fromhex("06ed3f947613023f004a23e753f01336" * 2)
OTHER_SEED = bytes.fromhex("ff" * 32)


# --- determinism ------------------------------------------------------------


def test_the_same_seed_always_gives_the_same_problems() -> None:
    # The validator and the sealed room derive problems independently. If these
    # ever diverge, the room answers questions the validator is not grading.
    first = generate_problems(SEED)
    second = generate_problems(SEED)

    assert [problem.prompt for problem in first] == [problem.prompt for problem in second]
    assert [problem.problem_id for problem in first] == [problem.problem_id for problem in second]
    assert [problem.answer_key for problem in first] == [problem.answer_key for problem in second]


def test_a_different_seed_gives_different_problems() -> None:
    mine = {problem.prompt for problem in generate_problems(SEED)}
    theirs = {problem.prompt for problem in generate_problems(OTHER_SEED)}

    assert not mine & theirs


def test_generation_is_stable_across_versions() -> None:
    """Golden vector. If this fails, a challenge is no longer reproducible.

    Changing it is a rule change: bump GENERATOR_VERSION and say so, because
    every past challenge's published seed stops regenerating its problems.
    """
    problems = generate_problems(SEED)

    assert GENERATOR_VERSION == "imagent-problems-v1.0.0"
    assert problems[0].problem_id == "geneval-count-00-d75d8920"
    assert problems[0].prompt == "a photo of three cakes"
    assert problems[2].prompt == "a photo of a keyboard below a truck"
    assert problems[5].prompt == (
        'a photo of a blue sign with the word "MERIDIAN" written on it '
        "in large clear letters"
    )


# --- shape ------------------------------------------------------------------


def test_a_challenge_draws_the_intended_benchmark_mix() -> None:
    problems = generate_problems(SEED, count=7)

    assert [problem.task for problem in problems] == list(TASK_CYCLE)
    assert [problem.source for problem in problems] == [
        "geneval", "geneval", "geneval",
        "t2i_compbench", "t2i_compbench",
        "text_rendering", "dpg_bench",
    ]


def test_problem_ids_are_unique_and_filename_safe() -> None:
    # These ids name artifact files and are embedded in the room's per-problem
    # inference tokens, which accept [A-Za-z0-9._-]{1,64}.
    problems = generate_problems(SEED, count=21)
    identifiers = [problem.problem_id for problem in problems]

    assert len(set(identifiers)) == len(identifiers)
    for identifier in identifiers:
        assert re.fullmatch(r"[A-Za-z0-9._-]{1,64}", identifier)


def test_the_cycle_repeats_beyond_its_length() -> None:
    problems = generate_problems(SEED, count=9)

    assert problems[7].task == TASK_CYCLE[0]
    # A repeated task still gets its own distinct problem.
    assert problems[7].prompt != problems[0].prompt


def test_a_non_positive_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        generate_problems(SEED, count=0)


# --- the answer keys are the whole point ------------------------------------


def test_every_generated_answer_key_parses() -> None:
    # A key that fails to parse would stop the run; a key that parses as empty
    # would grade every image as perfect. Both are caught here.
    for problem in generate_problems(SEED, count=21):
        key = parse_answer_key(problem.answer_key)

        assert key.problem_id == problem.problem_id
        assert key.prompt == problem.prompt
        assert not key.is_empty()


def test_counting_problems_ask_for_the_number_in_the_prompt() -> None:
    problem = generate_problems(SEED)[0]
    key = parse_answer_key(problem.answer_key)
    words = {2: "two", 3: "three", 4: "four"}

    assert len(key.objects) == 1
    assert words[key.objects[0].count] in problem.prompt
    assert key.objects[0].name in problem.prompt


def test_colour_problems_bind_each_colour_to_its_own_object() -> None:
    key = parse_answer_key(generate_problems(SEED)[1].answer_key)

    assert len(key.objects) == 2
    assert key.objects[0].color and key.objects[1].color
    assert key.objects[0].color != key.objects[1].color
    assert key.objects[0].name != key.objects[1].name


def test_position_problems_name_a_checkable_relation() -> None:
    problem = generate_problems(SEED)[2]
    key = parse_answer_key(problem.answer_key)

    assert len(key.relations) == 1
    relation = key.relations[0]
    assert relation.relation in {"left_of", "right_of", "above", "below"}
    assert relation.subject in problem.prompt
    assert relation.object in problem.prompt


def test_text_problems_are_graded_by_ocr_not_by_opinion() -> None:
    problem = generate_problems(SEED)[5]
    key = parse_answer_key(problem.answer_key)

    assert len(key.text) == 1
    assert f'"{key.text[0].value}"' in problem.prompt
    # The OCR check outweighs the two supporting questions.
    assert key.text[0].weight == 3.0


def test_checklist_dependencies_always_point_at_a_real_parent() -> None:
    for problem in generate_problems(SEED, count=21):
        key = parse_answer_key(problem.answer_key)
        identifiers = {question.id for question in key.questions}
        for question in key.questions:
            if question.depends_on is not None:
                assert question.depends_on in identifiers


def test_no_object_is_described_by_two_clauses_at_once() -> None:
    # A dense prompt that counted the same object twice would contradict itself,
    # and its answer key would be unsatisfiable.
    for problem in generate_problems(SEED, count=21):
        key = parse_answer_key(problem.answer_key)
        names = [requirement.name for requirement in key.objects]
        assert len(names) == len(set(names)), problem.prompt


# --- challenge ids ----------------------------------------------------------


def test_challenge_ids_are_unpredictable() -> None:
    # A sequential id would let anyone derive the next challenge's problems in
    # advance and prepare for them.
    identifiers = {new_challenge_id() for _ in range(50)}

    assert len(identifiers) == 50
    for identifier in identifiers:
        assert re.fullmatch(r"[0-9a-f]{32}", identifier)
