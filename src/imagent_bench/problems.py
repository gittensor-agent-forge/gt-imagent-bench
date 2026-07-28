from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any


# Deterministic benchmark problem generation.
#
# Published benchmarks like GenEval are not fixed prompt lists, they are
# templates over vocabularies. Running the template with a fresh seed produces
# problems nobody has seen, using a methodology that is already peer reviewed —
# and because the template knows what it asked for, the answer key comes free.
#
# Two consumers must agree byte for byte: the validator, which generates the
# answer keys, and the sealed room, which generates the prompts inside an
# attested image. So randomness here is derived from SHA-256 in counter mode
# rather than from `random`, whose selection internals have changed between
# CPython releases and would silently desynchronise the two sides.
#
# Sources:
#   GenEval          - counting, colour binding, position (detector-checkable)
#   T2I-CompBench++  - attribute binding, spatial relations
#   AnyText/MARIO    - text rendering (OCR-checkable)
#   DPG-Bench / DSG  - dense prompts (checklist-checkable)


GENERATOR_VERSION = "imagent-problems-v1.0.0"

# One challenge draws this cycle in order, so a 7-problem challenge is always
# 3 GenEval, 2 T2I-CompBench, 1 text rendering, 1 dense.
TASK_CYCLE = (
    "geneval_count",
    "geneval_color",
    "geneval_position",
    "compbench_attribute",
    "compbench_spatial",
    "text_render",
    "dense",
)

# (singular, plural). Plurals are stored rather than derived: an English
# morphology rule is one more thing that could differ between two machines.
# "orange" is deliberately absent — it is also a colour, and a colour-binding
# problem must never be ambiguous about which sense was meant.
OBJECTS: tuple[tuple[str, str], ...] = (
    ("airplane", "airplanes"), ("apple", "apples"), ("backpack", "backpacks"),
    ("banana", "bananas"), ("bear", "bears"), ("bench", "benches"),
    ("bicycle", "bicycles"), ("bird", "birds"), ("boat", "boats"),
    ("book", "books"), ("bottle", "bottles"), ("bowl", "bowls"),
    ("bus", "buses"), ("cake", "cakes"), ("car", "cars"), ("cat", "cats"),
    ("chair", "chairs"), ("clock", "clocks"), ("cup", "cups"), ("dog", "dogs"),
    ("donut", "donuts"), ("elephant", "elephants"), ("fork", "forks"),
    ("giraffe", "giraffes"), ("horse", "horses"), ("keyboard", "keyboards"),
    ("kite", "kites"), ("laptop", "laptops"), ("pizza", "pizzas"),
    ("sandwich", "sandwiches"), ("suitcase", "suitcases"),
    ("teddy bear", "teddy bears"), ("train", "trains"), ("truck", "trucks"),
    ("umbrella", "umbrellas"), ("vase", "vases"), ("zebra", "zebras"),
)

COLORS = ("black", "blue", "brown", "green", "purple", "red", "white", "yellow")
TEXTURES = ("fluffy", "glass", "leather", "metallic", "plastic", "wooden")
SHAPES = ("cubic", "cylindrical", "oval", "rectangular", "round", "square")
# Stored as complete prepositional phrases. A bare noun would need the template
# to choose "in"/"on" per word, which is one more way two machines could differ.
SETTINGS = (
    "on a beach", "on a city street", "in a forest",
    "in a kitchen", "in a park", "in a photo studio",
)

# GenEval tests small exact counts; beyond four, detector error dominates the
# signal and the check stops measuring the agent.
COUNTS = (2, 3, 4)

RELATIONS = (
    ("to the left of", "left_of"),
    ("to the right of", "right_of"),
    ("above", "above"),
    ("below", "below"),
)

# Short, unambiguous, upper-case words: rendering accuracy is the variable, not
# the reader's ability to guess a rare word from a smudge.
TEXT_WORDS = (
    "ANCHOR", "BRIGHT", "CANVAS", "DELTA", "EMBER", "FALCON", "GRAVITY",
    "HARBOR", "IVORY", "JUNIPER", "KINETIC", "LANTERN", "MERIDIAN", "NIMBUS",
    "ORBIT", "PRISM", "QUARTZ", "RIVET", "SUMMIT", "TIDAL", "VECTOR", "ZENITH",
)


@dataclass(frozen=True)
class Problem:
    """One generated problem and the answer key that grades it."""

    problem_id: str
    prompt: str
    task: str
    source: str
    answer_key: dict[str, Any]


def new_challenge_id() -> str:
    """A fresh, unpredictable challenge id.

    The seed is derived from this id, so it MUST NOT be sequential or guessable:
    anyone who can predict the next id can generate the next challenge's problems
    ahead of time and prepare for them. 128 bits of entropy, issued at run time.
    """
    return secrets.token_hex(16)


def generate_problems(seed: bytes, count: int = 7) -> list[Problem]:
    """Derive `count` problems from `seed`. Same seed, same problems, always."""
    if count <= 0:
        raise ValueError("count must be positive")
    problems: list[Problem] = []
    for index in range(count):
        task = TASK_CYCLE[index % len(TASK_CYCLE)]
        problems.append(_BUILDERS[task](seed, index))
    return problems


# --- deterministic selection ------------------------------------------------


def _digest(seed: bytes, label: str) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(seed)
    hasher.update(b"\x00")
    hasher.update(GENERATOR_VERSION.encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(label.encode("utf-8"))
    return hasher.digest()


def _pick(items: tuple, seed: bytes, label: str):
    """Choose one item. Modulo bias over a 64-bit draw is below 2^-58 here."""
    draw = int.from_bytes(_digest(seed, label)[:8], "big")
    return items[draw % len(items)]


def _pick_distinct(items: tuple, seed: bytes, label: str, howmany: int) -> list:
    """Choose `howmany` different items, deterministically.

    Redraws with a widening label rather than shuffling the whole vocabulary,
    so adding a word to a list changes as few existing problems as possible.
    """
    if howmany > len(items):
        raise ValueError("cannot draw more distinct items than the vocabulary holds")
    chosen: list = []
    attempt = 0
    while len(chosen) < howmany:
        candidate = _pick(items, seed, f"{label}:{len(chosen)}:{attempt}")
        if candidate not in chosen:
            chosen.append(candidate)
        attempt += 1
    return chosen


def _problem_id(task: str, seed: bytes, index: int) -> str:
    # Filename-safe and short: this id names artifact files and is embedded in
    # the sealed room's per-problem inference tokens.
    suffix = _digest(seed, f"id:{task}:{index}")[:4].hex()
    return f"{task.replace('_', '-')}-{index:02d}-{suffix}"


def _key(problem_id: str, prompt: str, source: str, task: str, requirements: dict) -> dict[str, Any]:
    return {
        "version": "1.0",
        "problem_id": problem_id,
        "source": source,
        "task": task,
        "prompt": prompt,
        "requirements": requirements,
    }


# --- builders ---------------------------------------------------------------


def _geneval_count(seed: bytes, index: int) -> Problem:
    task = "geneval_count"
    problem_id = _problem_id(task, seed, index)
    singular, plural = _pick(OBJECTS, seed, f"{task}:{index}:object")
    number = _pick(COUNTS, seed, f"{task}:{index}:count")
    prompt = f"a photo of {_spell(number)} {plural}"
    return Problem(
        problem_id=problem_id,
        prompt=prompt,
        task=task,
        source="geneval",
        answer_key=_key(
            problem_id, prompt, "geneval", task,
            {"objects": [{"name": singular, "count": number}]},
        ),
    )


def _geneval_color(seed: bytes, index: int) -> Problem:
    task = "geneval_color"
    problem_id = _problem_id(task, seed, index)
    first, second = _pick_distinct(OBJECTS, seed, f"{task}:{index}:object", 2)
    color_a, color_b = _pick_distinct(COLORS, seed, f"{task}:{index}:color", 2)
    prompt = f"a photo of a {color_a} {first[0]} and a {color_b} {second[0]}"
    return Problem(
        problem_id=problem_id,
        prompt=prompt,
        task=task,
        source="geneval",
        answer_key=_key(
            problem_id, prompt, "geneval", task,
            {
                "objects": [
                    {"name": first[0], "count": 1, "color": color_a},
                    {"name": second[0], "count": 1, "color": color_b},
                ]
            },
        ),
    )


def _geneval_position(seed: bytes, index: int) -> Problem:
    task = "geneval_position"
    problem_id = _problem_id(task, seed, index)
    first, second = _pick_distinct(OBJECTS, seed, f"{task}:{index}:object", 2)
    phrase, relation = _pick(RELATIONS, seed, f"{task}:{index}:relation")
    prompt = f"a photo of a {first[0]} {phrase} a {second[0]}"
    return Problem(
        problem_id=problem_id,
        prompt=prompt,
        task=task,
        source="geneval",
        answer_key=_key(
            problem_id, prompt, "geneval", task,
            {
                "objects": [{"name": first[0], "count": 1}, {"name": second[0], "count": 1}],
                "relations": [
                    {"subject": first[0], "relation": relation, "object": second[0]}
                ],
            },
        ),
    )


def _compbench_attribute(seed: bytes, index: int) -> Problem:
    task = "compbench_attribute"
    problem_id = _problem_id(task, seed, index)
    first, second = _pick_distinct(OBJECTS, seed, f"{task}:{index}:object", 2)
    texture = _pick(TEXTURES, seed, f"{task}:{index}:texture")
    shape = _pick(SHAPES, seed, f"{task}:{index}:shape")
    prompt = f"a photo of a {texture} {first[0]} next to a {shape} {second[0]}"
    # Presence is detector-checkable; texture and shape are not, so they become
    # checklist questions that only run once their object is confirmed present.
    return Problem(
        problem_id=problem_id,
        prompt=prompt,
        task=task,
        source="t2i_compbench",
        answer_key=_key(
            problem_id, prompt, "t2i_compbench", task,
            {
                "objects": [{"name": first[0], "count": 1}, {"name": second[0], "count": 1}],
                "questions": [
                    {"id": "q1", "text": f"Is there a {first[0]} in the image?"},
                    {
                        "id": "q2",
                        "text": f"Is the {first[0]} {texture}?",
                        "depends_on": "q1",
                    },
                    {"id": "q3", "text": f"Is there a {second[0]} in the image?"},
                    {
                        "id": "q4",
                        "text": f"Is the {second[0]} {shape} in shape?",
                        "depends_on": "q3",
                    },
                ],
            },
        ),
    )


def _compbench_spatial(seed: bytes, index: int) -> Problem:
    task = "compbench_spatial"
    problem_id = _problem_id(task, seed, index)
    first, second = _pick_distinct(OBJECTS, seed, f"{task}:{index}:object", 2)
    phrase, relation = _pick(RELATIONS, seed, f"{task}:{index}:relation")
    color = _pick(COLORS, seed, f"{task}:{index}:color")
    setting = _pick(SETTINGS, seed, f"{task}:{index}:setting")
    prompt = f"a photo of a {color} {first[0]} {phrase} a {second[0]}, {setting}"
    return Problem(
        problem_id=problem_id,
        prompt=prompt,
        task=task,
        source="t2i_compbench",
        answer_key=_key(
            problem_id, prompt, "t2i_compbench", task,
            {
                "objects": [
                    {"name": first[0], "count": 1, "color": color},
                    {"name": second[0], "count": 1},
                ],
                "relations": [
                    {"subject": first[0], "relation": relation, "object": second[0]}
                ],
                "questions": [{"id": "q1", "text": f"Is the scene {setting}?"}],
            },
        ),
    )


def _text_render(seed: bytes, index: int) -> Problem:
    task = "text_render"
    problem_id = _problem_id(task, seed, index)
    word = _pick(TEXT_WORDS, seed, f"{task}:{index}:word")
    color = _pick(COLORS, seed, f"{task}:{index}:color")
    prompt = f'a photo of a {color} sign with the word "{word}" written on it in large clear letters'
    # OCR decides this one. No model is asked whether the text "looks right".
    return Problem(
        problem_id=problem_id,
        prompt=prompt,
        task=task,
        source="text_rendering",
        answer_key=_key(
            problem_id, prompt, "text_rendering", task,
            {
                "text": [{"value": word, "match": "contains", "weight": 3.0}],
                "questions": [
                    {"id": "q1", "text": "Is there a sign in the image?"},
                    {
                        "id": "q2",
                        "text": "Is the text on the sign a single word, without duplicates?",
                        "depends_on": "q1",
                    },
                ],
            },
        ),
    )


def _dense(seed: bytes, index: int) -> Problem:
    task = "dense"
    problem_id = _problem_id(task, seed, index)
    # Four distinct objects, so no object is described by two clauses at once:
    # a counted background object that also appeared in the foreground would make
    # its own answer key ambiguous.
    first, second, third, fourth = _pick_distinct(OBJECTS, seed, f"{task}:{index}:object", 4)
    color_a, color_b = _pick_distinct(COLORS, seed, f"{task}:{index}:color", 2)
    texture = _pick(TEXTURES, seed, f"{task}:{index}:texture")
    setting = _pick(SETTINGS, seed, f"{task}:{index}:setting")
    number = _pick(COUNTS, seed, f"{task}:{index}:count")
    phrase, relation = _pick(RELATIONS, seed, f"{task}:{index}:relation")
    prompt = (
        f"a detailed photo taken {setting}: a {color_a} {first[0]} {phrase} "
        f"a {texture} {second[0]}, with {_spell(number)} {third[1]} in the background, "
        f"and a {color_b} {fourth[0]} in sharp focus"
    )
    return Problem(
        problem_id=problem_id,
        prompt=prompt,
        task=task,
        source="dpg_bench",
        answer_key=_key(
            problem_id, prompt, "dpg_bench", task,
            {
                "objects": [
                    {"name": first[0], "count": 1, "color": color_a},
                    {"name": second[0], "count": 1},
                    {"name": third[0], "count": number},
                    {"name": fourth[0], "count": 1, "color": color_b},
                ],
                "relations": [
                    {"subject": first[0], "relation": relation, "object": second[0]}
                ],
                "questions": [
                    {"id": "q1", "text": f"Is the scene {setting}?"},
                    {"id": "q2", "text": f"Is there a {second[0]} in the image?"},
                    {
                        "id": "q3",
                        "text": f"Is the {second[0]} {texture}?",
                        "depends_on": "q2",
                    },
                ],
            },
        ),
    )


_BUILDERS = {
    "geneval_count": _geneval_count,
    "geneval_color": _geneval_color,
    "geneval_position": _geneval_position,
    "compbench_attribute": _compbench_attribute,
    "compbench_spatial": _compbench_spatial,
    "text_render": _text_render,
    "dense": _dense,
}

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four"}


def _spell(number: int) -> str:
    # Image models follow spelled-out small numbers more reliably than digits,
    # and GenEval's own count prompts are written this way.
    return _NUMBER_WORDS.get(number, str(number))
