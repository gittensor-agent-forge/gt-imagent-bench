from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# An answer key describes what a correct image must contain. It is produced by a
# benchmark generator at the same time as the prompt, never written by hand, so
# grading never depends on someone's opinion of what the prompt "meant".


MatchMode = Literal["exact", "contains"]
Relation = Literal["left_of", "right_of", "above", "below"]


@dataclass(frozen=True)
class TextRequirement:
    """A string that must be legible in the image."""

    value: str
    match: MatchMode = "contains"
    weight: float = 1.0
    # Below this normalised similarity a near-miss earns nothing. Above it, the
    # requirement earns partial credit equal to the similarity, so a single
    # wrong character is not scored the same as missing text entirely.
    partial_threshold: float = 0.8


@dataclass(frozen=True)
class ObjectRequirement:
    """An object that must appear, optionally a given number of times."""

    name: str
    count: int | None = None
    color: str | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class RelationRequirement:
    """A spatial relation between two objects, checked from detection boxes."""

    subject: str
    relation: Relation
    object: str
    weight: float = 1.0


@dataclass(frozen=True)
class ChecklistQuestion:
    """One atomic yes/no question, answered by a VQA model.

    `depends_on` implements the Davidsonian Scene Graph rule: a child question is
    only meaningful when its parent passed. Asking "is the hat red?" when there is
    no hat produces a meaningless answer, so the child is auto-failed instead.
    """

    id: str
    text: str
    expect: Literal["yes", "no"] = "yes"
    depends_on: str | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class AnswerKey:
    version: str
    problem_id: str
    prompt: str
    source: str
    task: str = ""
    text: tuple[TextRequirement, ...] = field(default_factory=tuple)
    objects: tuple[ObjectRequirement, ...] = field(default_factory=tuple)
    relations: tuple[RelationRequirement, ...] = field(default_factory=tuple)
    questions: tuple[ChecklistQuestion, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not (self.text or self.objects or self.relations or self.questions)


# --- what a backend hands back ---------------------------------------------


@dataclass(frozen=True)
class TextSpan:
    """One piece of text an OCR engine found."""

    text: str
    confidence: float = 1.0


@dataclass(frozen=True)
class Detection:
    """One object an object detector found.

    `box` is (x0, y0, x1, y1) in pixels. `color` is filled in by the backend when
    a colour classifier ran over the crop; the pure checks never guess colour.
    """

    label: str
    box: tuple[float, float, float, float]
    confidence: float = 1.0
    color: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# --- what the scorer hands back --------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One graded requirement. This is what gets published to the miner."""

    kind: Literal["text", "object", "relation", "question"]
    label: str
    passed: bool
    score: float
    weight: float
    detail: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "passed": self.passed,
            "score": round(self.score, 6),
            "weight": self.weight,
            "detail": self.detail,
            "skipped": self.skipped,
        }


@dataclass(frozen=True)
class FactReport:
    """The objective half of the score for one image."""

    problem_id: str
    fact_score: float
    checks: tuple[CheckResult, ...]

    @property
    def passed_count(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "fact_score": round(self.fact_score, 6),
            "passed": self.passed_count,
            "total": len(self.checks),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class ValidityReport:
    """S1: is this image usable at all? Cheap, deterministic, runs first."""

    valid: bool
    reasons: tuple[str, ...]
    width: int
    height: int
    phash: str
    stddev: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "width": self.width,
            "height": self.height,
            "phash": self.phash,
            "stddev": round(self.stddev, 6),
        }
