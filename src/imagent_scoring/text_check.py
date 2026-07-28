from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from .models import CheckResult, TextRequirement, TextSpan


# Text rendering is graded by reading the image, not by asking a model whether
# the text "looks right". OCR output compared against the required string is a
# fact a miner can re-derive from the published image.


_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize(value: str) -> str:
    """Fold away the differences OCR legitimately introduces.

    Case, accents, punctuation and whitespace runs are not what the benchmark is
    testing, and penalising them would make the score depend on the OCR engine's
    typographic habits rather than on the image.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    without_punctuation = _PUNCTUATION.sub(" ", stripped)
    return _WHITESPACE.sub(" ", without_punctuation).strip().casefold()


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(left, right) / longest


def best_window_similarity(needle: str, haystack: str) -> tuple[float, str]:
    """Best match for `needle` anywhere in `haystack`, plus what was matched.

    OCR returns the whole image's text, so a required string has to be located
    inside it. Windows are taken over words rather than characters so the
    reported near-miss is something a human can read in the report.
    """
    if not needle:
        return 1.0, ""
    words = haystack.split()
    if not words:
        return 0.0, ""

    needle_words = max(1, len(needle.split()))
    best_score = 0.0
    best_text = ""
    for width in {max(1, needle_words - 1), needle_words, needle_words + 1}:
        for start in range(0, max(1, len(words) - width + 1)):
            window = " ".join(words[start : start + width])
            score = similarity(needle, window)
            if score > best_score:
                best_score = score
                best_text = window
    return best_score, best_text


def check_text(
    requirements: Sequence[TextRequirement], spans: Sequence[TextSpan]
) -> list[CheckResult]:
    haystack = normalize(" ".join(span.text for span in spans))
    results: list[CheckResult] = []

    for requirement in requirements:
        needle = normalize(requirement.value)

        if requirement.match == "exact":
            passed = needle == haystack
        else:
            passed = bool(needle) and needle in haystack

        if passed:
            results.append(
                CheckResult(
                    kind="text",
                    label=f"text: {requirement.value!r}",
                    passed=True,
                    score=1.0,
                    weight=requirement.weight,
                    detail="found exactly",
                )
            )
            continue

        score, matched = best_window_similarity(needle, haystack)
        if score >= requirement.partial_threshold:
            results.append(
                CheckResult(
                    kind="text",
                    label=f"text: {requirement.value!r}",
                    passed=False,
                    score=score,
                    weight=requirement.weight,
                    detail=f"near miss, read as {matched!r} (similarity {score:.2f})",
                )
            )
            continue

        results.append(
            CheckResult(
                kind="text",
                label=f"text: {requirement.value!r}",
                passed=False,
                score=0.0,
                weight=requirement.weight,
                detail="not found" if not matched else f"closest text was {matched!r}",
            )
        )

    return results
