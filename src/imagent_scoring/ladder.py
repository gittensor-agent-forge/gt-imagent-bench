from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .judge import JudgeVerdict, Side

# Deciding which of two images answered one problem better.
#
# This is a benchmark question, not a competition question: it asks only "which
# image is better", and knows nothing about crowns, margins, or promotion. What a
# competition then does with a run of these verdicts lives in gt-imagent.
#
# The ladder is a strict order of precedence, not a weighted blend. Weights would
# invite an endless argument about whether beauty is worth 0.3 or 0.4 of
# correctness; an order only has to answer "which evidence is stronger?", and the
# answer is that a measured fact beats a model's taste.

# A fact-score gap this large is beyond what taste may overturn. Fifteen points
# is roughly one whole failed requirement out of seven, which no amount of
# composition should be able to buy back.
FACT_DOMINANCE_GAP = 0.15
# Below this, the preference models are not separated enough to break a tie the
# judge could not break either.
PREFERENCE_GAP = 0.10


@dataclass(frozen=True)
class SideResult:
    """What is known about one competitor's answer to one problem."""

    valid: bool
    fact_score: float = 0.0
    preference: float | None = None
    error: str = ""


@dataclass(frozen=True)
class ProblemVerdict:
    problem_id: str
    winner: Side
    rule: str
    detail: str
    king: SideResult
    challenger: SideResult
    judge: JudgeVerdict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "winner": self.winner,
            "rule": self.rule,
            "detail": self.detail,
            "king": {"valid": self.king.valid, "fact_score": round(self.king.fact_score, 6)},
            "challenger": {
                "valid": self.challenger.valid,
                "fact_score": round(self.challenger.fact_score, 6),
            },
            "judge": self.judge.to_dict() if self.judge else None,
        }


def decide_problem(
    *,
    problem_id: str,
    king: SideResult,
    challenger: SideResult,
    judge: JudgeVerdict | None = None,
) -> ProblemVerdict:
    """Apply the ladder. The first rule that applies decides."""

    def verdict(winner: Side, rule: str, detail: str) -> ProblemVerdict:
        return ProblemVerdict(
            problem_id=problem_id,
            winner=winner,
            rule=rule,
            detail=detail,
            king=king,
            challenger=challenger,
            judge=judge,
        )

    # 1. Validity. A broken image is not a candidate for anything.
    if not king.valid and not challenger.valid:
        return verdict("tie", "validity", "neither side produced a usable image")
    if not challenger.valid:
        return verdict("king", "validity", f"challenger image unusable: {challenger.error or 'invalid'}")
    if not king.valid:
        return verdict("challenger", "validity", f"king image unusable: {king.error or 'invalid'}")

    # 2. A decisive objective gap. Taste does not get a vote here.
    gap = challenger.fact_score - king.fact_score
    if abs(gap) >= FACT_DOMINANCE_GAP:
        winner: Side = "challenger" if gap > 0 else "king"
        return verdict(
            winner,
            "fact_dominance",
            f"fact score {challenger.fact_score:.2f} vs {king.fact_score:.2f}"
            f" (gap {abs(gap):.2f} ≥ {FACT_DOMINANCE_GAP:.2f})",
        )

    # 3. The blind head-to-head.
    if judge is not None and judge.winner != "tie":
        return verdict(judge.winner, "judge", f"blind majority chose the {judge.winner}")

    # 4. Human-preference models, only once the judge could not separate them.
    if king.preference is not None and challenger.preference is not None:
        preference_gap = challenger.preference - king.preference
        if abs(preference_gap) >= PREFERENCE_GAP:
            winner = "challenger" if preference_gap > 0 else "king"
            return verdict(
                winner,
                "preference",
                f"preference {challenger.preference:.2f} vs {king.preference:.2f}",
            )

    return verdict("tie", "tie", "no rule separated the two images")


def mean_fact_score(verdicts: list[ProblemVerdict], *, side: str) -> float:
    """Average objective score across a run of problems, for one side."""
    scores = [
        (verdict.king if side == "king" else verdict.challenger).fact_score for verdict in verdicts
    ]
    return sum(scores) / len(scores) if scores else 0.0
