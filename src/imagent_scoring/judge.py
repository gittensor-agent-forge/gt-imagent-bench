from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

# S4. The only stochastic stage in the pipeline.
#
# The judge is never asked to score an image out of 100. Absolute scores from a
# model drift between sessions, so a one-point difference between two runs weeks
# apart measures the judge's mood as much as the agents. It is asked the question
# models answer reliably instead: shown two images side by side, which is better?
#
# Three defences make that answer usable:
#   * blind      - the judge is never told which side is the incumbent;
#   * positioned - which agent occupies slot A is decided by a hash, so no side
#                  can learn a positional habit and neither knows its slot;
#   * repeated   - three independent votes, majority wins, which cancels most of
#                  the per-call randomness for very little money.

Side = Literal["king", "challenger", "tie"]
Slot = Literal["A", "B"]


@runtime_checkable
class PairwiseJudge(Protocol):
    def compare(
        self,
        *,
        prompt: str,
        image_a: Path,
        image_b: Path,
        notes_a: str = "",
        notes_b: str = "",
    ) -> str:
        """Return 'A', 'B', or 'tie'. Anything else is treated as a tie."""


@dataclass(frozen=True)
class JudgeVote:
    raw: str
    slot: str
    winner: Side
    error: str = ""


@dataclass(frozen=True)
class JudgeVerdict:
    winner: Side
    challenger_slot: Slot
    votes: tuple[JudgeVote, ...]

    @property
    def unanimous(self) -> bool:
        decided = [vote.winner for vote in self.votes if not vote.error]
        return len(decided) > 1 and len(set(decided)) == 1

    def to_dict(self) -> dict[str, object]:
        # Published verbatim: the judge stage is not reproducible, so its raw
        # answers are what make it auditable.
        return {
            "winner": self.winner,
            "challenger_slot": self.challenger_slot,
            "unanimous": self.unanimous,
            "votes": [
                {"raw": vote.raw, "slot": vote.slot, "winner": vote.winner, "error": vote.error}
                for vote in self.votes
            ],
        }


def challenger_slot(challenge_id: str, problem_id: str) -> Slot:
    """Which slot the challenger occupies for this problem.

    Derived from the challenge and problem ids, so it is unpredictable before the
    challenge is issued and reproducible by anyone afterwards. A fixed assignment
    would let a judge's positional preference favour the same side every time.
    """
    digest = sha256(f"imagent-slot-v1:{challenge_id}:{problem_id}".encode("utf-8")).digest()
    return "A" if digest[0] & 1 else "B"


def _normalize(raw: str) -> str:
    text = (raw or "").strip().casefold()
    if text.startswith("a"):
        return "A"
    if text.startswith("b"):
        return "B"
    # An unparseable answer is a tie, never a win. A judge that rambles must not
    # hand anyone a crown.
    return "tie"


def judge_problem(
    *,
    judge: PairwiseJudge,
    challenge_id: str,
    problem_id: str,
    prompt: str,
    king_image: Path,
    challenger_image: Path,
    king_notes: str = "",
    challenger_notes: str = "",
    votes: int = 3,
    slot: Slot | None = None,
) -> JudgeVerdict:
    """Run the blind comparison and return the majority verdict.

    `slot` overrides the derived position. It exists for the scheduled
    position-bias probe, which re-runs a comparison with the sides swapped: a
    judge that picks the same *slot* rather than the same *image* is biased.
    """
    if votes < 1:
        raise ValueError("votes must be positive")

    assigned = slot or challenger_slot(challenge_id, problem_id)
    if assigned == "A":
        image_a, image_b = challenger_image, king_image
        notes_a, notes_b = challenger_notes, king_notes
    else:
        image_a, image_b = king_image, challenger_image
        notes_a, notes_b = king_notes, challenger_notes

    cast: list[JudgeVote] = []
    for _ in range(votes):
        try:
            raw = judge.compare(
                prompt=prompt,
                image_a=image_a,
                image_b=image_b,
                notes_a=notes_a,
                notes_b=notes_b,
            )
        except Exception as exc:  # noqa: BLE001 - a failed call abstains, it does not crash
            cast.append(JudgeVote(raw="", slot="", winner="tie", error=f"{type(exc).__name__}: {exc}"[:200]))
            continue
        picked = _normalize(raw)
        cast.append(JudgeVote(raw=str(raw)[:200], slot=picked, winner=_winner_for(picked, assigned)))

    return JudgeVerdict(winner=_majority(cast), challenger_slot=assigned, votes=tuple(cast))


def _winner_for(picked: str, challenger_at: Slot) -> Side:
    if picked == "tie":
        return "tie"
    return "challenger" if picked == challenger_at else "king"


def _majority(votes: list[JudgeVote]) -> Side:
    """A strict majority of all votes cast, abstentions included.

    Counting abstentions in the denominator is deliberate: if two of three calls
    fail, one surviving opinion should not decide a crown.
    """
    if not votes:
        return "tie"
    counts = Counter(vote.winner for vote in votes if not vote.error)
    for side in ("challenger", "king"):
        if counts[side] * 2 > len(votes):
            return side  # type: ignore[return-value]
    return "tie"
