from __future__ import annotations

from pathlib import Path

import pytest

from imagent_scoring.judge import challenger_slot, judge_problem

KING = Path("king.png")
CHALLENGER = Path("challenger.png")


class _StubJudge:
    """Always picks whichever slot it was told to prefer."""

    def __init__(self, *picks: str) -> None:
        self.picks = list(picks)
        self.seen: list[tuple[Path, Path]] = []

    def compare(self, *, prompt, image_a, image_b, notes_a="", notes_b=""):
        self.seen.append((image_a, image_b))
        return self.picks.pop(0) if self.picks else "tie"


def _judge(*picks: str, slot=None, votes=3):
    stub = _StubJudge(*picks)
    verdict = judge_problem(
        judge=stub,
        challenge_id="c1",
        problem_id="p1",
        prompt="a red cup",
        king_image=KING,
        challenger_image=CHALLENGER,
        votes=votes,
        slot=slot,
    )
    return verdict, stub


# --- blinding ---------------------------------------------------------------


def test_the_slot_is_deterministic_and_reproducible() -> None:
    assert challenger_slot("c1", "p1") == challenger_slot("c1", "p1")


def test_the_slot_varies_across_problems_and_challenges() -> None:
    slots = {challenger_slot("c1", f"p{index}") for index in range(20)}
    assert slots == {"A", "B"}

    across = {challenger_slot(f"c{index}", "p1") for index in range(20)}
    assert across == {"A", "B"}


def test_the_challenger_is_placed_in_its_assigned_slot() -> None:
    verdict, stub = _judge("A", "A", "A", slot="A")

    assert verdict.challenger_slot == "A"
    assert stub.seen[0] == (CHALLENGER, KING)
    assert verdict.winner == "challenger"


def test_the_same_votes_favour_the_other_side_when_slots_swap() -> None:
    # This is the whole point of blinding: the judge picked slot A both times,
    # and the crown went to a different agent because the slot moved.
    as_a, _ = _judge("A", "A", "A", slot="A")
    as_b, _ = _judge("A", "A", "A", slot="B")

    assert as_a.winner == "challenger"
    assert as_b.winner == "king"


# --- majority ---------------------------------------------------------------


def test_a_two_to_one_split_is_a_majority() -> None:
    verdict, _ = _judge("A", "A", "B", slot="A")

    assert verdict.winner == "challenger"
    assert not verdict.unanimous


def test_a_unanimous_verdict_is_flagged() -> None:
    verdict, _ = _judge("B", "B", "B", slot="A")

    assert verdict.winner == "king"
    assert verdict.unanimous


def test_a_three_way_split_is_a_tie() -> None:
    verdict, _ = _judge("A", "B", "tie", slot="A")

    assert verdict.winner == "tie"


def test_an_unparseable_answer_never_wins() -> None:
    verdict, _ = _judge("I cannot decide", "also unclear", "hmm", slot="A")

    assert verdict.winner == "tie"


def test_one_surviving_vote_cannot_decide_a_crown() -> None:
    # Two calls failed. Counting abstentions in the denominator means the single
    # opinion left is not a majority of three.
    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def compare(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("provider down")
            return "A"

    verdict = judge_problem(
        judge=_Flaky(),
        challenge_id="c1",
        problem_id="p1",
        prompt="a red cup",
        king_image=KING,
        challenger_image=CHALLENGER,
        slot="A",
    )

    assert verdict.winner == "tie"
    assert sum(1 for vote in verdict.votes if vote.error) == 2


def test_a_judge_failure_is_recorded_not_raised() -> None:
    class _Broken:
        def compare(self, **kwargs):
            raise RuntimeError("provider down")

    verdict = judge_problem(
        judge=_Broken(),
        challenge_id="c1",
        problem_id="p1",
        prompt="a red cup",
        king_image=KING,
        challenger_image=CHALLENGER,
    )

    assert verdict.winner == "tie"
    assert all("provider down" in vote.error for vote in verdict.votes)


def test_votes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="votes must be positive"):
        _judge(votes=0)


def test_the_verdict_serialises_the_raw_answers() -> None:
    verdict, _ = _judge("A", "A", "B", slot="A")

    payload = verdict.to_dict()

    # The judge stage is not reproducible, so its raw answers are what make it
    # auditable after the fact.
    assert payload["challenger_slot"] == "A"
    assert [vote["raw"] for vote in payload["votes"]] == ["A", "A", "B"]
