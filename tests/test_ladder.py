from __future__ import annotations

from imagent_scoring.judge import JudgeVerdict, JudgeVote
from imagent_scoring.ladder import SideResult, decide_problem, mean_fact_score


def _side(fact: float = 0.9, *, valid: bool = True, preference: float | None = None, error: str = ""):
    return SideResult(valid=valid, fact_score=fact, preference=preference, error=error)


def _judge(winner: str) -> JudgeVerdict:
    return JudgeVerdict(
        winner=winner,  # type: ignore[arg-type]
        challenger_slot="A",
        votes=(JudgeVote(raw="A", slot="A", winner=winner),) * 3,  # type: ignore[arg-type]
    )


def _decide(king, challenger, judge=None):
    return decide_problem(problem_id="p1", king=king, challenger=challenger, judge=judge)


# --- rule 1: validity -------------------------------------------------------


def test_an_unusable_challenger_image_loses_outright() -> None:
    verdict = _decide(_side(0.2), _side(valid=False, error="blank image"), _judge("challenger"))

    assert verdict.winner == "king"
    assert verdict.rule == "validity"
    assert "blank image" in verdict.detail


def test_two_unusable_images_are_a_tie() -> None:
    verdict = _decide(_side(valid=False), _side(valid=False))

    assert verdict.winner == "tie"
    assert verdict.rule == "validity"


# --- rule 2: facts beat taste ----------------------------------------------


def test_a_decisive_fact_gap_overrules_the_judge() -> None:
    # The judge preferred the challenger's picture. It dropped a required object.
    verdict = _decide(_side(0.95), _side(0.70), _judge("challenger"))

    assert verdict.winner == "king"
    assert verdict.rule == "fact_dominance"


def test_the_gap_works_in_the_challenger_s_favour_too() -> None:
    verdict = _decide(_side(0.60), _side(0.90), _judge("king"))

    assert verdict.winner == "challenger"
    assert verdict.rule == "fact_dominance"


def test_a_gap_just_below_the_threshold_leaves_it_to_the_judge() -> None:
    verdict = _decide(_side(0.90), _side(0.76), _judge("challenger"))

    assert verdict.winner == "challenger"
    assert verdict.rule == "judge"


def test_a_gap_exactly_at_the_threshold_is_decisive() -> None:
    verdict = _decide(_side(0.90), _side(0.75), _judge("challenger"))

    assert verdict.rule == "fact_dominance"
    assert verdict.winner == "king"


# --- rules 3-5: judge, preference, tie --------------------------------------


def test_a_close_call_goes_to_the_blind_judge() -> None:
    verdict = _decide(_side(0.90), _side(0.88), _judge("challenger"))

    assert verdict.winner == "challenger"
    assert verdict.rule == "judge"


def test_preference_only_speaks_after_the_judge_ties() -> None:
    verdict = _decide(
        _side(0.90, preference=0.50), _side(0.88, preference=0.75), _judge("tie")
    )

    assert verdict.winner == "challenger"
    assert verdict.rule == "preference"


def test_a_narrow_preference_difference_is_not_enough() -> None:
    verdict = _decide(
        _side(0.90, preference=0.60), _side(0.88, preference=0.65), _judge("tie")
    )

    assert verdict.winner == "tie"
    assert verdict.rule == "tie"


def test_two_indistinguishable_images_tie() -> None:
    verdict = _decide(_side(0.90), _side(0.90))

    assert verdict.winner == "tie"

def test_mean_fact_score_averages_one_side() -> None:
    verdicts = [
        _decide(_side(0.90), _side(0.80), _judge("king")),
        _decide(_side(0.70), _side(1.00), _judge("challenger")),
    ]

    assert mean_fact_score(verdicts, side="king") == 0.80
    assert mean_fact_score(verdicts, side="challenger") == 0.90
