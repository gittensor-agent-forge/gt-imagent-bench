from __future__ import annotations

from .answer_key import AnswerKeyError, load_answer_key, parse_answer_key
from .facts import MissingBackendError, aggregate, score_facts
from .imaging import ImageData, hamming_distance, phash
from .judge import JudgeVerdict, PairwiseJudge, challenger_slot, judge_problem
from .ladder import ProblemVerdict, SideResult, decide_problem, mean_fact_score
from .models import (
    AnswerKey,
    CheckResult,
    ChecklistQuestion,
    Detection,
    FactReport,
    ObjectRequirement,
    RelationRequirement,
    TextRequirement,
    TextSpan,
    ValidityReport,
)
from .validity import check_validity

__all__ = [
    "AnswerKey",
    "AnswerKeyError",
    "CheckResult",
    "ChecklistQuestion",
    "Detection",
    "FactReport",
    "ImageData",
    "JudgeVerdict",
    "MissingBackendError",
    "ObjectRequirement",
    "PairwiseJudge",
    "ProblemVerdict",
    "SideResult",
    "RelationRequirement",
    "TextRequirement",
    "TextSpan",
    "ValidityReport",
    "aggregate",
    "challenger_slot",
    "check_validity",
    "decide_problem",
    "hamming_distance",
    "judge_problem",
    "load_answer_key",
    "mean_fact_score",
    "parse_answer_key",
    "phash",
    "score_facts",
]
