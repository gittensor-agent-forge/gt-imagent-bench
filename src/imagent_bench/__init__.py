"""Benchmark problem generation.

Given a seed, produce problems and the answer keys that grade them. Scoring the
images lives in `imagent_scoring`; deciding what a result means for a crown lives
in the competition repository.
"""

from .problems import GENERATOR_VERSION, Problem, generate_problems, new_challenge_id

__all__ = ["GENERATOR_VERSION", "Problem", "generate_problems", "new_challenge_id"]
