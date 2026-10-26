"""GRPO rewards for the two RL conditions.

M3 rewards correctness only. M4 adds a Brier penalty on the stated confidence, adapting RLCR
(arXiv:2507.16806). The Brier score is a proper scoring rule: if the answer is correct with
probability p, the expected M4 reward is highest when the stated confidence equals p.
"""

from __future__ import annotations

from .output import Parsed

INVALID_REWARD = -1.1
KINDS = ("correctness", "calibration")


def correctness_reward(correct: bool) -> float:
    return 1.0 if correct else 0.0


def calibration_reward(correct: bool, confidence: float) -> float:
    c = 1.0 if correct else 0.0
    return c - (confidence - c) ** 2


def reward(parsed: Parsed, answer: str, kind: str) -> float:
    """Reward for one sampled output. Invalid output gets the same penalty in both conditions."""
    if kind not in KINDS:
        raise ValueError(f"unknown reward kind: {kind}")
    if not parsed.valid:
        return INVALID_REWARD
    correct = parsed.choice == answer
    if kind == "correctness":
        return correctness_reward(correct)
    return calibration_reward(correct, parsed.confidence)
