"""Summaries of prediction records: the numbers behind each results table."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from . import metrics
from .calibration import Temperature


def letter_confidence(record: dict, temperature: float = 1.0) -> float | None:
    """The probability of the chosen letter among the offered letters, after dividing by a temperature.

    None when the output was invalid or the choice position was not found.
    """
    logprobs = record["letter_logprobs"]
    if record["choice"] is None or logprobs is None:
        return None
    letters = list(logprobs)
    p = Temperature(temperature).probabilities(np.array([[logprobs[letter] for letter in letters]]))[0]
    return float(p[letters.index(record["choice"])])


def category_guess(train_examples: list[dict]):
    """A function giving, for a record, the most common training label of its category.

    Categories absent from training (the held-out ones) get the attribute's most common label.
    """
    by_category, by_attribute = defaultdict(Counter), defaultdict(Counter)
    for e in train_examples:
        if e["condition"] == "full":
            by_category[(e["attribute"], e["category"])][e["reference"]] += 1
            by_attribute[e["attribute"]][e["reference"]] += 1

    def guess(record: dict) -> str:
        counts = by_category.get((record["attribute"], record["category"])) or by_attribute[record["attribute"]]
        return counts.most_common(1)[0][0]

    return guess


def summarize(records: list[dict], resamples: int = 10_000, guess=None) -> dict:
    """Metrics for one group of records, such as one attribute under one text condition and evidence mode."""
    correct = np.array([r["correct"] for r in records], dtype=float)
    stated = [r["confidence"] for r in records]
    letter = [letter_confidence(r) for r in records]
    accuracy, low, high = metrics.family_bootstrap(lambda rows: float(correct[rows].mean()),
                                                   [r["family_id"] for r in records], resamples)
    values = Counter(q for q in stated if q is not None)
    n_valid = sum(values.values())
    top_value, top_count = values.most_common(1)[0] if values else (None, 0)
    return {
        "n": len(records),
        "accuracy": accuracy,
        "accuracy_interval": (low, high),
        "macro_accuracy": metrics.macro_accuracy(correct, [r["category"] for r in records]),
        "category_guess": (float(np.mean([guess(r) == r["reference"] for r in records]))
                           if guess is not None else None),
        "invalid": 1 - n_valid / len(records),
        "errors": dict(Counter(r["error"] for r in records if r["error"])),
        "brier": metrics.brier(stated, correct),
        "brier_invalid_as_1": metrics.brier(stated, correct, invalid_loss=1.0),
        "ece": metrics.ece(stated, correct),
        "auroc": metrics.auroc(stated, correct),
        "aurc": metrics.aurc(stated, correct),
        "mean_confidence": sum(q * k for q, k in values.items()) / n_valid if n_valid else float("nan"),
        "distinct_confidences": len(values),
        "top_confidence": top_value,
        "top_confidence_share": top_count / n_valid if n_valid else float("nan"),
        "letter_missing": sum(r["choice"] is not None and r["letter_logprobs"] is None for r in records),
        "letter_ece": metrics.ece(letter, correct),
        "letter_auroc": metrics.auroc(letter, correct),
        "letter_aurc": metrics.aurc(letter, correct),
    }
