"""Metrics for answers that come with a stated confidence.

Inputs are plain sequences or numpy arrays. `correct` holds 1 for a right answer and 0 for a
wrong or invalid one. `confidence` holds the stated confidence, or None (or NaN) when the output
was invalid. Invalid outputs always count as wrong for accuracy, and they are never accepted for
auto-fill.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


def _confidence(values) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype.kind == "f":
        return values
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _correct(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def accuracy(correct) -> float:
    return float(_correct(correct).mean())


def macro_accuracy(correct, groups: Sequence) -> float:
    """Accuracy averaged equally over groups, such as (category, attribute) pairs."""
    totals: dict = {}
    for c, g in zip(correct, groups):
        hits, n = totals.get(g, (0.0, 0))
        totals[g] = (hits + float(c), n + 1)
    return sum(hits / n for hits, n in totals.values()) / len(totals)


def brier(confidence, correct, invalid_loss: float | None = None) -> float:
    """Mean of (confidence - correct)^2 over valid outputs. Lower is better.

    With invalid_loss=1.0, every invalid output adds a loss of 1, which gives the "Brier score
    with invalid-output penalty".
    """
    q, c = _confidence(confidence), _correct(correct)
    valid = ~np.isnan(q)
    losses = (q[valid] - c[valid]) ** 2
    if invalid_loss is not None:
        losses = np.concatenate([losses, np.full(int((~valid).sum()), invalid_loss)])
    return float(losses.mean())


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability(confidence, correct, bins: int = 10) -> list[Bin]:
    """Equal-width confidence bins over [0, 1], the data behind a reliability diagram.

    Confidence 1.0 falls in the last bin. Invalid outputs are skipped.
    """
    q, c = _confidence(confidence), _correct(correct)
    valid = ~np.isnan(q)
    q, c = q[valid], c[valid]
    index = np.minimum((q * bins).astype(int), bins - 1)
    table = []
    for b in range(bins):
        members = index == b
        n = int(members.sum())
        table.append(Bin(b / bins, (b + 1) / bins, n,
                         float(q[members].mean()) if n else float("nan"),
                         float(c[members].mean()) if n else float("nan")))
    return table


def ece(confidence, correct, bins: int = 10) -> float:
    """Expected calibration error: the count-weighted mean gap between accuracy and confidence per bin.

    Invalid outputs are skipped. Returns NaN when there are no valid outputs.
    """
    table = reliability(confidence, correct, bins)
    total = sum(b.count for b in table)
    if total == 0:
        return float("nan")
    return float(sum(b.count / total * abs(b.accuracy - b.mean_confidence) for b in table if b.count))


def auroc(confidence, correct) -> float:
    """Chance that a random right answer has higher confidence than a random wrong one; ties count half.

    It measures ranking only, so a monotone recalibration such as a Platt map leaves it unchanged.
    Invalid outputs are skipped. Returns NaN when every valid answer is right, or every one is wrong.
    """
    q, c = _confidence(confidence), _correct(correct)
    valid = ~np.isnan(q)
    q, c = q[valid], c[valid]
    positives = c.sum()
    negatives = len(c) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    _, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2
    ranks = average_rank[inverse]
    u = ranks[c == 1].sum() - positives * (positives + 1) / 2
    return float(u / (positives * negatives))


@dataclass(frozen=True)
class CoveragePoint:
    threshold: float
    accepted: int
    coverage: float
    risk: float


def risk_coverage(confidence, correct) -> list[CoveragePoint]:
    """Accept every output with confidence at or above each distinct threshold, from the highest down.

    Coverage is the accepted share of all items, so invalid outputs, which are never accepted,
    stay in the denominator. Risk is the error rate among the accepted outputs.
    """
    q, c = _confidence(confidence), _correct(correct)
    valid = ~np.isnan(q)
    values, inverse, counts = np.unique(q[valid], return_inverse=True, return_counts=True)
    errors = np.bincount(inverse, weights=1 - c[valid], minlength=len(values))
    accepted = np.cumsum(counts[::-1])
    wrong = np.cumsum(errors[::-1])
    return [CoveragePoint(float(t), int(a), a / len(q), float(w / a))
            for t, a, w in zip(values[::-1], accepted, wrong)]


def aurc(confidence, correct) -> float:
    """Area under the risk-coverage curve: the mean, over k = 1..n, of the error rate among the top k items.

    Items are ranked by confidence, with invalid outputs last and counted as wrong, so emitting
    an invalid output on a hard item never improves the area. Tied confidences have no order, so
    the result is the expected value over a random order within each tie, which interpolates the
    cumulative error count linearly across the tie. Lower is better.
    """
    q, c = _confidence(confidence), _correct(correct)
    points = risk_coverage(q, c)
    accepted = [0] + [p.accepted for p in points]
    errors = [0.0] + [p.risk * p.accepted for p in points]
    invalid = len(q) - accepted[-1]
    accepted.append(len(q))
    errors.append(errors[-1] + invalid)
    k = np.arange(1, len(q) + 1)
    return float(np.mean(np.interp(k, accepted, errors) / k))


def select_threshold(confidence, correct, max_risk: float = 0.05) -> float:
    """The lowest threshold whose accepted outputs have risk at most max_risk, giving the most coverage.

    Returns infinity, meaning accept nothing, when no threshold qualifies. Choose it on the
    threshold half of the calibration split, then freeze it for the test sets.
    """
    qualifying = [p.threshold for p in risk_coverage(confidence, correct) if p.risk <= max_risk]
    return min(qualifying) if qualifying else float("inf")


def at_threshold(confidence, correct, threshold: float) -> tuple[float, float, int]:
    """Coverage, risk among accepted outputs, and the accepted count for a frozen threshold."""
    q, c = _confidence(confidence), _correct(correct)
    accepted = ~np.isnan(q) & (q >= threshold)
    n_accepted = int(accepted.sum())
    risk = float((1 - c[accepted]).mean()) if n_accepted else float("nan")
    return n_accepted / len(q), risk, n_accepted


def confidence_and_accuracy_drop(full_confidence, full_correct, redacted_confidence,
                                 redacted_correct) -> tuple[float, float]:
    """For the same items under both text conditions: the fall in mean confidence and the fall in accuracy.

    When the text stops stating the answer, a well-calibrated model lowers its confidence about
    as much as its accuracy falls. Mean confidence uses items that are valid in both conditions.
    """
    qf, qr = _confidence(full_confidence), _confidence(redacted_confidence)
    both = ~np.isnan(qf) & ~np.isnan(qr)
    confidence_drop = float(qf[both].mean() - qr[both].mean())
    return confidence_drop, accuracy(full_correct) - accuracy(redacted_correct)


def family_bootstrap(statistic: Callable[[np.ndarray], float], families: Sequence, resamples: int = 10_000,
                     seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Point estimate and percentile interval of statistic(rows), resampling product families with replacement.

    statistic receives row indices, with repeats, and computes its value on those rows. Every
    row of a family is drawn together, so colour variants never count as independent evidence.
    For a paired difference between two methods, compute both methods on the same rows inside
    the statistic and return the difference.
    """
    _, family_index = np.unique(np.asarray([str(f) for f in families]), return_inverse=True)
    n_families = int(family_index.max()) + 1
    rng = np.random.default_rng(seed)
    rows = np.arange(len(family_index))
    draws = np.empty(resamples)
    for i in range(resamples):
        counts = np.bincount(rng.integers(0, n_families, n_families), minlength=n_families)
        draws[i] = statistic(np.repeat(rows, counts[family_index]))
    low, high = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float(statistic(rows)), float(low), float(high)
