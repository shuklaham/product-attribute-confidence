"""Post-hoc calibration for the two cheap baselines.

M2 fits a Platt map to the stated confidence: q' = sigmoid(a * logit(q) + b), with a >= 0 so
the map never reverses the order of confidences. M2-logit rescales the log-probabilities the
model gives each offered option letter by a temperature. Both are fitted on the map-fitting half
of the calibration split and then frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-4


def logit(q) -> np.ndarray:
    q = np.clip(np.asarray(q, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(q / (1 - q))


def sigmoid(z) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


@dataclass(frozen=True)
class PlattMap:
    a: float
    b: float

    def __call__(self, confidence) -> np.ndarray:
        """Calibrated confidence. Invalid outputs (NaN) stay NaN."""
        return sigmoid(self.a * logit(confidence) + self.b)


def fit_platt(confidence, correct, l2: float = 1e-3, iterations: int = 100) -> PlattMap:
    """Fit a and b by minimising binary cross-entropy with Newton's method, keeping a >= 0.

    A small L2 penalty keeps the fit finite when every answer is right or every answer is
    wrong. If the unconstrained step would make a negative, a stays at 0 and only b is fitted:
    the map then returns the base rate for every confidence.
    """
    q, c = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    valid = ~np.isnan(q)
    x, y = logit(q[valid]), c[valid]
    a, b = 1.0, 0.0
    for _ in range(iterations):
        p = sigmoid(a * x + b)
        w = p * (1 - p)
        gradient = np.array([np.dot(p - y, x) + l2 * a, np.sum(p - y) + l2 * b])
        hessian = np.array([[np.dot(w, x * x) + l2, np.dot(w, x)], [np.dot(w, x), np.sum(w) + l2]])
        step = np.linalg.solve(hessian, gradient)
        new_a, new_b = a - step[0], b - step[1]
        if new_a < 0:
            new_a = 0.0
            p0 = sigmoid(b)
            new_b = b - (np.sum(p0 - y) + l2 * b) / (len(y) * p0 * (1 - p0) + l2)
        converged = abs(new_a - a) < 1e-10 and abs(new_b - b) < 1e-10
        a, b = float(new_a), float(new_b)
        if converged:
            break
    return PlattMap(a, b)


@dataclass(frozen=True)
class Temperature:
    t: float

    def probabilities(self, logprobs) -> np.ndarray:
        """Softmax over each row of option log-probabilities, after dividing by the temperature."""
        z = np.asarray(logprobs, dtype=float) / self.t
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logprobs, answers, low: float = 0.05, high: float = 20.0, iterations: int = 100) -> Temperature:
    """The temperature that minimises the negative log-likelihood of the correct options.

    logprobs has one row per item and one column per offered option; answers holds each item's
    correct option index. Golden-section search on log t, which suits this one-dimensional,
    single-dip objective.
    """
    z, y = np.asarray(logprobs, dtype=float), np.asarray(answers, dtype=int)
    rows = np.arange(len(y))

    def nll(log_t: float) -> float:
        p = Temperature(float(np.exp(log_t))).probabilities(z)
        return float(-np.mean(np.log(p[rows, y] + 1e-12)))

    lo, hi = np.log(low), np.log(high)
    ratio = (np.sqrt(5) - 1) / 2
    for _ in range(iterations):
        c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
    return Temperature(float(np.exp((lo + hi) / 2)))
