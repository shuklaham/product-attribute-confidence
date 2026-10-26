"""Post-hoc calibration for the two cheap baselines.

M2 fits a Platt map to the stated confidence: q' = sigmoid(a * logit(q) + b), with a >= 0 so
the map never reverses the order of confidences. M2-logit rescales the log-probabilities the
model gives each offered option letter by a temperature. Both are fitted on the map-fitting half
of the calibration split and then frozen.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# 0 and 1 have no finite logit, so stated confidence is clipped to [EPSILON, 1 - EPSILON]. Models
# state confidence in steps of 0.01 or coarser, so a stated 1.0 is read as 0.995, half a step above
# 0.99, and 0.0 as 0.005. Fixed before any model was trained; values beyond the clip tie.
EPSILON = 0.005


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


def _fit_intercept(x: np.ndarray, y: np.ndarray, a: float, l2: float, iterations: int = 100) -> float:
    """The b that minimises cross-entropy for a fixed slope a, by Newton's method."""
    b = 0.0
    for _ in range(iterations):
        p = sigmoid(a * x + b)
        step = (np.sum(p - y) + l2 * b) / (np.sum(p * (1 - p)) + l2)
        b -= step
        if abs(step) < 1e-10:
            break
    return float(b)


def fit_platt(confidence, correct, l2: float = 1e-3, iterations: int = 100) -> PlattMap:
    """Fit a and b by minimising binary cross-entropy with Newton's method, keeping a >= 0.

    A small L2 penalty keeps the fit finite when every answer is right or every answer is
    wrong. If the unconstrained step would make a negative, a stays at 0 and only b is fitted:
    the map then returns the base rate for every confidence. When every stated confidence is the
    same, the data say nothing about the slope, so a stays at 1 and only the level moves.
    """
    q, c = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    valid = ~np.isnan(q)
    if not valid.any():
        raise ValueError("no valid outputs to fit a Platt map on")
    x, y = logit(q[valid]), c[valid]
    if np.ptp(x) == 0:
        return PlattMap(1.0, _fit_intercept(x, y, 1.0, l2))
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

    logprobs has one row per item and one column per offered option, so each attribute is
    fitted on its own; answers holds each item's correct option index. Golden-section search
    on log t, which suits this one-dimensional, single-dip objective. Non-finite inputs raise
    an error, and a result at a search bound gives a warning.
    """
    z, y = np.asarray(logprobs, dtype=float), np.asarray(answers, dtype=int)
    if z.ndim != 2 or len(z) != len(y) or not np.isfinite(z).all() or y.min() < 0 or y.max() >= z.shape[1]:
        raise ValueError("logprobs must be a finite 2-D array with one row per answer, and answers valid indices")
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
    t = float(np.exp((lo + hi) / 2))
    if min(t / low, high / t) < 1.01:
        warnings.warn(f"the fitted temperature {t:.3g} is at the edge of the search range", stacklevel=2)
    return Temperature(t)
