"""The confidence methods compared in phase 1, built on prediction records.

- stated: the confidence the model writes (M0, M1).
- platt: M1's stated confidence through a Platt map (M2).
- letter: the probability M1 gives its chosen letter, after temperature scaling (M2-logit).

Maps are fitted per attribute and text condition on the map-fitting half of the calibration
split, because both the evidence and the label set differ between those groups. Every method
keeps M1's answers, so only the confidence attached to them changes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import metrics
from .calibration import PlattMap, Temperature, fit_platt, fit_temperature

METHODS = ("stated", "platt", "letter")


def group_key(record: dict) -> tuple[str, str]:
    return record["attribute"], record["condition"]


def letter_rows(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Letter log-probabilities as a matrix, the correct letter's column, and the chosen letter's column.

    Only records whose choice position was read are included; a chosen column of -1 marks an
    invalid output.
    """
    rows, answers, chosen = [], [], []
    for r in records:
        logprobs = r["letter_logprobs"]
        letters = list(logprobs)
        rows.append([logprobs[letter] for letter in letters])
        answers.append(letters.index(r["answer"]))
        chosen.append(letters.index(r["choice"]) if r["choice"] in letters else -1)
    return np.array(rows, dtype=float), np.array(answers), np.array(chosen)


@dataclass(frozen=True)
class Fitted:
    platt: dict[tuple[str, str], PlattMap]
    temperature: dict[tuple[str, str], Temperature]


def fit(records: list[dict], temperature_iterations: int = 100) -> Fitted:
    """Fit the Platt map and the temperature per (attribute, condition) on map-fitting records."""
    groups = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(r)
    platt, temperature = {}, {}
    for key, rows in groups.items():
        platt[key] = fit_platt([np.nan if r["confidence"] is None else r["confidence"] for r in rows],
                               [r["correct"] for r in rows])
        with_letters = [r for r in rows if r["letter_logprobs"] is not None]
        logprobs, answers, _ = letter_rows(with_letters)
        temperature[key] = fit_temperature(logprobs, answers, iterations=temperature_iterations)
    return Fitted(platt, temperature)


def confidence(records: list[dict], method: str, fitted: Fitted | None = None) -> list[float | None]:
    """Each record's confidence under a method; None for an invalid output or an unreadable choice position."""
    if method == "stated":
        return [r["confidence"] for r in records]
    out = []
    for r in records:
        if r["error"] is not None:
            out.append(None)
        elif method == "platt":
            out.append(float(fitted.platt[group_key(r)](np.array([r["confidence"]]))[0]))
        elif method == "letter":
            if r["letter_logprobs"] is None:
                out.append(None)
                continue
            logprobs, _, chosen = letter_rows([r])
            out.append(float(fitted.temperature[group_key(r)].probabilities(logprobs)[0, chosen[0]]))
        else:
            raise ValueError(f"unknown method: {method}")
    return out


def thresholds(records: list[dict], method: str, fitted: Fitted | None = None,
               max_risk: float = 0.05) -> dict[tuple[str, str], float]:
    """The auto-fill threshold per (attribute, condition), chosen on threshold-half records."""
    q = confidence(records, method, fitted)
    groups = defaultdict(list)
    for r, value in zip(records, q):
        groups[group_key(r)].append((value, r["correct"]))
    return {key: metrics.select_threshold([v for v, _ in pairs], [c for _, c in pairs], max_risk)
            for key, pairs in groups.items()}
