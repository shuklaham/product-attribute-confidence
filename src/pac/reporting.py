"""Helpers the phase reports share: loading prediction runs, formatting, and per-method metrics.

The reports themselves are scripts (`scripts/report_phase1.py`, `scripts/report_phase2.py`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import methods, metrics
from .calibration import Temperature, logit, sigmoid


def load_run(artifacts: Path, name: str) -> list[dict]:
    """The prediction records of one run, checked against the count in its run record."""
    path = artifacts / name
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    with open(path / "predictions.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if len(records) != run["n"]:
        raise ValueError(f"{name}: {len(records)} records but the run record says {run['n']}")
    return records


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def num(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def evaluate(records: list[dict], q: list, thresholds: dict) -> dict:
    """Metrics for one method on one set of records, pooled over attributes.

    thresholds maps (attribute, condition) to the method's frozen auto-fill threshold.
    """
    correct = np.array([r["correct"] for r in records], dtype=float)
    accepted = np.array([v is not None and v >= thresholds[methods.group_key(r)] for r, v in zip(records, q)])
    n_accepted = int(accepted.sum())
    return {
        "n": len(records),
        "accuracy": float(correct.mean()),
        "invalid": float(np.mean([r["error"] is not None for r in records])),
        "brier": metrics.brier(q, correct, invalid_loss=1.0),
        "ece": metrics.ece(q, correct),
        "auroc": metrics.auroc(q, correct),
        "aurc": metrics.aurc(q, correct),
        "coverage": n_accepted / len(records),
        "risk": float((1 - correct[accepted]).mean()) if n_accepted else float("nan"),
    }


def arrays(records: list[dict]) -> dict:
    """Per-record arrays for the bootstrap: stated confidence, correctness, letter log-probabilities."""
    width = max(len(r["letter_logprobs"]) for r in records if r["letter_logprobs"])
    rows, answers, chosen = [], [], []
    for r in records:
        letters = list(r["letter_logprobs"] or {})
        rows.append([r["letter_logprobs"][k] for k in letters] if letters else [0.0] * width)
        answers.append(letters.index(r["answer"]) if letters else 0)
        chosen.append(letters.index(r["choice"]) if r["error"] is None and r["choice"] in letters else -1)
    return {"stated": np.array([np.nan if r["confidence"] is None else r["confidence"] for r in records]),
            "correct": np.array([r["correct"] for r in records], dtype=float),
            "letters": np.array(rows), "answer": np.array(answers), "chosen": np.array(chosen),
            "has_letters": np.array([bool(r["letter_logprobs"]) for r in records]),
            "family": np.array([r["family_id"] for r in records])}


def mapped(a: dict, rows: np.ndarray, platt, t: float) -> tuple[np.ndarray, np.ndarray]:
    """M2 and M2-logit confidences for the given rows, NaN where a method has no confidence."""
    stated = a["stated"][rows]
    m2 = np.where(np.isnan(stated), np.nan, sigmoid(platt.a * logit(np.nan_to_num(stated, nan=0.5)) + platt.b))
    probabilities = Temperature(t).probabilities(a["letters"][rows])
    chosen = a["chosen"][rows]
    m2_logit = np.where(chosen >= 0, probabilities[np.arange(len(rows)), np.maximum(chosen, 0)], np.nan)
    return m2, m2_logit


def family_index(by_attribute: dict) -> tuple[dict, int]:
    """For records grouped by attribute, each row's family number in one shared numbering, and the count.

    A family's rows in both attributes get the same number, so a bootstrap draw keeps them together.
    """
    values = np.concatenate([by_attribute[k]["family"] for k in by_attribute])
    _, index = np.unique(values.astype(str), return_inverse=True)
    out, start = {}, 0
    for k in by_attribute:
        n = len(by_attribute[k]["family"])
        out[k], start = index[start:start + n], start + n
    return out, int(index.max()) + 1
