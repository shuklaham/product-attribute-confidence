"""Strict parsing of model output. Invalid output is never repaired."""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class Parsed:
    choice: str | None
    confidence: float | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.error is None


class _DuplicateKey(ValueError):
    pass


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    if len({key for key, _ in pairs}) != len(pairs):
        raise _DuplicateKey
    return dict(pairs)


def parse_output(text: str, letters: Collection[str]) -> Parsed:
    """Accept exactly {"choice": <offered letter>, "confidence": <finite number in [0, 1]>}.

    A repeated key is invalid: JSON would keep the last value, while the letter probabilities
    are read where the first one was written.
    """
    if isinstance(letters, str):
        raise TypeError("letters must be a collection of option letters, not one string")
    try:
        obj = json.loads(text.strip(), object_pairs_hook=_unique_keys)
    except _DuplicateKey:
        return Parsed(None, None, "duplicate_keys")
    except (json.JSONDecodeError, AttributeError):
        return Parsed(None, None, "malformed_json")
    if not isinstance(obj, dict):
        return Parsed(None, None, "not_an_object")
    if set(obj) != {"choice", "confidence"}:
        return Parsed(None, None, "wrong_keys")
    choice, confidence = obj["choice"], obj["confidence"]
    if not isinstance(choice, str) or choice not in letters:
        return Parsed(None, None, "unknown_choice")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return Parsed(None, None, "confidence_not_a_number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return Parsed(None, None, "confidence_out_of_range")
    return Parsed(choice, confidence, None)
