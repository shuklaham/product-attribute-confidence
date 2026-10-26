"""From raw generations to scored predictions.

vLLM runs only on the GPU pod (`scripts/predict.py`). The rules that decide what the model
answered, and what probability it gave each option letter, live here as plain Python so they
can be tested anywhere.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .examples import LETTERS
from .output import parse_output

MODES = ("photo_text", "text_only", "photo_only")
CHOICE_PREFIX = re.compile(r'"choice"\s*:\s*"')


def evidence_view(example: dict, mode: str) -> tuple[dict, bool]:
    """The example as the model sees it under an evidence mode, and whether the photo is shown.

    The instructions and options never change; only the evidence is removed.
    """
    if mode == "photo_text":
        return example, True
    if mode == "text_only":
        return example, False
    if mode == "photo_only":
        return {**example, "title": "", "description": ""}, True
    raise ValueError(f"unknown mode: {mode}")


def letter_token_ids(encode: Callable[[str], list[int]]) -> dict[str, int]:
    """Token id of every option letter. Each letter must be a single token."""
    ids = {}
    for letter in LETTERS:
        tokens = encode(letter)
        if len(tokens) != 1:
            raise ValueError(f"option letter {letter!r} is {len(tokens)} tokens")
        ids[letter] = tokens[0]
    return ids


def choice_token_index(pieces: Sequence[str]) -> int | None:
    """Index of the generated token that starts right after `"choice": "`, or None.

    pieces are the generated tokens, decoded one at a time. The Qwen tokenizer splits every
    JSON layout seen so far (spaced, compact, fenced, multi-line) so that the option letter is
    a token of its own at this point. If a token straddles the opening quote, there is no such
    position and the result is None.
    """
    match = CHOICE_PREFIX.search("".join(pieces))
    if match is None:
        return None
    offset = 0
    for i, piece in enumerate(pieces):
        if offset == match.end():
            return i
        if offset > match.end():
            return None
        offset += len(piece)
    return None


def letter_logprobs(pieces: Sequence[str], logprobs: Sequence[dict[int, float]], letter_ids: dict[str, int],
                    letters: Sequence[str]) -> dict[str, float] | None:
    """The log-probability of each offered letter where the model writes its choice, or None.

    logprobs holds, for every generated position, a map from token id to log-probability that
    must include every letter id.
    """
    index = choice_token_index(pieces)
    if index is None:
        return None
    return {letter: logprobs[index][letter_ids[letter]] for letter in letters}


def score(example: dict, text: str, letter_logprobs: dict[str, float] | None) -> dict:
    """One prediction record: the strict parse of the output, its correctness and the letter log-probabilities."""
    parsed = parse_output(text, [o["letter"] for o in example["options"]])
    keys = ("example_id", "product_id", "family_id", "category", "split", "attribute", "condition", "answer",
            "reference")
    return {
        **{k: example[k] for k in keys},
        "output": text,
        "choice": parsed.choice,
        "confidence": parsed.confidence,
        "error": parsed.error,
        "correct": int(parsed.valid and parsed.choice == example["answer"]),
        "letter_logprobs": letter_logprobs,
    }
