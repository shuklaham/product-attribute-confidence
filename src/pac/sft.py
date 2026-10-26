"""Supervised fine-tuning data for M1.

Each training item is presented once per epoch, under one text condition drawn by a seeded coin
flip per (family, attribute, epoch), so every model sees the same draws and colour variants share
a condition. Its options are reshuffled per (product, attribute, epoch), so the position of the
answer letter carries no signal. The target is the JSON answer in the layout the base model
writes. The loss covers the letter and the JSON format but not the confidence value, because the
catalog has no true confidence to copy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .examples import LETTERS, option_order

SALT = "sft-v1"
# Stands in the target where the confidence goes. It is masked out of the loss, so it teaches
# nothing; it is the value the base model writes almost every time.
TARGET_CONFIDENCE = "0.95"


def condition_for(family_id: int, attribute: str, epoch: int, salt: str = SALT) -> str:
    """The text condition, full or redacted, for one family and attribute in one epoch."""
    digest = int(hashlib.sha256(f"{salt}:{family_id}:{attribute}:{epoch}".encode()).hexdigest(), 16)
    return "full" if digest % 2 == 0 else "redacted"


def reshuffle(example: dict, epoch: int, salt: str = SALT) -> dict:
    """The example with its options in a new seeded order for this epoch, and the answer letter to match."""
    definitions = {o["label"]: o["definition"] for o in example["options"]}
    order = option_order(example["product_id"], example["attribute"], sorted(definitions), salt=f"{salt}:{epoch}")
    return {**example,
            "options": [{"letter": LETTERS[i], "label": label, "definition": definitions[label]}
                        for i, label in enumerate(order)],
            "answer": LETTERS[order.index(example["reference"])]}


def presentations(examples: Sequence[dict], epochs: int) -> list[dict]:
    """The training presentations for all epochs: one per (product, attribute) and epoch.

    examples holds both text conditions of every item, as the example builder writes them.
    """
    out = []
    for epoch in range(epochs):
        chosen = {}
        for e in examples:
            if e["condition"] == condition_for(e["family_id"], e["attribute"], epoch):
                chosen[(e["product_id"], e["attribute"])] = reshuffle(e, epoch)
        out += [chosen[key] for key in sorted(chosen)]
    return out


def target_text(letter: str, confidence: str = TARGET_CONFIDENCE) -> tuple[str, tuple[int, int]]:
    """The JSON answer in the base model's own layout, and the character span of the confidence value."""
    head = '{\n  "choice": "' + letter + '",\n  "confidence": '
    return head + confidence + "\n}", (len(head), len(head) + len(confidence))


def completion_labels(token_ids: Sequence[int], offsets: Sequence[tuple[int, int]],
                      masked: tuple[int, int]) -> list[int]:
    """Training labels for the completion: each token's id, or -100 (ignored) where it overlaps the masked span."""
    start, end = masked
    return [-100 if s < end and e > start else t for t, (s, e) in zip(token_ids, offsets)]
