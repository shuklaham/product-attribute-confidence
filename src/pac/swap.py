"""Photo swap: an item's text with another product's photo that has a different label.

An evaluation-only ablation (docs/plan.md, section 6). On a fixed subset of clean items, each
item's photo is replaced by a donor's: a clean product from the same split and category, outside
the item's family, whose catalog label for the attribute differs. The item keeps its text, options
and answer; the donor's label is recorded as `photo_answer` in the item's own letters. Subset and
donors come from seeded hashes, so the swaps are fixed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

SALT = "swap-v1"
PER_ATTRIBUTE = 500


def _rank(*parts) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in (SALT, *parts)).encode()).hexdigest(), 16)


def donors(examples: Sequence[dict]) -> dict[tuple[int, str], int]:
    """For every clean item that has one, the donor product for its attribute.

    examples may hold both text conditions of each item; only labels and categories are used.
    """
    info = {(e["product_id"], e["attribute"]): (e["category"], e["family_id"], e["reference"])
            for e in examples if e["clean"]}
    pool = defaultdict(list)
    for (pid, attribute), (category, _, _) in info.items():
        pool[(attribute, category)].append(pid)
    out = {}
    for (pid, attribute), (category, family, label) in sorted(info.items()):
        eligible = [d for d in pool[(attribute, category)]
                    if info[(d, attribute)][1] != family and info[(d, attribute)][2] != label]
        if eligible:
            out[(pid, attribute)] = min(eligible, key=lambda d: _rank("donor", pid, attribute, d))
    return out


def subset(candidates: Sequence[tuple[int, str]], per_attribute: int = PER_ATTRIBUTE) -> list[tuple[int, str]]:
    """A fixed sample of per_attribute items per attribute from the candidates."""
    by_attribute = defaultdict(list)
    for pid, attribute in candidates:
        by_attribute[attribute].append(pid)
    chosen = []
    for attribute, pids in sorted(by_attribute.items()):
        ranked = sorted(pids, key=lambda pid: _rank("subset", pid, attribute))
        chosen += [(pid, attribute) for pid in ranked[:per_attribute]]
    return sorted(chosen)


def swapped(example: dict, donor: dict) -> dict:
    """The example with the donor's photo; its text, options and answer are its own."""
    letters = {o["label"]: o["letter"] for o in example["options"]}
    return {**example, "example_id": f"{example['example_id']}:swap", "image": donor["image"],
            "donor_product_id": donor["product_id"], "photo_label": donor["reference"],
            "photo_answer": letters[donor["reference"]]}
