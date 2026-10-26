"""Colour-variant families, frozen split assignment and per-attribute label decisions."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

from .ontology import DROPPED, UNLABELED, UNMAPPED, Ontology
from .records import Product
from .text import mask_brand

SEEN = ("Tshirts", "Tops", "Shirts")
HELD_OUT = ("Kurtas", "Kurtis", "Tunics", "Dresses", "Nightdress", "Night suits", "Sweatshirts", "Sweaters")
# Seen-category families are hashed into these buckets (percent of families).
BUCKETS = (("train", 70), ("dev", 10), ("cal_fit", 5), ("cal_threshold", 5), ("test", 10))
HELD_OUT_SPLIT = "heldout_test"
CROSSING_SPLIT = "excluded_crossing"
EVAL_SPLITS = ("dev", "cal_fit", "cal_threshold", "test", HELD_OUT_SPLIT)
SALT = "splits-v0"

# A usable photo needs at least this many pixels on its long side.
MIN_PHOTO_LONG_SIDE = 480

# Exclusion reasons, in order of precedence. An attribute's own text-based exclusion, such as
# roll_up_sleeves for sleeve length, comes from configs/ontology.yaml and ranks after UNMAPPED_VALUE.
CROSSING_FAMILY = "crossing_family"
NO_USABLE_PHOTO = "no_usable_photo"
DROPPED_LABEL = "dropped_label"
UNMAPPED_VALUE = "unmapped_value"
FAMILY_DISAGREEMENT = "family_disagreement"
TEXT_CONFLICT = "text_conflict"
UNSUPPORTED_LABEL = "unsupported_label"


def families(products: list[Product]) -> dict[int, int]:
    """Map product ID to family ID: the smallest product ID among linked colour variants."""
    parent = {p.id: p.id for p in products}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in products:
        for s in p.siblings:
            if s in parent:
                a, b = find(p.id), find(s)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return {pid: find(pid) for pid in parent}


def bucket(family_id: int, salt: str = SALT) -> str:
    """Seen-category split for a family. Depends only on the family ID and the salt."""
    position = int(hashlib.sha256(f"{salt}:{family_id}".encode()).hexdigest(), 16) % 100
    cumulative = 0
    for name, share in BUCKETS:
        cumulative += share
        if position < cumulative:
            return name
    raise AssertionError("bucket shares must sum to 100")


@dataclass(frozen=True)
class SplitRow:
    product_id: int
    family_id: int
    category: str
    split: str


def assign_splits(products: list[Product], family_of: dict[int, int], salt: str = SALT) -> dict[int, SplitRow]:
    in_scope = [p for p in products if p.category in SEEN + HELD_OUT]
    sides = defaultdict(set)
    for p in in_scope:
        sides[family_of[p.id]].add(p.category in SEEN)
    rows = {}
    for p in in_scope:
        family = family_of[p.id]
        if len(sides[family]) > 1:
            split = CROSSING_SPLIT
        elif p.category in HELD_OUT:
            split = HELD_OUT_SPLIT
        else:
            split = bucket(family, salt)
        rows[p.id] = SplitRow(p.id, family, p.category, split)
    return rows


@dataclass(frozen=True)
class LabelRow:
    product_id: int
    attribute: str
    raw_value: str
    label: str        # "" when the raw value does not map to a kept label
    text_status: str  # agrees, conflicts or silent; "" without a label
    included: bool
    reason: str       # "" when included


def matching_text(product: Product) -> str:
    """Listing text used to detect named labels. The brand is masked."""
    return mask_brand(" ".join((product.title, product.variant, product.description)), product.brand)


def text_status(attribute, label: str | None, text: str) -> str:
    if not label:
        return ""
    named = attribute.named_labels(text)
    if label in named:
        return "agrees"
    return "conflicts" if named else "silent"


def usable_photos(photo_rows: list[dict]) -> set[int]:
    """Product IDs whose photo decoded and is large enough, from photos.csv rows."""
    return {int(r["product_id"]) for r in photo_rows
            if r["status"] == "ok" and max(int(r["width"]), int(r["height"])) >= MIN_PHOTO_LONG_SIDE}


def label_rows(products: list[Product], splits: dict[int, SplitRow], ontology: Ontology,
               photo_ok: set[int] | None = None) -> list[LabelRow]:
    """Label decisions per (product, attribute). photo_ok=None skips the photo rule."""
    by_id = {p.id: p for p in products}
    drafts = []
    family_labels = defaultdict(set)
    for pid, split_row in splits.items():
        product = by_id[pid]
        text = matching_text(product)
        for key, attribute in ontology.attributes.items():
            raw = product.raw_labels.get(attribute.source_field)
            label, outcome = attribute.canonical(raw)
            if outcome == UNLABELED:
                continue
            ambiguous = attribute.excluded_by_text(text)
            drafts.append((pid, key, str(raw).strip(), label or "", outcome, text_status(attribute, label, text),
                           ambiguous))
            if label:
                family_labels[(split_row.family_id, key)].add(label)

    def reason(pid, key, outcome, status, ambiguous):
        split_row = splits[pid]
        if split_row.split == CROSSING_SPLIT:
            return CROSSING_FAMILY
        if photo_ok is not None and pid not in photo_ok:
            return NO_USABLE_PHOTO
        if outcome == DROPPED:
            return DROPPED_LABEL
        if outcome == UNMAPPED:
            return UNMAPPED_VALUE
        if ambiguous:
            return ontology.attributes[key].exclusion_reason
        if len(family_labels[(split_row.family_id, key)]) > 1:
            return FAMILY_DISAGREEMENT
        if status == "conflicts":
            return TEXT_CONFLICT
        return ""

    decided = [(pid, key, raw, label, status, reason(pid, key, outcome, status, ambiguous))
               for pid, key, raw, label, outcome, status, ambiguous in drafts]
    support = Counter((key, label) for pid, key, raw, label, status, why in decided
                      if not why and splits[pid].split == "train")
    rows = []
    for pid, key, raw, label, status, why in decided:
        if not why and splits[pid].split in EVAL_SPLITS and support[(key, label)] < ontology.min_train_support:
            why = UNSUPPORTED_LABEL
        rows.append(LabelRow(pid, key, raw, label, status, not why, why))
    return rows
