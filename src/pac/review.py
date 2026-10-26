"""The hand-check queue, shared by the review app and scripts/review_queue.py."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

QUEUE_SALT = "hand-check-v0"
QUEUE_SIZE = 100
# Test splits stay unseen: checking them could change the exclusion rule after we had looked at their items.
REVIEW_SPLITS = ("train", "dev", "cal_fit", "cal_threshold")
VERDICTS = ("Catalog label is wrong", "Description is wrong", "Phrase matcher misread the text", "Can't tell")
ANNOTATION_FIELDS = ("product_id", "attribute", "catalog_label", "text_names", "verdict", "note", "reviewer",
                     "reviewed_at")


def hand_check_queue(rows: Iterable[dict], n: int = QUEUE_SIZE) -> list[tuple[int, str]]:
    """Seeded sample of (product_id, attribute) whose description names another label, from training-side splits.

    Each row needs product_id, attribute, split and text_status.
    """
    conflicts = [(int(r["product_id"]), r["attribute"]) for r in rows
                 if r["text_status"] == "conflicts" and r["split"] in REVIEW_SPLITS]
    conflicts.sort(key=lambda key: hashlib.sha256(f"{QUEUE_SALT}:{key[0]}:{key[1]}".encode()).hexdigest())
    return conflicts[:n]
