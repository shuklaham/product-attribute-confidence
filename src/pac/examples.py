"""Model-facing examples built from the frozen manifests."""

from __future__ import annotations

import hashlib
import random

from .ontology import Attribute
from .records import Product
from .redact import redact_description, redact_title

CONDITIONS = ("full", "redacted")
LETTERS = "ABCDEFGHIJKL"
OPTION_SALT = "options-v0"


def option_order(product_id: int, attribute_key: str, labels: list[str], salt: str = OPTION_SALT) -> list[str]:
    """Seeded option order for one item. Both text conditions get the same order."""
    seed = int(hashlib.sha256(f"{salt}:{product_id}:{attribute_key}".encode()).hexdigest(), 16)
    order = list(labels)
    random.Random(seed).shuffle(order)
    return order


def evidence(product: Product, attribute: Attribute, condition: str) -> tuple[str, str]:
    """Title and description shown to the model under a text condition."""
    if condition == "full":
        return product.title, product.description
    if condition == "redacted":
        return (redact_title(product.title, attribute, product.brand),
                redact_description(product.description, attribute, product.brand))
    raise ValueError(f"unknown condition: {condition}")


def make_example(product: Product, attribute: Attribute, label: str, split: str, family_id: int,
                 condition: str, *, text_status: str, clean: bool) -> dict:
    """One model-facing example. text_status says whether the full text names the label, another label or
    none; clean is False for a test item kept only for the "all labeled items" results."""
    title, description = evidence(product, attribute, condition)
    order = option_order(product.id, attribute.key, attribute.label_names)
    definitions = attribute.definitions
    return {
        "example_id": f"{product.id}:{attribute.key}:{condition}",
        "product_id": product.id,
        "family_id": family_id,
        "category": product.category,
        "split": split,
        "attribute": attribute.key,
        "condition": condition,
        "image": f"{product.id}.jpg",
        "question": attribute.question,
        "title": title,
        "description": description,
        "options": [{"letter": LETTERS[i], "label": name, "definition": definitions[name]}
                    for i, name in enumerate(order)],
        "answer": LETTERS[order.index(label)],
        "reference": label,
        "text_status": text_status,
        "clean": clean,
    }
