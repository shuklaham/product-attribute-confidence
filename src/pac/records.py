"""Load Myntra style JSON records into Product objects."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

from .text import clean_html, clean_text


@dataclass(frozen=True, eq=False)
class Product:
    id: int
    category: str | None
    brand: str
    title: str
    variant: str
    description: str
    raw_labels: dict = field(default_factory=dict)
    siblings: frozenset[int] = frozenset()
    views: tuple[str, ...] = ()


def sibling_ids(data: dict) -> set[int]:
    """Style IDs of the product's colour variants, from `colours.colors.<id>.styleid`."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("styleid", "styleId") and str(value).isdigit():
                    found.add(int(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data.get("colours"))
    return found


def parse(record: dict, fallback_id: int) -> Product:
    data = record.get("data") or {}
    descriptors = data.get("productDescriptors") or {}
    return Product(
        id=int(data.get("id") or fallback_id),
        category=(data.get("articleType") or {}).get("typeName"),
        brand=(data.get("brandName") or "").strip(),
        title=clean_text(data.get("productDisplayName")),
        variant=(data.get("variantName") or "").strip(),
        description=clean_html((descriptors.get("description") or {}).get("value")),
        raw_labels=dict(data.get("articleAttributes") or {}),
        siblings=frozenset(sibling_ids(data)),
        views=tuple(sorted((data.get("styleImages") or {}).keys())),
    )


def style_files(styles_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(styles_dir, "*.json")))


def load_products(styles_dir: str) -> tuple[list[Product], int]:
    """Parse every style JSON in a directory. Returns the products and the unreadable count."""
    products, failures = [], 0
    for path in style_files(styles_dir):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except (OSError, ValueError):
            failures += 1
            continue
        products.append(parse(record, int(os.path.basename(path).split(".")[0])))
    return products, failures


def fingerprint(files: Iterable[tuple[str, bytes]]) -> str:
    """Order-independent hash over (file name, content) pairs, to compare data sources."""
    digest = hashlib.sha256()
    for name, content in sorted((os.path.basename(n), hashlib.sha256(c).hexdigest()) for n, c in files):
        digest.update(f"{name}\0{content}\n".encode())
    return digest.hexdigest()


def directory_fingerprint(styles_dir: str) -> str:
    def read():
        for path in style_files(styles_dir):
            with open(path, "rb") as f:
                yield path, f.read()

    return fingerprint(read())
