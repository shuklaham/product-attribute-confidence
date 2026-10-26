"""Remove attribute evidence from product text for the redacted condition.

Rule 1. The label is never an input, so the output cannot depend on the answer.
Rule 2. The phrases of every label of the attribute are removed, dropped labels included.
Rule 3. A clause that still mentions the attribute after phrase removal is dropped whole, so
        no gap shows what was there. The title and the leading noun phrase only lose the
        attribute words, because dropping them would also drop the product type.
Rule 4. The brand name is protected.
"""

from __future__ import annotations

import re

from .ontology import Attribute
from .text import BRAND_TOKEN, brand_pattern

_PUNCTUATION = re.compile(r"([,;.!?]+)")
_CLAUSE_START = re.compile(r"\s+(?=(?:and|with|has|having|featuring)\b)", re.IGNORECASE)
_LEADING_CONNECTOR = re.compile(r"^(?:and|with)\s+", re.IGNORECASE)
_FILLER = frozenset("a an the and with has having featuring it its is are comes of in on at to".split())
_WORD = re.compile(r"[\w%/'-]+")


def _strip_phrases(text: str, attribute: Attribute) -> str:
    for pattern in attribute.all_patterns:
        text = pattern.sub(" ", text)
    return text


def _clean_head(text: str, attribute: Attribute) -> str:
    return attribute.head_fallback.sub(" ", _strip_phrases(text, attribute))


def _is_filler(text: str) -> bool:
    return all(word in _FILLER for word in _WORD.findall(text.lower()))


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;.!?])", r"\1", text)
    text = re.sub(r"[,;]\s*(?=[,;.!?])", "", text)
    text = re.sub(r"^[\s,;.!?]+|[\s,;]+$", "", text)
    return text.strip()


def _protect(text: str, brand: str | None) -> tuple[str, str | None]:
    pattern = brand_pattern(brand)
    if not pattern:
        return text, None
    match = pattern.search(text)
    return pattern.sub(BRAND_TOKEN, text), match.group(0) if match else None


def _restore(text: str, original: str | None) -> str:
    return text.replace(BRAND_TOKEN, original) if original else text


def _redact_segment(body: str, attribute: Attribute, head: bool) -> str:
    kept, dropped_before = [], False
    for position, piece in enumerate(_CLAUSE_START.split(body.strip())):
        piece = _clean_head(piece, attribute) if head and position == 0 else _strip_phrases(piece, attribute)
        if attribute.attribute_words.search(piece) or _is_filler(piece):
            dropped_before = True
            continue
        if dropped_before and not kept:
            piece = _LEADING_CONNECTOR.sub("", piece.strip())
        kept.append(piece.strip())
    return " ".join(p for p in kept if p)


def redact_title(title: str, attribute: Attribute, brand: str | None = None) -> str:
    masked, original = _protect(title, brand)
    cleaned = _clean_head(masked, attribute)
    if attribute.attribute_words.search(cleaned):
        cleaned = attribute.attribute_words.sub(" ", cleaned)
    return _restore(_tidy(cleaned), original)


def redact_description(description: str, attribute: Attribute, brand: str | None = None) -> str:
    masked, original = _protect(description, brand)
    parts = _PUNCTUATION.split(masked)
    out = []
    for index in range(0, len(parts), 2):
        body = parts[index]
        punctuation = parts[index + 1] if index + 1 < len(parts) else ""
        kept = _redact_segment(body, attribute, head=index == 0)
        if kept:
            out.append(body[: len(body) - len(body.lstrip())] + kept + punctuation)
    return _restore(_tidy("".join(out)), original)
