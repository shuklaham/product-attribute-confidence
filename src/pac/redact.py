"""Remove attribute evidence from product text for the redacted condition.

Rule 1. The label is never an input, so the output cannot depend on the answer.
Rule 2. The phrases of every label of the attribute are removed, dropped labels included, along
        with the attribute's redaction cues.
Rule 3. A clause that names a label, contains a cue or mentions the attribute is dropped whole,
        so no gap shows what was there. The title and the leading noun phrase cannot be dropped
        without losing the product type, so there each mention is removed together with the
        words that describe it.
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
_TOKEN = re.compile(r"\S+")
_EDGE = re.compile(r"^\W+|\W+$")
MODIFIER_REACH = 3


def _mentions(text: str, attribute: Attribute) -> bool:
    return bool(attribute.attribute_words.search(text)) or any(p.search(text) for p in attribute.redaction_patterns)


def _strip_phrases(text: str, attribute: Attribute) -> str:
    for pattern in attribute.redaction_patterns:
        text = pattern.sub(" ", text)
    return text


def _clean_head(text: str, attribute: Attribute) -> str:
    """Remove label phrases and cues, then each remaining mention with the words that describe it.

    A mention takes with it the words before it back to the earliest describing word within
    three words ("Three Fourth Sleeve", "Round Red Neck"), or else the one word before it. The
    brand placeholder is never removed.
    """
    tokens = _TOKEN.findall(_strip_phrases(text, attribute))
    drop = set()
    for i, token in enumerate(tokens):
        if not attribute.attribute_words.search(token):
            continue
        start = max(i - 1, 0)
        for j in range(max(i - MODIFIER_REACH, 0), i):
            if attribute.head_modifiers.fullmatch(_EDGE.sub("", tokens[j])):
                start = j
                break
        drop.update(k for k in range(start, i + 1) if BRAND_TOKEN not in tokens[k])
    return " ".join(t for k, t in enumerate(tokens) if k not in drop)


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
        if head and position == 0:
            piece = _clean_head(piece, attribute)
        elif _mentions(piece, attribute):
            piece = ""
        if _is_filler(piece):
            dropped_before = True
            continue
        if dropped_before and not kept:
            piece = _LEADING_CONNECTOR.sub("", piece.strip())
        kept.append(piece.strip())
    return " ".join(p for p in kept if p)


def redact_title(title: str, attribute: Attribute, brand: str | None = None) -> str:
    masked, original = _protect(title, brand)
    return _restore(_tidy(_clean_head(masked, attribute)), original)


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
