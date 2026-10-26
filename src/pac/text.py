"""Text helpers shared by label matching and redaction."""

import html
import re

_EDITOR_PASTE = re.compile(r"<div[^>]*_mcePaste[^>]*>.*?</div>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")

# Stands in for the brand name during redaction. A private-use character is never a word
# character, so no attribute pattern can match it.
BRAND_TOKEN = ""


def clean_html(raw: str | None) -> str:
    """Strip markup, including hidden editor paste blocks, and normalise whitespace."""
    text = _EDITOR_PASTE.sub(" ", raw or "")
    text = _TAG.sub(" ", text)
    return _SPACE.sub(" ", html.unescape(text)).strip()


def brand_pattern(brand: str | None) -> re.Pattern | None:
    """Whole-word match for the brand, so "Lee" never matches inside "sleeves"."""
    return re.compile(rf"(?<!\w){re.escape(brand)}(?!\w)", re.IGNORECASE) if brand else None


def mask_brand(text: str, brand: str | None, token: str = " ") -> str:
    """Replace mentions of the brand, so brand words are never read as attribute evidence."""
    pattern = brand_pattern(brand)
    return pattern.sub(token, text) if pattern else text
