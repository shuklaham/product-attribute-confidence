"""Text helpers shared by label matching and redaction."""

import html
import re

_EDITOR_PASTE = re.compile(r"<div[^>]*_mcePaste[^>]*>.*?</div>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
# Runs of mis-decoded characters ("Ã?Â?Ã?Â?…", sometimes after "â€") in some listings: 37 in-scope
# products have them, and one held-out description is 27,000 characters of nothing else.
_GARBLED = re.compile(r"(?:â€)?[ÃÂ?]{3,}")

# Stands in for the brand name during redaction. A private-use character is never a word
# character, so no attribute pattern can match it.
BRAND_TOKEN = ""


def clean_text(raw: str | None) -> str:
    """Remove runs of mis-decoded characters and normalise whitespace."""
    return _SPACE.sub(" ", _GARBLED.sub(" ", raw or "")).strip()


def clean_html(raw: str | None) -> str:
    """Strip markup, including hidden editor paste blocks, then clean the text."""
    text = _EDITOR_PASTE.sub(" ", raw or "")
    text = _TAG.sub(" ", text)
    return clean_text(html.unescape(text))


def brand_pattern(brand: str | None) -> re.Pattern | None:
    """Whole-word match for the brand, so "Lee" never matches inside "sleeves"."""
    return re.compile(rf"(?<!\w){re.escape(brand)}(?!\w)", re.IGNORECASE) if brand else None


def mask_brand(text: str, brand: str | None, token: str = " ") -> str:
    """Replace mentions of the brand, so brand words are never read as attribute evidence."""
    pattern = brand_pattern(brand)
    return pattern.sub(token, text) if pattern else text
