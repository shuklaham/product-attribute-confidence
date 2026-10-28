import inspect

import pytest

from pac.redact import redact_description, redact_title
from pac.text import clean_html, mask_brand

CASES = [
    # (description, attribute, expected redacted description)
    ("Navy blue round neck t-shirt, has short sleeves", "neckline",
     "Navy blue t-shirt, has short sleeves"),
    ("Navy blue round neck t-shirt, has short sleeves", "sleeve_length",
     "Navy blue round neck t-shirt"),
    ("Red printed sleeveless dress, has a scoop neck and a drawstring at the waist, curved hem", "neckline",
     "Red printed sleeveless dress, a drawstring at the waist, curved hem"),
    ("Red printed sleeveless dress, has a scoop neck and a drawstring at the waist, curved hem", "sleeve_length",
     "Red printed dress, has a scoop neck and a drawstring at the waist, curved hem"),
    ("Navy blue floral print top, V-neck with full placket, short sleeves", "neckline",
     "Navy blue floral print top, full placket, short sleeves"),
    ("Black top made of cotton with Lycra used for the neck to prevent de-shaping", "neckline",
     "Black top made of cotton"),
    ("Pink top, has three quarter batwing sleeves, panel stitch on chest", "sleeve_length",
     "Pink top, panel stitch on chest"),
    ("Grey melange sweatshirt, has a hood, long sleeves, kangaroo pocket", "neckline",
     "Grey melange sweatshirt, long sleeves, kangaroo pocket"),
    # Rule 3 drops the whole clause, so no remnant like "adjustable" or "strap detail" survives (data review).
    ("Black top, has adjustable straps, curved hem", "sleeve_length", "Black top, curved hem"),
    ("Yellow top, has spaghetti strap detail", "sleeve_length", "Yellow top"),
    ("Grey top with armholes and a round hem", "sleeve_length", "Grey top and a round hem"),
    ("Blue sleevless top with sphagetti straps", "sleeve_length", "Blue top"),
    ("White top, frills at the neckine, curved hem", "neckline", "White top, curved hem"),
    ("", "neckline", ""),
]


@pytest.mark.parametrize("text, key, expected", CASES)
def test_redacted_descriptions(ontology, text, key, expected):
    assert redact_description(text, ontology.attributes[key]) == expected


@pytest.mark.parametrize("text, key, expected", CASES)
def test_no_attribute_evidence_survives(ontology, text, key, expected):
    attribute = ontology.attributes[key]
    redacted = redact_description(text, attribute)
    assert not attribute.attribute_words.search(redacted)
    assert not any(p.search(redacted) for p in attribute.redaction_patterns)


def test_titles_keep_the_product_type(neckline, sleeve):
    assert redact_title("Puma Men Navy Polo T-shirt", neckline, "Puma") == "Puma Men Navy T-shirt"
    assert redact_title("Vero Moda Women Black Sleeveless Top", sleeve, "Vero Moda") == "Vero Moda Women Black Top"
    assert redact_title("Nike Men Red Round Neck T-shirt", neckline, "Nike") == "Nike Men Red T-shirt"


def test_titles_lose_the_words_that_describe_a_mention(neckline, sleeve):
    assert redact_title("Nike Men Town Round Red Neck T-Shirts", neckline, "Nike") == "Nike Men Town T-Shirts"
    assert redact_title("Biba Women Three Fourth Sleeve Kurta", sleeve, "Biba") == "Biba Women Kurta"
    assert redact_title("Nike Men Icon SS T-shirt", sleeve, "Nike") == "Nike Men Icon T-shirt"
    assert redact_title("Puma Men's Sless Black T-shirt", sleeve, "Puma") == "Puma Men's Black T-shirt"
    # The word before a mention goes with it, but never the brand.
    assert redact_title("U.S. Polo Assn. Collar T-shirt", neckline, "U.S. Polo Assn.") == "U.S. Polo Assn. T-shirt"


def test_garbled_runs_are_removed():
    garbled = "Regular fit. The model (height 5'8â€Ã?Â??Ã?Â??Ã?Â??Ã?Â?) wears size M"
    assert clean_html(garbled) == "Regular fit. The model (height 5'8 ) wears size M"
    assert clean_html("Is it soft? Yes.") == "Is it soft? Yes."


def test_brand_is_protected(neckline):
    brand = "U.S. Polo Assn."
    assert redact_title("U.S. Polo Assn. Men Navy T-shirt", neckline, brand) == "U.S. Polo Assn. Men Navy T-shirt"
    text = "U.S. Polo Assn. presents a navy T-shirt, has a polo collar"
    assert redact_description(text, neckline, brand) == "U.S. Polo Assn. presents a navy T-shirt"


def test_brand_only_matches_whole_words(sleeve):
    # Regression: the brand "Lee" used to match inside "sleeves" and shield it from redaction.
    text = "Lee presents a white t-shirt, has short sleeves"
    assert sleeve.named_labels(mask_brand(text, "Lee")) == {"Short sleeves"}
    assert redact_description(text, sleeve, "Lee") == "Lee presents a white t-shirt"


def test_redaction_never_sees_the_label():
    for function in (redact_title, redact_description):
        assert "label" not in inspect.signature(function).parameters


def test_other_attributes_are_untouched(neckline):
    text = "White cotton top, has short sleeves and a curved hem"
    assert redact_description(text, neckline) == text
