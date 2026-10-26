from pac.ontology import DROPPED, OK, UNLABELED, UNMAPPED
from pac.text import mask_brand


def test_kept_labels_match_the_decision(neckline, sleeve):
    assert neckline.label_names == ["Round neck", "V-neck", "Scoop neck", "Boat neck", "Square neck",
                                    "Mandarin collar", "Polo collar", "Shirt collar", "Cowl neck"]
    assert sleeve.label_names == ["Short sleeves", "Three-quarter sleeves", "Long sleeves", "Sleeveless"]


def test_raw_values_map_to_labels(neckline, sleeve):
    assert neckline.canonical("Round Neck") == ("Round neck", OK)
    assert neckline.canonical(" V-Neck ") == ("V-neck", OK)
    assert sleeve.canonical("No Sleeves") == ("Sleeveless", OK)


def test_strap_styles_and_rare_labels_are_dropped(neckline):
    for raw in ["Shoulder Straps", "Strapless", "Off-Shoulder", "One Shoulder", "Halter Neck",
                "Tie-Up Neck", "High Neck", "Mock Neck", "Keyhole Neck", "Henley Neck", "Hood"]:
        assert neckline.canonical(raw) == (None, DROPPED), raw


def test_missing_and_unknown_values(neckline):
    for raw in [None, "", "NA", "na", " N/A "]:
        assert neckline.canonical(raw) == (None, UNLABELED), raw
    assert neckline.canonical("Sweetheart Neck") == (None, UNMAPPED)


def test_every_label_has_a_definition_and_phrases(ontology):
    for attribute in ontology.attributes.values():
        for label in attribute.labels:
            assert label.definition.endswith("."), label.name
            assert label.patterns, label.name


def test_label_phrases(neckline, sleeve):
    assert neckline.named_labels("Navy round-neck T-shirt") == {"Round neck"}
    assert neckline.named_labels("has a crew neck") == {"Round neck"}
    assert neckline.named_labels("white boat neck top") == {"Boat neck"}
    assert neckline.named_labels("Navy polo T-shirt") == {"Polo collar"}
    assert sleeve.named_labels("has three quarter sleeves") == {"Three-quarter sleeves"}
    assert sleeve.named_labels("black camisole") == {"Sleeveless"}
    assert sleeve.named_labels("came with a camera") == set()


def test_hand_check_regressions(neckline, sleeve):
    # Each case was a phrase-matcher misread found in the 100-item hand check.
    assert neckline.named_labels("Blue jersey with a spread collar in contrast colour") == set()
    assert neckline.named_labels("featuring a deep round neck, this top") == {"Round neck", "Scoop neck"}
    assert sleeve.named_labels("Pair them with torn jeans or three quarter pants") == set()
    assert sleeve.named_labels("snipped strips around the armholes") == set()
    assert sleeve.named_labels("long sleeves with a mid-arm buttoned tab") == {"Long sleeves"}
    assert sleeve.named_labels("has three quarter bat wing sleeves") == {"Three-quarter sleeves"}
    assert sleeve.named_labels("3/4th length sleeves") == {"Three-quarter sleeves"}


def test_roll_up_sleeves_are_excluded_from_sleeve_length(neckline, sleeve):
    for text in ["long sleeves with buttoned cuffs and a mid-arm buttoned tab",
                 "three quarter sleeves with button tabs and a curved hem",
                 "has roll-up sleeves", "foldover sleeves with mid-sleeve buttons"]:
        assert sleeve.excluded_by_text(text), text
    assert not sleeve.excluded_by_text("has short sleeves and a curved hem")
    assert not neckline.excluded_by_text("long sleeves with a mid-arm buttoned tab")
    assert sleeve.exclusion_reason == "roll_up_sleeves"


def test_brand_names_are_not_evidence(neckline):
    text = "U.S. Polo Assn. Men Navy Round Neck T-shirt"
    assert "Polo collar" in neckline.named_labels(text)
    assert neckline.named_labels(mask_brand(text, "U.S. Polo Assn.")) == {"Round neck"}
