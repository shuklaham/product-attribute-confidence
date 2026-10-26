from conftest import make_product

from pac.examples import CONDITIONS, make_example
from pac.prompts import messages, prompt_text

PRODUCT = make_product(
    424242, "Tops", neck="Boat Neck", brand="Vero Moda", title="Vero Moda Women Black Boat Neck Top",
    description="Black boat neck top made of cotton, has three quarter sleeves",
)


def _example(attribute, condition, label="Boat neck"):
    return make_example(PRODUCT, attribute, label, "dev", 424242, condition)


def test_options_cover_every_kept_label_once(neckline):
    example = _example(neckline, "full")
    assert sorted(o["label"] for o in example["options"]) == sorted(neckline.label_names)
    assert [o["letter"] for o in example["options"]] == list("ABCDEFGHI")


def test_answer_letter_points_to_the_reference(neckline):
    example = _example(neckline, "redacted")
    by_letter = {o["letter"]: o["label"] for o in example["options"]}
    assert by_letter[example["answer"]] == example["reference"] == "Boat neck"


def test_both_conditions_share_the_option_order(neckline):
    orders = [[o["label"] for o in _example(neckline, c)["options"]] for c in CONDITIONS]
    assert orders[0] == orders[1]


def test_redacted_evidence_names_no_label(neckline):
    example = _example(neckline, "redacted")
    evidence = f"{example['title']} {example['description']}"
    assert example["title"] == "Vero Moda Women Black Top"
    assert not neckline.attribute_words.search(evidence)
    assert not any(p.search(evidence) for p in neckline.all_patterns)


def test_prompt_never_contains_identifiers_or_the_reference_outside_the_options(neckline):
    text = prompt_text(_example(neckline, "redacted"))
    evidence_part = text.split("Question:")[0]
    assert "424242" not in text and "http" not in text
    assert "boat" not in evidence_part.lower()


def test_messages_put_the_photo_first(neckline):
    content = messages(_example(neckline, "full"), image="424242.jpg")[0]["content"]
    assert content[0] == {"type": "image", "image": "424242.jpg"}
    assert content[1]["type"] == "text"
