from conftest import make_product

from pac.examples import make_example
from pac.swap import donors, subset, swapped


def _examples(neckline, rows):
    """rows: (product id, family id, category, label, clean); both text conditions of each."""
    out = []
    for pid, family, category, label, clean in rows:
        product = make_product(pid, category=category)
        for condition in ("full", "redacted"):
            out.append(make_example(product, neckline, label, "heldout_test", family, condition,
                                    text_status="none", clean=clean))
    return out


def test_donor_is_same_category_other_family_and_another_label(neckline):
    examples = _examples(neckline, [
        (1, 1, "Kurtas", "Round neck", True),
        (2, 1, "Kurtas", "V-neck", True),       # same family as 1: never its donor
        (3, 3, "Kurtas", "V-neck", True),
        (4, 4, "Kurtas", "Round neck", True),   # same label as 1: never its donor
        (5, 5, "Dresses", "V-neck", True),      # other category: never a donor for Kurtas
        (6, 6, "Kurtas", "Boat neck", False),   # not clean: neither item nor donor
    ])
    chosen = donors(examples)
    assert chosen[(1, "neckline")] == 3 and chosen[(4, "neckline")] == 3
    assert chosen[(3, "neckline")] in (1, 4)
    assert (5, "neckline") not in chosen and (6, "neckline") not in chosen
    assert chosen == donors(list(reversed(examples)))


def test_subset_is_fixed_and_capped_per_attribute():
    candidates = [(pid, "neckline") for pid in range(50)] + [(pid, "sleeve_length") for pid in range(30)]
    picked = subset(candidates, per_attribute=10)
    assert len([k for k in picked if k[1] == "neckline"]) == 10
    assert len([k for k in picked if k[1] == "sleeve_length"]) == 10
    assert picked == subset(list(reversed(candidates)), per_attribute=10)


def test_swapped_keeps_the_text_and_labels_the_photo_in_its_own_letters(neckline):
    item, donor = _examples(neckline, [(1, 1, "Kurtas", "Round neck", True), (3, 3, "Kurtas", "V-neck", True)])[::2]
    swap = swapped(item, donor)
    letters = {o["label"]: o["letter"] for o in item["options"]}
    assert swap["image"] == donor["image"] and swap["photo_answer"] == letters["V-neck"]
    assert (swap["answer"], swap["title"], swap["description"], swap["options"]) == (
        item["answer"], item["title"], item["description"], item["options"])
    assert swap["example_id"] == item["example_id"] + ":swap"
