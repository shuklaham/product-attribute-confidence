import re
from collections import Counter

from conftest import make_product

from pac.examples import make_example
from pac.output import parse_output
from pac.sft import completion_labels, condition_for, presentations, reshuffle, target_text


def _examples(neckline, n=40):
    out = []
    for pid in range(1, n + 1):
        product = make_product(pid, description="Navy T-shirt, has a round neck")
        for condition in ("full", "redacted"):
            out.append(make_example(product, neckline, "Round neck", "train", pid // 2, condition,
                                    text_status="agrees", clean=True))
    return out


def test_condition_draws_are_seeded_and_balanced():
    draws = [condition_for(f, "neckline", 0) for f in range(4000)]
    assert draws == [condition_for(f, "neckline", 0) for f in range(4000)]
    assert abs(Counter(draws)["full"] / 4000 - 0.5) < 0.03
    assert [condition_for(7, "neckline", e) for e in range(20)].count("full") not in (0, 20)


def test_one_presentation_per_item_and_epoch_with_the_drawn_condition(neckline):
    examples = _examples(neckline)
    shown = presentations(examples, epochs=2)
    assert len(shown) == 2 * 40
    for e in shown[:40]:
        assert e["condition"] == condition_for(e["family_id"], e["attribute"], 0)
    # Colour variants (same family) share the condition within an epoch.
    by_family = {}
    for e in shown[:40]:
        assert by_family.setdefault(e["family_id"], e["condition"]) == e["condition"]


def test_reshuffled_options_keep_the_answer_right(neckline):
    example = _examples(neckline, 1)[0]
    orders = set()
    for epoch in range(6):
        shuffled = reshuffle(example, epoch)
        by_letter = {o["letter"]: o["label"] for o in shuffled["options"]}
        assert by_letter[shuffled["answer"]] == "Round neck"
        assert sorted(by_letter.values()) == sorted(o["label"] for o in example["options"])
        orders.add(tuple(by_letter.values()))
    assert len(orders) > 1


def test_target_is_valid_json_in_the_base_layout_and_the_span_is_the_value():
    text, (start, end) = target_text("G")
    assert text == '{\n  "choice": "G",\n  "confidence": 0.95\n}'
    assert text[start:end] == "0.95"
    assert parse_output(text, list("ABCDEFGHI")).choice == "G"


def test_only_tokens_overlapping_the_value_are_masked():
    text, span = target_text("B")
    # A character-level stand-in for the tokenizer: one token per character, plus <|im_end|>.
    tokens = [(ord(c), (i, i + 1)) for i, c in enumerate(text)] + [(151645, (len(text), len(text)))]
    labels = completion_labels([t for t, _ in tokens], [o for _, o in tokens], span)
    kept = "".join(chr(t) for t in labels if t not in (-100, 151645))
    assert kept == re.sub(r"0\.95", "", text)
    assert labels[-1] == 151645  # learning to stop
