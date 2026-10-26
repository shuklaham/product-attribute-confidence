import math

import pytest

from pac.evaluation import category_guess, evidence_sensitivity, letter_confidence, summarize
from pac.examples import LETTERS
from pac.inference import choice_token_index, evidence_view, letter_logprobs, letter_token_ids, score

# Generated tokens decoded one at a time, as the Qwen3-VL-4B-Instruct tokenizer
# (revision ebb281ec) splits each layout.
SPACED = ['{"', 'choice', '":', ' "', 'A', '",', ' "', 'confidence', '":', ' ', '0', '.', '8', '5', '}']
COMPACT = ['{"', 'choice', '":"', 'A', '","', 'confidence', '":', '0', '.', '8', '5', '}']
FENCED = ['```', 'json', '\n', '{"', 'choice', '":', ' "', 'C', '",', ' "', 'confidence', '":', ' ', '0', '.', '9',
          '}\n', '```']
MULTILINE = ['{\n', ' ', ' "', 'choice', '":', ' "', 'B', '",\n', ' ', ' "', 'confidence', '":', ' ', '1', '.', '0',
             '\n', '}']


@pytest.mark.parametrize("pieces, index", [(SPACED, 4), (COMPACT, 3), (FENCED, 7), (MULTILINE, 6)])
def test_choice_position_in_every_layout(pieces, index):
    assert choice_token_index(pieces) == index
    assert pieces[index] in LETTERS


@pytest.mark.parametrize("pieces", [
    ['{"', 'choice', '":', ' ', '"A', '",', ' "', 'confidence', '":', ' ', '1', '}'],  # quote and letter merged
    ['The', ' neckline', ' is', ' round', '.'],
    ['{"', 'choice', '":', ' "'],  # cut off before the letter
])
def test_no_choice_position(pieces):
    assert choice_token_index(pieces) is None


def test_letter_token_ids_must_be_single_tokens():
    assert letter_token_ids(lambda s: [ord(s)])["C"] == ord("C")
    with pytest.raises(ValueError):
        letter_token_ids(lambda s: [1, ord(s)])


def test_letter_logprobs_reads_the_choice_position():
    ids = {letter: 100 + i for i, letter in enumerate(LETTERS)}
    logprobs = [{0: -0.1}] * len(SPACED)
    logprobs[4] = {100: -0.2, 101: -1.9, 102: -3.0, 103: -4.0, 999: -5.0}
    assert letter_logprobs(SPACED, logprobs, ids, "ABCD") == {"A": -0.2, "B": -1.9, "C": -3.0, "D": -4.0}
    assert letter_logprobs(['Round'], [{}], ids, "ABCD") is None


def example(answer="B", category="Tshirts", family=1):
    return {"example_id": f"{family}:neckline:redacted", "product_id": family, "family_id": family,
            "category": category, "split": "dev", "attribute": "neckline", "condition": "redacted",
            "answer": answer, "reference": {"A": "V-neck", "B": "Round neck"}[answer], "text_status": "silent",
            "clean": True,
            "options": [{"letter": "A", "label": "V-neck"}, {"letter": "B", "label": "Round neck"}]}


def test_score_uses_the_strict_parser():
    right = score(example(), '{"choice": "B", "confidence": 0.7}', {"A": -2.0, "B": -0.2})
    assert (right["correct"], right["choice"], right["confidence"], right["error"]) == (1, "B", 0.7, None)
    assert right["prediction"] == "Round neck"
    fenced = score(example(), '```json\n{"choice": "B", "confidence": 0.7}\n```', None)
    assert (fenced["correct"], fenced["error"]) == (0, "malformed_json")


def test_evidence_modes_remove_only_evidence():
    e = {**example(), "title": "Acme tee", "description": "Soft cotton"}
    assert evidence_view(e, "photo_text") == (e, True)
    assert evidence_view(e, "text_only") == (e, False)
    view, photo = evidence_view(e, "photo_only")
    assert photo and view["title"] == view["description"] == "" and view["options"] == e["options"]


def test_letter_confidence_is_renormalized_over_offered_letters():
    record = {"choice": "B", "letter_logprobs": {"A": math.log(0.1), "B": math.log(0.3)}}
    assert letter_confidence(record) == pytest.approx(0.75)
    assert letter_confidence(record, temperature=1e6) == pytest.approx(0.5, abs=1e-3)
    assert letter_confidence({"choice": None, "letter_logprobs": None}) is None


def test_summarize_and_category_guess():
    train = [{**example(answer="B", family=f), "condition": "full"} for f in range(3)]
    records = [
        score(example("B", family=1), '{"choice": "B", "confidence": 0.9}', {"A": -3.0, "B": -0.05}),
        score(example("A", family=2), '{"choice": "B", "confidence": 0.9}', {"A": -1.0, "B": -0.4}),
        score(example("B", family=3), 'not json', None),
        score(example("B", category="Tops", family=4), '{"choice": "B", "confidence": 0.6}', {"A": -1.0, "B": -0.5}),
    ]
    s = summarize(records, resamples=200, guess=category_guess(train))
    assert s["n"] == 4 and s["accuracy"] == pytest.approx(0.5)
    assert s["macro_accuracy"] == pytest.approx((1 / 3 + 1) / 2)
    assert s["category_guess"] == pytest.approx(0.75)  # always "Round neck"; Tops falls back to the attribute
    assert s["invalid"] == pytest.approx(0.25) and s["errors"] == {"malformed_json": 1}
    assert (s["distinct_confidences"], s["top_confidence"], s["top_confidence_share"]) == (2, 0.9, pytest.approx(2 / 3))
    assert s["mean_confidence"] == pytest.approx(0.8)
    assert s["letter_missing"] == 0
    assert s["accuracy_valid"] == pytest.approx(2 / 3)


def test_evidence_sensitivity_joins_items_and_reports_invalid_rates():
    def record(family, condition, text):
        return {**score(example(family=family), text, None), "condition": condition}

    full = [record(1, "full", '{"choice": "B", "confidence": 0.9}'),
            record(2, "full", '{"choice": "B", "confidence": 0.9}')]
    redacted = [record(2, "redacted", 'no answer'), record(1, "redacted", '{"choice": "B", "confidence": 0.6}')]
    s = evidence_sensitivity(full, redacted)
    assert (s["n"], s["valid_in_both"]) == (2, 1)
    assert (s["confidence_drop"], s["accuracy_drop"]) == pytest.approx((0.3, 0.0))
    assert (s["invalid_full"], s["invalid_redacted"]) == (0.0, 0.5)
    with pytest.raises(ValueError):
        evidence_sensitivity(full, redacted[:1])
