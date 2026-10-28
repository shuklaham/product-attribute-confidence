from collections import Counter

import pytest

from pac.calibsft import balance, examples_from_pool, target_confidence


def test_targets_mix_the_success_rate_with_the_answers_correctness():
    assert target_confidence(0.625, 1) == "0.81"
    assert target_confidence(0.625, 0) == "0.31"
    assert target_confidence(1.0, 1) == "1.00" and target_confidence(0.0, 0) == "0.00"


def _prompt(pid):
    return {"example_id": f"{pid}:neckline:full", "product_id": pid, "attribute": "neckline", "condition": "full",
            "answer": "B"}


def test_examples_keep_right_and_wrong_answers_and_skip_invalid_ones():
    record = {"example_id": "1:neckline:full", "product_id": 1, "attribute": "neckline", "condition": "full",
              "answer": "B", "choices": ["B", "A", None, "B"], "correct": [1, 0, 0, 1],
              "errors": [None, None, "malformed_json", None]}
    out = examples_from_pool([record], {(1, "neckline"): _prompt(1)})
    # The invalid answer is not trained on, but counts as wrong in the success rate: q = 2/4.
    assert [(e["target_choice"], e["target_confidence"], e["target_correct"]) for e in out] == [
        ("B", "0.75", 1), ("A", "0.25", 0), ("B", "0.75", 1)]
    assert len({e["example_id"] for e in out}) == 3
    with pytest.raises(ValueError):
        examples_from_pool([record | {"answer": "C"}], {(1, "neckline"): _prompt(1)})


def test_balance_caps_each_target_value_and_keeps_rare_ones():
    examples = ([{"example_id": f"a{i}", "target_confidence": "1.00"} for i in range(50)]
                + [{"example_id": f"b{i}", "target_confidence": "0.25"} for i in range(3)])
    kept = balance(examples, cap=10)
    assert Counter(e["target_confidence"] for e in kept) == {"1.00": 10, "0.25": 3}
    assert kept == balance(list(reversed(examples)), cap=10)
