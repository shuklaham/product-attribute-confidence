import math

import numpy as np
import pytest

from pac.methods import confidence, fit, letter_rows, thresholds


def record(answer, choice, stated, logprobs, attribute="neckline", condition="redacted", error=None):
    return {"attribute": attribute, "condition": condition, "answer": answer, "choice": choice,
            "confidence": stated, "error": error, "correct": int(error is None and choice == answer),
            "letter_logprobs": logprobs}


def overconfident_records(n=400, seed=0):
    """Letters reported twice as sharp as the truth, and a stated confidence that never changes."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.normal(size=4) * 1.5
        p = np.exp(z) / np.exp(z).sum()
        answer = "ABCD"[rng.choice(4, p=p)]
        reported = 2 * z - np.log(np.exp(2 * z).sum())
        choice = "ABCD"[int(np.argmax(reported))]
        out.append(record(answer, choice, 0.95, dict(zip("ABCD", reported))))
    return out


def test_letter_rows_follow_the_offered_order():
    rows, answers, chosen = letter_rows([record("B", "C", 0.9, {"A": -3.0, "B": -0.5, "C": -1.0})])
    assert rows.tolist() == [[-3.0, -0.5, -1.0]] and answers.tolist() == [1] and chosen.tolist() == [2]


def test_fit_recovers_the_temperature_and_maps_a_constant_to_the_hit_rate():
    records = overconfident_records()
    fitted = fit(records)
    assert fitted.temperature[("neckline", "redacted")].t == pytest.approx(2.0, abs=0.35)
    hit_rate = np.mean([r["correct"] for r in records])
    assert confidence(records[:1], "platt", fitted)[0] == pytest.approx(hit_rate, abs=0.02)


def test_methods_keep_invalid_outputs_invalid():
    records = overconfident_records(50)
    bad = record("A", None, None, {"A": -1.0, "B": -1.0, "C": -2.0, "D": -3.0}, error="malformed_json")
    fitted = fit(records)
    for method in ("stated", "platt", "letter"):
        assert confidence([bad], method, fitted) == [None]


def test_letter_confidence_is_the_chosen_letters_probability():
    fitted = fit(overconfident_records())
    t = fitted.temperature[("neckline", "redacted")].t
    item = record("A", "A", 0.95, {"A": math.log(0.6), "B": math.log(0.3), "C": math.log(0.1), "D": math.log(1e-9)})
    p = np.array([0.6, 0.3, 0.1, 1e-9]) ** (1 / t)
    assert confidence([item], "letter", fitted)[0] == pytest.approx(p[0] / p.sum())


def test_thresholds_are_per_attribute_and_condition():
    records = overconfident_records(300) + [dict(r, condition="full") for r in overconfident_records(300, seed=1)]
    chosen = thresholds(records, "letter", fit(records))
    assert set(chosen) == {("neckline", "redacted"), ("neckline", "full")}
    assert all(0 < t < 1 or t == math.inf for t in chosen.values())
