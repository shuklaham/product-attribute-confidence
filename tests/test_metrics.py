import math

import numpy as np
import pytest

from pac import metrics as m


def test_accuracy_counts_invalid_outputs_as_wrong():
    assert m.accuracy([1, 0, 0, 1]) == 0.5


def test_macro_accuracy_weights_groups_equally():
    assert m.macro_accuracy([1, 1, 1, 0], ["a", "a", "a", "b"]) == pytest.approx(0.5)


def test_brier_with_and_without_the_invalid_penalty():
    q, c = [0.9, 0.2, None], [1, 0, 0]
    assert m.brier(q, c) == pytest.approx((0.01 + 0.04) / 2)
    assert m.brier(q, c, invalid_loss=1.0) == pytest.approx((0.01 + 0.04 + 1.0) / 3)


def test_ece_is_zero_when_confidence_matches_accuracy():
    q = [0.3] * 10 + [0.8] * 10
    c = [1] * 3 + [0] * 7 + [1] * 8 + [0] * 2
    assert m.ece(q, c) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    assert m.ece([0.9] * 10, [1] * 6 + [0] * 4) == pytest.approx(0.3)


def test_confidence_one_falls_in_the_last_bin():
    table = m.reliability([1.0, 0.0], [1, 0], bins=10)
    assert table[-1].count == 1 and table[0].count == 1


def test_auroc_extremes_ties_and_brute_force():
    assert m.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert m.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert m.auroc([0.5] * 4, [1, 0, 1, 0]) == 0.5
    assert math.isnan(m.auroc([0.9, 0.8], [1, 1]))
    rng = np.random.default_rng(0)
    q = rng.choice([0.1, 0.5, 0.9, 0.95], size=200).tolist()
    c = rng.integers(0, 2, size=200).tolist()
    right = [x for x, y in zip(q, c) if y]
    wrong = [x for x, y in zip(q, c) if not y]
    brute = sum((r > w) + 0.5 * (r == w) for r in right for w in wrong) / (len(right) * len(wrong))
    assert m.auroc(q, c) == pytest.approx(brute)


def test_risk_coverage_groups_ties_and_keeps_invalid_outputs_in_coverage():
    first, second = m.risk_coverage([0.9, 0.9, 0.5, None], [1, 0, 1, 0])
    assert (first.threshold, first.accepted, first.coverage, first.risk) == (0.9, 2, 0.5, 0.5)
    assert (second.threshold, second.accepted, second.coverage) == (0.5, 3, 0.75)
    assert second.risk == pytest.approx(1 / 3)


def test_aurc():
    # Accepting 0.9 first gives risk 1; adding 0.5 gives risk 1/2. Area = (1 + 1/2) / 2.
    assert m.aurc([0.9, 0.5], [0, 1]) == pytest.approx(0.75)


def test_aurc_averages_over_the_order_of_tied_items():
    # Right then wrong gives risks 0 and 1/2; wrong then right gives 1 and 1/2. Mean area = 1/2.
    assert m.aurc([0.9, 0.9], [1, 0]) == pytest.approx(0.5)
    rng = np.random.default_rng(2)
    q = rng.choice([0.6, 0.8, 0.9], size=12)
    c = rng.integers(0, 2, size=12)
    areas = []
    for _ in range(4000):
        order = np.lexsort((rng.random(12), -q))  # by confidence, ties in random order
        errors = np.cumsum(1 - c[order])
        areas.append(np.mean(errors / np.arange(1, 13)))
    assert m.aurc(q, c) == pytest.approx(np.mean(areas), abs=0.005)


def test_aurc_ranks_invalid_outputs_last_as_errors():
    # The invalid output is wrong and ranked after both valid ones: risks 0, 1/2, 2/3.
    assert m.aurc([0.9, 0.5, None], [1, 0, 0]) == pytest.approx((0 + 1 / 2 + 2 / 3) / 3)
    # An invalid output scores exactly like a wrong answer with the lowest confidence, so it beats
    # a confidently wrong answer; that is why the invalid rate is reported beside the area.
    assert m.aurc([None, 0.5], [0, 1]) == pytest.approx(m.aurc([0.0, 0.5], [0, 1])) == pytest.approx(0.25)
    assert m.aurc([0.9, 0.5], [0, 1]) == pytest.approx(0.75)


def test_risk_curve_is_flat_for_one_stated_value_and_averages_to_the_area():
    coverage, risk = m.risk_curve([0.95] * 10, [1] * 8 + [0] * 2)
    assert coverage[-1] == 1.0 and np.allclose(risk, 0.2)
    q, c = [0.9, 0.8, 0.8, None], [1, 0, 1, 0]
    assert np.mean(m.risk_curve(q, c)[1]) == pytest.approx(m.aurc(q, c))


def test_aurc_counts_a_right_answer_without_confidence_as_right():
    # Letter probabilities can be missing for a valid, correct answer; it ranks last but is not an error.
    assert m.aurc([0.9, None], [1, 1]) == 0.0


def test_threshold_gives_the_most_coverage_within_the_risk_target():
    q = [0.99] * 19 + [0.9] + [0.8] * 10
    c = [1] * 19 + [0] + [1] * 5 + [0] * 5
    # Accepting >= 0.9 takes 20 items with 1 error (5%); accepting >= 0.8 takes 30 with 6 errors (20%).
    assert m.select_threshold(q, c, max_risk=0.05) == 0.9
    assert m.select_threshold([0.9, 0.8], [0, 0], max_risk=0.05) == math.inf


def test_frozen_threshold_keeps_invalid_outputs_in_the_denominator():
    assert m.at_threshold([0.95, 0.6, None, 0.97], [1, 1, 0, 0], threshold=0.9) == (0.5, 0.5, 2)


def test_confidence_and_accuracy_drop_use_the_same_items():
    drop = m.confidence_and_accuracy_drop([0.9, 0.9], [1, 1], [0.8, 0.6], [1, 0])
    assert drop == pytest.approx((0.2, 0.5))
    # The third item is invalid under redaction, so it leaves both drops.
    drop = m.confidence_and_accuracy_drop([0.9, 0.9, 0.9], [1, 1, 1], [0.8, 0.6, None], [1, 0, 0])
    assert drop == pytest.approx((0.2, 0.5))
    with pytest.raises(ValueError):
        m.confidence_and_accuracy_drop([0.9], [1], [0.8, 0.6], [1, 0])


def test_family_bootstrap_draws_whole_families_and_is_reproducible():
    families = [7, 7, 8, 9, 9, 9]

    def families_stay_whole(rows):
        return float(np.sum(rows == 0) == np.sum(rows == 1) and np.sum(rows == 3) == np.sum(rows == 5))

    assert m.family_bootstrap(families_stay_whole, families, resamples=500) == (1.0, 1.0, 1.0)
    values = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    def mean(rows):
        return float(values[rows].mean())

    first = m.family_bootstrap(mean, families, resamples=500, seed=3)
    assert first == m.family_bootstrap(mean, families, resamples=500, seed=3)
    assert first[1] <= first[0] <= first[2]


def test_family_bootstrap_refuses_undefined_statistics():
    q, c = np.array([0.9, 0.8, 0.7]), np.array([1, 1, 0])
    with pytest.raises(ValueError, match="undefined"):
        m.family_bootstrap(lambda rows: m.auroc(q[rows], c[rows]), [1, 2, 3], resamples=200)


def test_paired_difference_of_identical_methods_is_exactly_zero():
    rng = np.random.default_rng(1)
    q = rng.random(100)
    c = (rng.random(100) < q).astype(float)
    families = rng.integers(0, 30, 100)

    def difference(rows):
        return m.brier(q[rows], c[rows]) - m.brier(q[rows], c[rows])

    assert m.family_bootstrap(difference, families, resamples=200) == (0.0, 0.0, 0.0)
