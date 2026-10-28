import warnings

import numpy as np
import pytest

from pac.calibration import PlattMap, fit_platt, fit_temperature, logit, sigmoid


def test_platt_recovers_known_parameters():
    rng = np.random.default_rng(0)
    q = rng.uniform(0.05, 0.99, 40_000)
    correct = (rng.random(q.size) < sigmoid(0.5 * logit(q) - 0.3)).astype(float)
    fitted = fit_platt(q, correct)
    assert fitted.a == pytest.approx(0.5, abs=0.05)
    assert fitted.b == pytest.approx(-0.3, abs=0.05)


def test_platt_fixes_uniform_overconfidence():
    q = np.full(1000, 0.9)
    correct = np.r_[np.ones(600), np.zeros(400)]
    assert fit_platt(q, correct)(np.array([0.9]))[0] == pytest.approx(0.6, abs=0.01)


def test_platt_never_reverses_the_order_of_confidences():
    q = np.linspace(0.05, 0.95, 50)
    correct = (q < 0.5).astype(float)  # higher confidence is more often wrong
    fitted = fit_platt(q, correct)
    assert fitted.a == 0.0
    assert np.all(np.diff(fitted(q)) >= 0)
    assert fitted(q)[0] == pytest.approx(correct.mean(), abs=0.01)


def test_platt_on_one_stated_value_keeps_the_slope_and_moves_the_level():
    fitted = fit_platt(np.full(500, 0.9), np.r_[np.ones(300), np.zeros(200)])
    assert fitted.a == 1.0
    assert fitted(np.array([0.9]))[0] == pytest.approx(0.6, abs=0.01)
    assert fitted(np.array([0.95]))[0] > fitted(np.array([0.9]))[0]  # order kept for unseen values


def test_platt_refuses_to_fit_without_valid_outputs():
    with pytest.raises(ValueError):
        fit_platt(np.array([np.nan, np.nan]), [0, 0])


def test_stated_one_is_read_as_half_a_step_above_0_99():
    assert logit([1.0])[0] == pytest.approx(logit([0.995])[0])
    assert logit([1.0])[0] > logit([0.99])[0]


def test_platt_keeps_invalid_outputs_invalid():
    assert np.isnan(PlattMap(1.0, 0.0)(np.array([np.nan]))[0])


def test_temperature_undoes_overconfident_logits():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(8_000, 4)) * 1.5
    true = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    answers = np.array([rng.choice(4, p=row) for row in true])
    reported = 2.0 * z  # twice as sharp as the truth
    logprobs = reported - np.log(np.exp(reported).sum(axis=1, keepdims=True))
    assert fit_temperature(logprobs, answers).t == pytest.approx(2.0, abs=0.15)


def test_temperature_refuses_bad_inputs_and_warns_at_the_bound():
    with pytest.raises(ValueError):
        fit_temperature(np.array([[0.0, np.nan], [-1.0, -0.5]]), [0, 1])
    with pytest.raises(ValueError):
        fit_temperature(np.array([[-0.1, -2.3]]), [2])
    # Correct answers always get the lower logit, so the best fit flattens as far as allowed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_temperature(np.array([[-0.1, -2.3], [-2.3, -0.1]]), [1, 0])
    assert any("edge of the search range" in str(w.message) for w in caught)


@pytest.mark.parametrize("stated, hit_rate", [(0.95, 0.6125), (0.99, 0.3), (0.6, 0.97)])
def test_platt_on_a_constant_far_from_the_hit_rate_converges(stated, hit_rate):
    # Regression: a plain Newton step overshot from a stated 0.95 and ran away to 1.0.
    n = 800
    correct = np.r_[np.ones(round(n * hit_rate)), np.zeros(n - round(n * hit_rate))]
    fitted = fit_platt(np.full(n, stated), correct)
    assert fitted(np.array([stated]))[0] == pytest.approx(hit_rate, abs=0.005)


def test_platt_matches_a_brute_force_search_of_the_same_objective():
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.choice([0.6, 0.8, 0.9, 0.95, 0.99, 1.0], size=300)
        correct = (rng.random(300) < rng.uniform(0.3, 0.9)).astype(float)
        fitted = fit_platt(q, correct)
        x = logit(q)
        grid_a, grid_b = np.meshgrid(np.linspace(0, 3, 301), np.linspace(-6, 6, 601))
        z = grid_a[..., None] * x + grid_b[..., None]
        losses = (np.logaddexp(0, z) - correct * z).sum(-1) + 0.5e-3 * (grid_a ** 2 + grid_b ** 2)
        best = np.unravel_index(losses.argmin(), losses.shape)
        z_fit = fitted.a * x + fitted.b
        fit_loss = (np.logaddexp(0, z_fit) - correct * z_fit).sum() + 0.5e-3 * (fitted.a ** 2 + fitted.b ** 2)
        assert fit_loss <= losses[best] + 1e-6
