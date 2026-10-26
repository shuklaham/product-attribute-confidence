import pytest

from pac.output import parse_output
from pac.rewards import INVALID_REWARD, calibration_reward, reward

LETTERS = list("ABCD")


@pytest.mark.parametrize("text", [
    '{"choice": "B", "confidence": 0.83}',
    '  {"confidence": 1, "choice": "A"}\n',
])
def test_valid_outputs(text):
    assert parse_output(text, LETTERS).valid


@pytest.mark.parametrize("text, error", [
    ('```json\n{"choice": "B", "confidence": 0.8}\n```', "malformed_json"),
    ('{"choice": "B", "confidence": 0.8} because', "malformed_json"),
    ('["B", 0.8]', "not_an_object"),
    ('{"choice": "B"}', "wrong_keys"),
    ('{"choice": "B", "confidence": 0.8, "why": "neck"}', "wrong_keys"),
    ('{"choice": "E", "confidence": 0.8}', "unknown_choice"),
    ('{"choice": "B", "confidence": "high"}', "confidence_not_a_number"),
    ('{"choice": "B", "confidence": true}', "confidence_not_a_number"),
    ('{"choice": "B", "confidence": 1.2}', "confidence_out_of_range"),
    ('{"choice": "B", "confidence": NaN}', "confidence_out_of_range"),
])
def test_invalid_outputs_are_rejected_not_repaired(text, error):
    parsed = parse_output(text, LETTERS)
    assert not parsed.valid and parsed.error == error


def test_reward_values_from_the_plan():
    assert calibration_reward(True, 0.90) == pytest.approx(0.99)
    assert calibration_reward(False, 0.98) == pytest.approx(-0.9604)
    assert calibration_reward(False, 0.52) == pytest.approx(-0.2704)


def test_both_conditions_give_the_same_invalid_penalty():
    bad = parse_output("not json", LETTERS)
    assert reward(bad, "A", "correctness") == reward(bad, "A", "calibration") == INVALID_REWARD


def test_correctness_reward_ignores_confidence():
    sure = parse_output('{"choice": "A", "confidence": 0.99}', LETTERS)
    unsure = parse_output('{"choice": "A", "confidence": 0.01}', LETTERS)
    assert reward(sure, "A", "correctness") == reward(unsure, "A", "correctness") == 1.0


@pytest.mark.parametrize("p", [0.1, 0.35, 0.7, 0.95])
def test_brier_reward_is_maximised_by_honest_confidence(p):
    grid = [i / 100 for i in range(101)]
    expected = [p * calibration_reward(True, q) + (1 - p) * calibration_reward(False, q) for q in grid]
    assert grid[expected.index(max(expected))] == pytest.approx(p)
