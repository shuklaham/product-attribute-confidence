from collections import Counter

import pytest
from conftest import make_product

from pac.examples import make_example
from pac.grpo import EPOCH, advantages, prompts, schedule, steps_for, stratum
from pac.rewards import calibration_reward
from pac.sft import condition_for, presentation_epoch, presentations


def _examples(neckline, n=30):
    out = []
    for pid in range(1, n + 1):
        product = make_product(pid, description="Navy T-shirt, has a round neck")
        for condition in ("full", "redacted"):
            out.append(make_example(product, neckline, "Round neck", "train", pid // 2, condition,
                                    text_status="agrees", clean=True))
    return out


def test_prompts_are_the_second_sft_pass(neckline):
    examples = _examples(neckline)
    rl = prompts(examples)
    assert rl == presentations(examples, 2)[len(rl):] == presentation_epoch(examples, EPOCH)
    assert all(p["condition"] == condition_for(p["family_id"], p["attribute"], EPOCH) for p in rl)
    assert len(rl) == 30


def test_strata():
    assert stratum([1] * 8) == stratum([0] * 8) == "settled"
    assert stratum([1] * 7 + [0]) == stratum([0] * 7 + [1]) == "mixed"


def test_steps_give_each_mixed_prompt_three_passes_up_to_the_cap():
    assert steps_for(425) == 319  # 3 * 425 / 4 = 318.75
    assert steps_for(400) == 300
    assert steps_for(10_000) == 500


def _pool(n_mixed=10, n_settled=50):
    return {"mixed": [("m", i) for i in range(n_mixed)], "settled": [("s", i) for i in range(n_settled)]}


def test_schedule_is_seeded_balanced_and_shared_by_both_arms():
    pool = _pool()
    plan = schedule(pool, steps=40)
    assert plan == schedule(pool, steps=40) and plan != schedule(pool, steps=40, seed=1)
    for batch in plan:
        assert [k[0] for k in batch] == ["m"] * 4 + ["s"] * 4
        assert len(set(batch)) == 8


def test_schedule_finishes_a_pass_before_repeating():
    plan = schedule(_pool(n_mixed=10), steps=30)  # 120 mixed draws: 12 full passes over 10 prompts
    mixed = [k for batch in plan for k in batch[:4]]
    for start in range(0, 120, 10):
        assert sorted(mixed[start:start + 10]) == [("m", i) for i in range(10)]
    assert set(Counter(mixed).values()) == {12}


def test_schedule_never_repeats_a_prompt_within_a_step_across_a_pass_boundary():
    # 6 mixed prompts and 4 per step: the second step crosses the end of the first pass.
    for seed in range(50):
        for batch in schedule(_pool(n_mixed=6), steps=9, seed=seed):
            assert len(set(batch)) == 8


def test_schedule_needs_enough_prompts_per_stratum():
    with pytest.raises(ValueError, match="mixed"):
        schedule(_pool(n_mixed=3), steps=1)


def test_advantages_are_centred_but_not_scaled():
    assert advantages([1, 0, 1, 1]) == pytest.approx([0.25, -0.75, 0.25, 0.25])
    assert advantages([1.0] * 8) == [0.0] * 8
    # An all-right M4 group: the confidence differences stay as small as they are.
    rewards = [calibration_reward(True, q) for q in (0.95, 0.95, 0.98, 0.9)]
    assert max(abs(a) for a in advantages(rewards)) < 0.01
