"""GRPO for M3 and M4: the prompts, the two strata, the batch schedule and the advantages.

The rules live here as plain Python so they can be tested anywhere; the training loop is
`scripts/train_grpo.py` and the sampling pass that sorts prompts into strata is
`scripts/sample_pool.py`.

- The prompts are the training items as SFT's second pass would present them: the text condition
  and the option order come from the same seeded draws (src/pac/sft.py, epoch 1).
- Recipe v1 (Milestone 4): before RL, M1 answers every prompt GROUP times at temperature 1. A
  prompt is mixed if those answers are partly right and partly wrong, and settled if they are all
  right or all wrong. The strata stay fixed for the whole run.
- Recipe v2 (Milestone 5): every RESCORE_EVERY steps, the current policy scores every prompt with
  the probability p it gives the right letter where it writes its choice. A prompt is mixed if a
  group of GROUP samples would hold both right and wrong answers with probability at least
  SIGNAL_THRESHOLD. Under v1 the strata went stale: as the policy learned, 91% of groups ended up
  with no correctness signal.
- Every step takes half its prompts from each stratum, in a seeded order. Mixed prompts carry the
  correctness signal both arms learn from. Settled prompts, nearly all answered right, carry no
  correctness signal, but they are where M4's Brier term learns when to be sure.
- A sample's advantage is its reward minus its group's mean reward, not divided by the group's
  standard deviation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence

from .sft import presentation_epoch

EPOCH = 1
GROUP = 8
STRATA = ("mixed", "settled")
# Recipe v2. On M1's dev predictions, a threshold of 0.2 makes 5.6% of prompts mixed, and a group
# drawn from them holds both right and wrong answers 63% of the time.
SIGNAL_THRESHOLD = 0.2
RESCORE_EVERY = 50
TOTAL_STEPS = 461  # the length of the v1 runs, kept so the recipes train for as long


def prompts(examples: Sequence[dict]) -> list[dict]:
    """The RL prompts: one per training item, as SFT's second pass presents it."""
    return presentation_epoch(examples, EPOCH)


def stratum(correct: Sequence[int]) -> str:
    """mixed if the sampled answers are partly right and partly wrong; settled if all right or all wrong."""
    right = sum(correct)
    return "mixed" if 0 < right < len(correct) else "settled"


def expected_signal(p: float, group: int = GROUP) -> float:
    """Chance that `group` samples hold both right and wrong answers when each is right with probability p."""
    return 1 - p ** group - (1 - p) ** group


def strata_from_scores(scores: dict, threshold: float = SIGNAL_THRESHOLD, minimum: int = 4) -> dict[str, list]:
    """Recipe v2: sort prompts by the current policy's probability of the right letter.

    scores maps each prompt key to p. If fewer than `minimum` prompts clear the threshold, the
    prompts with the highest expected signal fill the mixed stratum up to `minimum`, so a batch can
    always be built.
    """
    ranked = sorted(scores, key=lambda k: (-expected_signal(scores[k]), k))
    n_mixed = max(minimum, sum(expected_signal(scores[k]) >= threshold for k in ranked))
    return {"mixed": ranked[:n_mixed], "settled": ranked[n_mixed:]}


def windows(total_steps: int, every: int = RESCORE_EVERY) -> list[tuple[int, int]]:
    """The [start, end) step ranges between re-scorings, the first starting at step 0."""
    return [(start, min(start + every, total_steps)) for start in range(0, total_steps, every)]


def window_seed(seed: int, window: int) -> int:
    """The schedule seed for one window of one training seed."""
    return seed * 1_000 + window


def steps_for(n_mixed: int, per_stratum: int = 4, passes: int = 3, cap: int = 500) -> int:
    """Training steps: enough for every mixed prompt to come up `passes` times, and at most `cap`."""
    return min(cap, math.ceil(passes * n_mixed / per_stratum))


def schedule(pool: dict[str, Sequence[Hashable]], steps: int, per_stratum: int = 4,
             seed: int = 0) -> list[list[Hashable]]:
    """The prompts of every step: per_stratum from each stratum, mixed first, in a seeded order.

    Each stratum is walked in a shuffled order and reshuffled after a full pass, so a prompt comes
    up again only after every other prompt in its stratum. When a pass ends inside a step, the
    prompts already in that step go last in the next pass, so no step holds a prompt twice. Both
    arms use the same schedule.
    """
    for s in STRATA:
        if len(set(pool[s])) < per_stratum:
            raise ValueError(f"the {s} stratum has fewer than {per_stratum} prompts")
    rng = random.Random(seed)
    queues: dict[str, list] = {s: [] for s in STRATA}
    out = []
    for _ in range(steps):
        batch = []
        for s in STRATA:
            taken: set = set()
            for _ in range(per_stratum):
                if not queues[s]:
                    order = sorted(set(pool[s]))
                    rng.shuffle(order)
                    # pop() takes from the end, so this step's prompts are drawn last.
                    queues[s] = [k for k in order if k in taken] + [k for k in order if k not in taken]
                key = queues[s].pop()
                taken.add(key)
                batch.append(key)
        out.append(batch)
    return out


def advantages(rewards: Sequence[float]) -> list[float]:
    """Each sample's reward minus its group's mean.

    Not dividing by the group's standard deviation matters for M4: the rewards in a group whose
    answers are all right differ only through the stated confidence, often by thousandths, and
    dividing by their tiny spread would turn them into updates as large as the gap between a right
    and a wrong answer.
    """
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]
