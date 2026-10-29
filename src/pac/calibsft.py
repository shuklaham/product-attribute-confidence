"""M5: SFT on measured success rates, in the style of CalibSFT (arXiv 2609.32470).

M1's own sampled answers to each RL prompt (scripts/sample_pool.py: 8 per prompt at temperature
1) become SFT targets with their confidence replaced by

    target = λ · q + (1 − λ) · z

where q is the prompt's measured success rate (invalid answers count as wrong) and z is 1 if that
answer is right. λ = 0.5 as in CalibSFT. Right and wrong answers are both kept.

Of M1's answers, 91% have a target of exactly 1.0, so answers are capped at CAP per target value,
a seeded sample. CalibSFT draws equal numbers per value; here the rarest value has 30 answers, so
a cap keeps enough data while still balancing.

The loss (scripts/train_sft.py, --calibrated) covers the confidence of every answer, and the rest of
the answer only when it is right. Each part is averaged over its own tokens, and the two are weighted
BETA and 1 − BETA.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

LAMBDA = 0.5
BETA = 0.5
CAP = 300
SALT = "calibsft-v1"


def _rank(*parts) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in (SALT, *parts)).encode()).hexdigest(), 16)


def target_confidence(q: float, z: int, lam: float = LAMBDA) -> str:
    """The confidence an answer is trained to state, written as the model writes numbers."""
    return f"{lam * q + (1 - lam) * z:.2f}"


def examples_from_pool(pool_records: Iterable[dict], by_key: dict) -> list[dict]:
    """One training example per valid sampled answer: the prompt as presented, with its target."""
    out = []
    for r in pool_records:
        prompt = by_key[(r["product_id"], r["attribute"])]
        if (prompt["answer"], prompt["condition"]) != (r["answer"], r["condition"]):
            raise ValueError(f"the pool does not describe the prompt {r['example_id']}")
        q = sum(r["correct"]) / len(r["correct"])
        for i, (choice, z, error) in enumerate(zip(r["choices"], r["correct"], r["errors"])):
            if error is None:
                out.append({**prompt, "example_id": f"{prompt['example_id']}:s{i}", "target_choice": choice,
                            "target_confidence": target_confidence(q, z), "target_correct": z})
    return out


def balance(examples: list[dict], cap: int = CAP) -> list[dict]:
    """At most `cap` examples per target value, a seeded sample; every example of rarer values."""
    by_value = defaultdict(list)
    for e in examples:
        by_value[e["target_confidence"]].append(e)
    kept = []
    for _, group in sorted(by_value.items()):
        kept += sorted(group, key=lambda e: _rank("keep", e["example_id"]))[:cap]
    return sorted(kept, key=lambda e: _rank("order", e["example_id"]))
