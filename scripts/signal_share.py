"""Expected share of GRPO groups whose sampled answers are partly right and partly wrong, from dev.

A group of k answers to one prompt teaches the correctness reward something only if it holds
both right and wrong answers. If the model samples the correct letter with probability p, that
happens with probability 1 - p^k - (1 - p)^k. p comes from the letter log-probabilities a
prediction run recorded where the model writes its choice, renormalized over the offered letters
and sharpened or flattened by the sampling temperature. Runs locally on prediction files; no GPU.

Usage:
    uv run python -I scripts/signal_share.py <predictions.jsonl> [--group 8] [--temperatures 1.0 1.5 2.0]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def correct_letter_probability(record: dict, temperature: float) -> float | None:
    logprobs = record["letter_logprobs"]
    if not logprobs:
        return None
    letters = list(logprobs)
    z = np.array([logprobs[x] for x in letters]) / temperature
    p = np.exp(z - z.max())
    return float(p[letters.index(record["answer"])] / p.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    keys = sorted({(r["attribute"], r["condition"]) for r in records})
    print(f"Expected share of groups of {args.group} with both right and wrong answers ({args.predictions})\n")
    print("| Temperature | All | " + " | ".join(f"{a}, {c}" for a, c in keys) + " |")
    print("|---:|---:|" + "---:|" * len(keys))
    for t in args.temperatures:
        mixed = defaultdict(list)
        for r in records:
            p = correct_letter_probability(r, t)
            if p is not None:
                mixed[(r["attribute"], r["condition"])].append(1 - p ** args.group - (1 - p) ** args.group)
        pooled = [m for values in mixed.values() for m in values]
        print(f"| {t:g} | {np.mean(pooled):.1%} | " + " | ".join(f"{np.mean(mixed[k]):.1%}" for k in keys) + " |")
    p = [correct_letter_probability(r, 1.0) for r in records]
    p = np.array([x for x in p if x is not None])
    print(f"\nItems: {len(p):,}. Probability of the correct letter at temperature 1: above 0.99 for "
          f"{np.mean(p > 0.99):.1%}, above 0.999 for {np.mean(p > 0.999):.1%}.")


if __name__ == "__main__":
    main()
