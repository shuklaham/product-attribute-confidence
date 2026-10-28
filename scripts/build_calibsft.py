"""Build M5's training data: M1's sampled answers with CalibSFT targets (src/pac/calibsft.py).

Runs locally on the sampling pass that sorted the GRPO prompts (artifacts/grpo_pool).

Usage:
    uv run python -I scripts/build_calibsft.py data/processed/v1/train.jsonl artifacts/grpo_pool/samples.jsonl \
        data/processed/v1/calibsft.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pac.calibsft import CAP, balance, examples_from_pool
from pac.grpo import prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    with open(args.train, encoding="utf-8") as f:
        by_key = {(p["product_id"], p["attribute"]): p for p in prompts([json.loads(line) for line in f])}
    with open(args.samples, encoding="utf-8") as f:
        examples = examples_from_pool((json.loads(line) for line in f), by_key)
    kept = balance(examples, CAP)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e) + "\n")
    counts = Counter(e["target_confidence"] for e in kept)
    print("kept per target:", " ".join(f"{v}:{counts[v]}" for v in sorted(counts, key=float)))
    print(f"{len(kept):,} of {len(examples):,} answers kept; {sum(e['target_correct'] for e in kept):,} right; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
