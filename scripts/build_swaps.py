"""Build the photo-swap examples: held-out items with another product's photo (src/pac/swap.py).

Writes both text conditions of 500 clean items per attribute, each with its donor's photo, and
prints how many items had a donor at all. With a third path, it also writes the same items
unswapped, for the photo-only and text-only runs on the same subset. Runs locally.

Usage:
    uv run python -I scripts/build_swaps.py data/processed/v1/heldout_test.jsonl data/processed/v1/swap_heldout.jsonl \
        [data/processed/v1/subset_heldout.jsonl]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pac.swap import PER_ATTRIBUTE, donors, subset, swapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("examples", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("subset_out", type=Path, nargs="?")
    args = parser.parse_args()

    with open(args.examples, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    chosen_donors = donors(examples)
    picked = subset(sorted(chosen_donors), PER_ATTRIBUTE)
    by_item = {(e["product_id"], e["attribute"], e["condition"]): e for e in examples}
    rows = []
    for pid, attribute in picked:
        donor = by_item[(chosen_donors[(pid, attribute)], attribute, "full")]
        rows += [swapped(by_item[(pid, attribute, condition)], donor) for condition in ("full", "redacted")]
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    if args.subset_out:
        with open(args.subset_out, "w", encoding="utf-8") as f:
            for pid, attribute in picked:
                for condition in ("full", "redacted"):
                    f.write(json.dumps(by_item[(pid, attribute, condition)]) + "\n")

    clean = {(e["product_id"], e["attribute"]) for e in examples if e["clean"]}
    for attribute in ("neckline", "sleeve_length"):
        n_clean = sum(k[1] == attribute for k in clean)
        n_donor = sum(k[1] == attribute for k in chosen_donors)
        print(f"{attribute}: {n_donor:,} of {n_clean:,} clean items have a donor; "
              f"{sum(k[1] == attribute for k in picked)} chosen")
    print("chosen by category:", dict(Counter(r["category"] for r in rows if r["condition"] == "full")))
    print(f"wrote {len(rows):,} examples to {args.out}")


if __name__ == "__main__":
    main()
