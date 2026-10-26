"""Cheap text-only leak check, run on CPU before the model-based check.

A naive Bayes bag-of-words classifier is trained on the training split and scored on dev,
for full and for redacted text, next to the category-majority guess. If redacted text beats
the guess, the words the classifier leans on are listed, so obvious clues can be added to the
redaction word list. Only train and dev are read; the test splits are never touched.

Usage:
    uv run python -I scripts/leak_check.py <examples_dir>
"""

import collections
import json
import math
import re
import sys
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9/'-]+")
MIN_COUNT = 8


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def words(example: dict) -> set[str]:
    return set(TOKEN.findall(f"{example['title']} {example['description']}".lower()))


class NaiveBayes:
    def __init__(self, examples):
        self.labels = collections.Counter(e["reference"] for e in examples)
        self.counts = collections.defaultdict(collections.Counter)
        for e in examples:
            self.counts[e["reference"]].update(words(e))
        self.vocab = set().union(*self.counts.values())
        self.totals = {label: sum(c.values()) + len(self.vocab) for label, c in self.counts.items()}

    def predict(self, example: dict) -> str:
        n = sum(self.labels.values())
        present = words(example) & self.vocab

        def score(label):
            return math.log(self.labels[label] / n) + sum(
                math.log((self.counts[label][w] + 1) / self.totals[label]) for w in present)

        return max(self.labels, key=score)

    def clue_words(self, label: str, k: int = 8) -> list[str]:
        """Words most over-represented in one label relative to the others."""
        rest = collections.Counter()
        for other, counts in self.counts.items():
            if other != label:
                rest.update(counts)
        rest_total = sum(rest.values()) + len(self.vocab)

        def log_odds(w):
            return math.log((self.counts[label][w] + 1) / self.totals[label]) - math.log((rest[w] + 1) / rest_total)

        frequent = [w for w, c in self.counts[label].items() if c >= MIN_COUNT]
        return sorted(frequent, key=log_odds, reverse=True)[:k]


def accuracy(pairs) -> tuple[float, float]:
    """Pooled accuracy and balanced accuracy (mean recall over labels)."""
    pairs = list(pairs)
    hits = collections.Counter(t for t, p in pairs if t == p)
    totals = collections.Counter(t for t, _ in pairs)
    pooled = sum(hits.values()) / len(pairs)
    balanced = sum(hits[t] / totals[t] for t in totals) / len(totals)
    return pooled, balanced


def main(examples_dir: str) -> None:
    root = Path(examples_dir)
    train, dev = load(root / "train.jsonl"), load(root / "dev.jsonl")
    for attribute in sorted({e["attribute"] for e in dev}):
        print(f"\n== {attribute} (dev)")
        by_category = collections.defaultdict(collections.Counter)
        for e in train:
            if e["attribute"] == attribute and e["condition"] == "full":
                by_category[e["category"]][e["reference"]] += 1
        dev_full = [e for e in dev if e["attribute"] == attribute and e["condition"] == "full"]
        guess = accuracy((e["reference"], by_category[e["category"]].most_common(1)[0][0]) for e in dev_full)
        print(f"   {'category guess':<16} accuracy {guess[0]:.0%}   balanced {guess[1]:.0%}")
        for condition in ("full", "redacted"):
            model = NaiveBayes([e for e in train if e["attribute"] == attribute and e["condition"] == condition])
            scored = accuracy((e["reference"], model.predict(e)) for e in dev
                              if e["attribute"] == attribute and e["condition"] == condition)
            print(f"   {condition + ' text':<16} accuracy {scored[0]:.0%}   balanced {scored[1]:.0%}")
        model = NaiveBayes([e for e in train if e["attribute"] == attribute and e["condition"] == "redacted"])
        print("   clue words in redacted training text:")
        for label, n in model.labels.most_common():
            if n >= 20:
                print(f"     {label:<22} {', '.join(model.clue_words(label))}")


if __name__ == "__main__":
    main(sys.argv[1])
