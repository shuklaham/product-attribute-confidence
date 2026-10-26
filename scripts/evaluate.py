"""Metric tables for prediction files written by scripts/predict.py.

One row per (evidence mode, attribute, text condition). Accuracy comes with a 95% interval from
a bootstrap over product families. With --train, the category guess (the most common training
label of the item's category) is shown for comparison.

Usage:
    uv run python -I scripts/evaluate.py <predictions.jsonl>... [--train <train.jsonl>] [--out <report.md>]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from pac.evaluation import category_guess, summarize


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pct(x) -> str:
    return "–" if x is None or math.isnan(x) else f"{x:.1%}"


def num(x) -> str:
    return "–" if x is None or math.isnan(x) else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    guess = category_guess(load(args.train)) if args.train else None
    groups = defaultdict(list)
    runs = []
    for path in args.predictions:
        for r in load(path):
            groups[(r["mode"], r["attribute"], r["condition"])].append(r)
        run_path = path.parent / "run.json"
        if run_path.exists():
            runs.append(json.loads(run_path.read_text(encoding="utf-8")))
    summaries = {key: summarize(rows, guess=guess) for key, rows in sorted(groups.items())}

    lines = ["# Results", ""]
    for run in runs:
        lines.append(f"- Run {run['started']}: {run['model']['name']} at `{run['model']['revision'][:8]}`, "
                     f"commit `{run['git']['commit'][:8]}`{' (dirty)' if run['git']['dirty'] else ''}, "
                     f"{run['filters']['mode']}, {run['n']:,} items, {run['gpu']}, "
                     f"{run['elapsed_seconds'] / 60:.1f} min")
    lines += ["", "## Answers", "",
              "| Mode | Attribute | Text | n | Accuracy (95% interval) | Macro accuracy | Category guess | Invalid |",
              "|---|---|---|---:|---|---:|---:|---:|"]
    for (mode, attribute, condition), s in summaries.items():
        low, high = s["accuracy_interval"]
        lines.append(f"| {mode} | {attribute} | {condition} | {s['n']:,} | {pct(s['accuracy'])} "
                     f"({pct(low)} to {pct(high)}) | {pct(s['macro_accuracy'])} | {pct(s['category_guess'])} | "
                     f"{pct(s['invalid'])} |")
    lines += ["", "## Stated confidence", "",
              ("| Mode | Attribute | Text | Brier | Brier, invalid = 1 | ECE | AUROC | Risk–coverage area | "
               "Mean | Distinct values | Most common (share) |"),
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for (mode, attribute, condition), s in summaries.items():
        top = "–" if s["top_confidence"] is None else f"{s['top_confidence']:g} ({pct(s['top_confidence_share'])})"
        lines.append(f"| {mode} | {attribute} | {condition} | {num(s['brier'])} | {num(s['brier_invalid_as_1'])} | "
                     f"{num(s['ece'])} | {num(s['auroc'])} | {num(s['aurc'])} | {num(s['mean_confidence'])} | "
                     f"{s['distinct_confidences']} | {top} |")
    lines += ["", "## Answer-letter probability (temperature 1)", "",
              "| Mode | Attribute | Text | ECE | AUROC | Risk–coverage area | Choice position not found |",
              "|---|---|---|---:|---:|---:|---:|"]
    for (mode, attribute, condition), s in summaries.items():
        lines.append(f"| {mode} | {attribute} | {condition} | {num(s['letter_ece'])} | {num(s['letter_auroc'])} | "
                     f"{num(s['letter_aurc'])} | {s['letter_missing']} |")
    lines += ["", "## Invalid outputs by reason", ""]
    for (mode, attribute, condition), s in summaries.items():
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(s["errors"].items(), key=lambda kv: -kv[1])) or "none"
        lines.append(f"- {mode} / {attribute} / {condition}: {reasons}")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
