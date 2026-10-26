"""Metric tables for prediction files written by scripts/predict.py.

Each predictions file is one run and must sit next to its run record (run.json), which
predict.py writes only after the last prediction; a file whose record count differs from the
run record is refused. One row per (run, split, subset, evidence mode, attribute, text
condition), so different models or splits never merge. The subset is the clean items, plus,
where a split has them, all labeled items including those excluded only as label noise.
Accuracy comes with a 95% interval from a bootstrap over product families. With --train, the
category guess (the most common training label of the item's category) is shown for comparison.

Usage:
    uv run python -I scripts/evaluate.py <predictions.jsonl>... [--train <train.jsonl>] [--out <report.md>]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from pac.evaluation import category_guess, evidence_sensitivity, summarize


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
    groups, runs = defaultdict(list), {}
    for path in args.predictions:
        run_path = path.parent / "run.json"
        if not run_path.exists():
            raise SystemExit(f"{path}: no run.json beside it, so the run may not have finished")
        run, records = json.loads(run_path.read_text(encoding="utf-8")), load(path)
        if len(records) != run["n"]:
            raise SystemExit(f"{path}: {len(records):,} records, but its run record says {run['n']:,}")
        name = path.parent.name
        if name in runs:
            raise SystemExit(f"two prediction files come from runs named {name!r}")
        runs[name] = run
        for r in records:
            if r["clean"]:
                groups[(name, r["split"], "clean", r["mode"], r["attribute"], r["condition"])].append(r)
            groups[(name, r["split"], "all labeled", r["mode"], r["attribute"], r["condition"])].append(r)
    groups = {key: rows for key, rows in groups.items()
              if key[2] == "clean" or len(rows) > len(groups.get(key[:2] + ("clean",) + key[3:], []))}
    summaries = {key: summarize(rows, guess=guess) for key, rows in sorted(groups.items())}

    lines = ["# Results", ""]
    for name, run in runs.items():
        lines.append(f"- `{name}`, started {run['started']}: {run['model']['name']} at "
                     f"`{run['model']['revision'][:8]}`, commit `{run['git']['commit'][:8]}`"
                     f"{' (uncommitted changes)' if run['git']['dirty'] else ''}, {run['filters']['mode']}, "
                     f"{run['n']:,} items, {run['gpu']}, {run['elapsed_seconds'] / 60:.1f} min")
    head = "| Run | Split | Subset | Mode | Attribute | Text |"
    rule = "|---|---|---|---|---|---|"
    lines += ["", "## Answers", "",
              head + " n | Accuracy (95% interval) | Accuracy on valid outputs | Macro accuracy | Category guess "
                     "| Invalid |",
              rule + "---:|---|---:|---:|---:|---:|"]
    for key, s in summaries.items():
        low, high = s["accuracy_interval"]
        lines.append("| " + " | ".join(key) + f" | {s['n']:,} | {pct(s['accuracy'])} ({pct(low)} to {pct(high)}) | "
                     f"{pct(s['accuracy_valid'])} | {pct(s['macro_accuracy'])} | {pct(s['category_guess'])} | "
                     f"{pct(s['invalid'])} |")
    lines += ["", "## Stated confidence", "",
              head + " Brier | Brier, invalid = 1 | ECE | AUROC | Risk–coverage area | Mean | Distinct values | "
                     "Most common (share) |",
              rule + "---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, s in summaries.items():
        top = "–" if s["top_confidence"] is None else f"{s['top_confidence']:g} ({pct(s['top_confidence_share'])})"
        lines.append("| " + " | ".join(key) + f" | {num(s['brier'])} | {num(s['brier_invalid_as_1'])} | "
                     f"{num(s['ece'])} | {num(s['auroc'])} | {num(s['aurc'])} | {num(s['mean_confidence'])} | "
                     f"{s['distinct_confidences']} | {top} |")
    lines += ["", "## Answer-letter probability (temperature 1)", "",
              head + " ECE | AUROC | Risk–coverage area | Choice position not found |",
              rule + "---:|---:|---:|---:|"]
    for key, s in summaries.items():
        lines.append("| " + " | ".join(key) + f" | {num(s['letter_ece'])} | {num(s['letter_auroc'])} | "
                     f"{num(s['letter_aurc'])} | {s['letter_missing']} |")
    lines += ["", "## Accuracy by category", "",
              "Pooled accuracy is dominated by T-shirts, so each category is shown on its own.", "",
              head + " Category | n | Accuracy |", rule + "---|---:|---:|"]
    for key, rows in sorted(groups.items()):
        by_category = defaultdict(list)
        for r in rows:
            by_category[r["category"]].append(r["correct"])
        for category, hits in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
            lines.append("| " + " | ".join(key) + f" | {category} | {len(hits):,} | {pct(sum(hits) / len(hits))} |")

    by_condition = defaultdict(dict)
    for key, rows in groups.items():
        by_condition[key[:5]][key[5]] = rows
    sensitivity = {key: evidence_sensitivity(c["full"], c["redacted"])
                   for key, c in sorted(by_condition.items()) if {"full", "redacted"} <= c.keys()}
    if sensitivity:
        lines += ["", "## Evidence sensitivity (full minus redacted text)", "",
                  "Both drops use the items valid in both conditions; the invalid rates are over all items.", "",
                  ("| Run | Split | Subset | Mode | Attribute | Items | Valid in both | Confidence drop "
                   "| Accuracy drop | Invalid, full | Invalid, redacted |"),
                  "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for key, s in sensitivity.items():
            lines.append("| " + " | ".join(key) + f" | {s['n']:,} | {s['valid_in_both']:,} | "
                         f"{num(s['confidence_drop'])} | {pct(s['accuracy_drop'])} | {pct(s['invalid_full'])} | "
                         f"{pct(s['invalid_redacted'])} |")
    lines += ["", "## Invalid outputs by reason", ""]
    for key, s in summaries.items():
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(s["errors"].items(), key=lambda kv: -kv[1])) or "none"
        lines.append(f"- {' / '.join(key)}: {reasons}")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
