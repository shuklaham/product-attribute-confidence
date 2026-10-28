"""Phase-2 report: RL from M1 without (M3) and with (M4) a calibration reward, against M0, M1, M2
and M2-logit, on the seen and held-out test splits.

Every method keeps its own auto-fill threshold, chosen on its predictions for the threshold half of
the calibration split. M2's Platt map and M2-logit's temperature are fitted on M1's map-fitting
half, as in phase 1; M3 and M4 are scored on their stated confidence as it is. Nothing is tuned on
test.

The primary comparison was fixed before any training (docs/plan.md, section 7): M4 against M2-logit
on the held-out test, redacted text, clean items, both attributes pooled. M4 counts as better if
the 95% interval of the difference in Brier score (invalid outputs count as loss 1) lies below zero
and the lower bound of the accuracy difference is above −1 point. Intervals come from a paired
bootstrap over product families that refits M2's map and M2-logit's temperature on a resample of
the calibration families in every draw.

Usage:
    uv run python -I scripts/report_phase2.py [--artifacts artifacts] [--out reports/phase2.md]
        [--figures reports/figures] [--resamples 10000]
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pac import evaluation, methods, metrics  # noqa: E402
from pac.calibration import fit_platt, fit_temperature  # noqa: E402
from pac.reporting import arrays, evaluate, family_index, load_run, mapped, num, pct  # noqa: E402

SPLITS = {"test": "Seen test", "heldout_test": "Held-out test"}
METHODS = {"M0": ("m0", "stated"), "M1": ("m1", "stated"), "M2": ("m1", "platt"), "M2-logit": ("m1", "letter"),
           "M3": ("m3", "stated"), "M4": ("m4", "stated")}
ARMS = {"M4": "m4", "M3": "m3"}
# Reference palette, light mode, in its fixed categorical order, so each method keeps its phase-1 colour.
# Charts with shared axes show M2-logit, M3 and M4 only: that set passes the validator for all pairs,
# with line style as the second cue its colour-blind check asks for.
COLORS = {"M0": "#2a78d6", "M1": "#eb6834", "M2-logit": "#1baf7a", "M2": "#eda100", "M3": "#e87ba4", "M4": "#008300"}
STYLES = {"M2-logit": "-", "M3": "--", "M4": "-."}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PRIMARY_BRIER = "Brier, M4 minus M2-logit"
PRIMARY_ACCURACY = "Accuracy, M4 minus M2-logit"


def bootstrap(cal: dict, test: dict, others: dict, resamples: int, seed: int = 0) -> dict:
    """Paired family bootstrap of the comparisons on one test set, refitting M2's and M2-logit's maps every draw.

    cal and test map attribute -> arrays of M1's records; others maps M0, M3 and M4 to arrays of
    their records for the same test items, in the same order.
    """
    cal_index, n_cal = family_index(cal)
    test_index, n_test = family_index(test)

    def statistics(cal_counts, test_counts):
        q = {m: [] for m in METHODS}
        c = {m: [] for m in METHODS}
        for k in test:
            cal_rows = np.repeat(np.arange(len(cal_index[k])), cal_counts[cal_index[k]])
            test_rows = np.repeat(np.arange(len(test_index[k])), test_counts[test_index[k]])
            fit = cal[k]
            platt = fit_platt(fit["stated"][cal_rows], fit["correct"][cal_rows])
            with_letters = cal_rows[fit["has_letters"][cal_rows]]
            t = fit_temperature(fit["letters"][with_letters], fit["answer"][with_letters], iterations=40).t
            m2, m2_logit = mapped(test[k], test_rows, platt, t)
            for name, values in (("M1", test[k]["stated"][test_rows]), ("M2", m2), ("M2-logit", m2_logit)):
                q[name].append(values)
                c[name].append(test[k]["correct"][test_rows])
            for name in ("M0", "M3", "M4"):
                q[name].append(others[name][k]["stated"][test_rows])
                c[name].append(others[name][k]["correct"][test_rows])
        q = {m: np.concatenate(v) for m, v in q.items()}
        c = {m: np.concatenate(v) for m, v in c.items()}
        brier = {m: metrics.brier(q[m], c[m], invalid_loss=1.0) for m in METHODS}
        auroc = {m: metrics.auroc(q[m], c[m]) for m in ("M2-logit", "M3", "M4")}
        accuracy = {m: float(c[m].mean()) for m in METHODS}
        return {
            PRIMARY_BRIER: brier["M4"] - brier["M2-logit"],
            PRIMARY_ACCURACY: accuracy["M4"] - accuracy["M2-logit"],
            "Brier, M4 minus M3": brier["M4"] - brier["M3"],
            "Brier, M4 minus M2": brier["M4"] - brier["M2"],
            "Brier, M3 minus M2-logit": brier["M3"] - brier["M2-logit"],
            "AUROC, M4 minus M2-logit": auroc["M4"] - auroc["M2-logit"],
            "AUROC, M4 minus M3": auroc["M4"] - auroc["M3"],
            "Accuracy, M4 minus M3": accuracy["M4"] - accuracy["M3"],
            "Accuracy, M3 minus M1": accuracy["M3"] - accuracy["M1"],
        }

    point = statistics(np.ones(n_cal, dtype=int), np.ones(n_test, dtype=int))
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in point}
    for _ in range(resamples):
        values = statistics(np.bincount(rng.integers(0, n_cal, n_cal), minlength=n_cal),
                            np.bincount(rng.integers(0, n_test, n_test), minlength=n_test))
        for key, value in values.items():
            draws[key].append(value)
    out = {}
    for key, values in draws.items():
        values = np.array(values)
        undefined = int(np.isnan(values).sum())
        low, high = np.nanquantile(values, [0.025, 0.975])
        out[key] = (point[key], float(low), float(high), undefined)
    return out


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def reliability_figure(confidences: dict, correct: dict, path: Path) -> None:
    shown = ("M2-logit", "M3", "M4")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharex=True, sharey=True, facecolor=SURFACE)
    for row, split in enumerate(SPLITS):
        for col, method in enumerate(shown):
            ax = axes[row][col]
            q, c = confidences[(split, method)], correct[(split, method)]
            bins = [b for b in metrics.reliability(q, c) if b.count]
            ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=0.8)
            ax.scatter([b.mean_confidence for b in bins], [b.accuracy for b in bins], color=COLORS[method],
                       s=[12 + 300 * b.count / len(q) for b in bins], edgecolor=SURFACE, linewidth=1.2, zorder=3)
            style(ax, f"{method} · {SPLITS[split]}\nECE {metrics.ece(q, c):.3f}")
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            if row == 1:
                ax.set_xlabel("Confidence", color=INK_2, fontsize=9)
            if col == 0:
                ax.set_ylabel("Accuracy", color=INK_2, fontsize=9)
    fig.suptitle("Reliability on redacted text (clean items, both attributes). Marker area: share of items in the bin",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def risk_coverage_figure(confidences: dict, correct: dict, operating: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, split in zip(axes, SPLITS):
        for method in ("M2-logit", "M3", "M4"):
            coverage, risk = metrics.risk_curve(confidences[(split, method)], correct[(split, method)])
            ax.plot(coverage, risk, color=COLORS[method], linestyle=STYLES[method], linewidth=1.8, label=method)
            at, error = operating[(split, method)]
            if not math.isnan(error):
                ax.scatter([at], [error], color=COLORS[method], s=48, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.axhline(0.05, linestyle=":", color=INK_2, linewidth=0.8)
        ax.text(0.4, 0.0495, "5% error target", color=INK_2, fontsize=8, va="top", ha="left")
        style(ax, SPLITS[split])
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Coverage (share auto-filled)", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Error rate among auto-filled items", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Risk–coverage on redacted text (clean items, both attributes)", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def rolling(values: list, window: int = 10) -> np.ndarray:
    """Trailing mean over up to `window` steps, skipping steps with no value."""
    x = np.array([np.nan if v is None else v for v in values], dtype=float)
    return np.array([np.nanmean(x[max(0, i - window + 1):i + 1]) if np.isfinite(x[max(0, i - window + 1):i + 1]).any()
                     else np.nan for i in range(len(x))])


def training_figure(histories: dict, path: Path) -> None:
    panels = [("accuracy", "Accuracy of sampled answers"), ("brier", "Brier score of sampled answers"),
              ("confidence", "Stated confidence: right (solid), wrong (dashed)"),
              ("no_correctness_signal", "Groups with no correctness signal")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, panels):
        for method, history in histories.items():
            steps = [h["step"] for h in history]
            if key == "confidence":
                ax.plot(steps, rolling([h["confidence_right"] for h in history]), color=COLORS[method],
                        linewidth=1.6, label=f"{method}, right")
                ax.plot(steps, rolling([h["confidence_wrong"] for h in history]), color=COLORS[method],
                        linewidth=1.6, linestyle="--", label=f"{method}, wrong")
            else:
                # M3 and M4 alone pass the colour checks, so solid lines; dashes mean "wrong" in the confidence panel.
                ax.plot(steps, rolling([h.get(key) for h in history]), color=COLORS[method], linewidth=1.6,
                        label=method)
        style(ax, title)
    for ax in axes[1]:
        ax.set_xlabel("Step (trailing mean over 10 steps)", color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("GRPO training, both arms on the same prompts in the same order", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def mean(values) -> float:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def decompose(q: list, correct: list, group) -> dict:
    """Murphy's decomposition of the Brier score over groups of forecasts; exact when each group holds one value.

    Every output here is valid, so the Brier score with invalid outputs counted as 1 is the plain one.
    """
    if any(v is None for v in q):
        raise ValueError("the decomposition needs a confidence for every answer")
    groups = defaultdict(list)
    for v, c in zip(q, correct):
        groups[group(v)].append((v, c))
    n, a = len(correct), sum(correct) / len(correct)
    calibration = sum(len(g) / n * (np.mean([v for v, _ in g]) - np.mean([c for _, c in g])) ** 2
                      for g in groups.values())
    resolution = sum(len(g) / n * (np.mean([c for _, c in g]) - a) ** 2 for g in groups.values())
    return {"calibration": float(calibration), "resolution": float(resolution), "uncertainty": a * (1 - a),
            "groups": len(groups)}


def rank_correlation(x, y) -> float:
    """Spearman's rank correlation, ties given their average rank; NaN if either side is constant."""
    def ranks(values):
        _, inverse, counts = np.unique(np.asarray(values, dtype=float), return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("reports/phase2.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    art = args.artifacts

    runs = {"m0": {s: load_run(art, f"m0_{s}") for s in ("cal_threshold", *SPLITS)},
            "m1": {s: load_run(art, f"m1_{s}") for s in ("cal_fit", "cal_threshold", *SPLITS)}}
    for arm in ARMS.values():
        runs[arm] = {s: load_run(art, f"{arm}_{s}") for s in ("cal_threshold", *SPLITS)}
    fitted = methods.fit(runs["m1"]["cal_fit"])
    thresholds = {name: methods.thresholds(runs[source]["cal_threshold"], key, fitted)
                  for name, (source, key) in METHODS.items()}

    pool = json.loads((art / "grpo_pool" / "pool.json").read_text(encoding="utf-8"))
    lines = ["# Phase 2: RL without and with a calibration reward", "",
             "## Training", "",
             (f"Pool: {pool['prompts']:,} training prompts, answered 8 times each by M1 at temperature 1: "
              f"{pool['strata']['mixed']:,} mixed (some answers right, some wrong) and {pool['strata']['settled']:,} "
              f"settled ({pool['settled_all_right']:,} all right, {pool['settled_all_wrong']:,} all wrong). "
              f"Planned steps: {pool['steps']}, of 4 mixed and 4 settled prompts with 8 samples each."), "",
             ("| Arm | Reward | Steps | Minutes | Groups with no correctness signal | Groups trained per step | "
              "Accuracy, first → last 25 steps | Brier, first → last 25 | Stated confidence when right | "
              "Stated confidence when wrong |"), "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    histories = {}
    for name, arm in ARMS.items():
        run = json.loads((art / arm / "run.json").read_text(encoding="utf-8"))
        with open(art / arm / "log.jsonl", encoding="utf-8") as f:
            history = [json.loads(line) for line in f]
        histories[name] = history
        first, last = history[:25], history[-25:]

        def change(key, first=first, last=last):
            return f"{num(mean(h[key] for h in first))} → {num(mean(h[key] for h in last))}"

        lines.append(f"| {name} | {run['reward']} | {len(history)} | {run['elapsed_seconds'] / 60:.0f} | "
                     f"{pct(run['no_correctness_signal'])} | {mean(h['groups_trained'] for h in history):.1f} | "
                     f"{change('accuracy')} | {change('brier')} | {change('confidence_right')} | "
                     f"{change('confidence_wrong')} |")

    lines += ["", "## Primary comparison (fixed before any training)", "",
              ("M4 against M2-logit on the held-out test, redacted text, clean items, both attributes pooled. M4 "
               "counts as better if the 95% interval of the Brier difference lies below zero and the lower bound of "
               "the accuracy difference is above −1 point.")]
    confidences, correct, operating = {}, {}, {}
    intervals = {}
    for split in SPLITS:
        cal, test, others = {}, {}, {m: {} for m in ("M0", "M3", "M4")}
        for attribute in ("neckline", "sleeve_length"):
            cal[attribute] = arrays([r for r in runs["m1"]["cal_fit"]
                                     if r["condition"] == "redacted" and r["attribute"] == attribute])
            test_records = [r for r in runs["m1"][split]
                            if r["condition"] == "redacted" and r["clean"] and r["attribute"] == attribute]
            test[attribute] = arrays(test_records)
            for name in others:
                by_id = {r["example_id"]: r for r in runs[METHODS[name][0]][split]}
                others[name][attribute] = arrays([by_id[r["example_id"]] for r in test_records])
        intervals[split] = bootstrap(cal, test, others, args.resamples)
    brier, accuracy = intervals["heldout_test"][PRIMARY_BRIER], intervals["heldout_test"][PRIMARY_ACCURACY]
    brier_met, accuracy_met = brier[2] < 0, accuracy[1] > -0.01
    lines += ["", "| Difference | Estimate | 95% interval | Criterion | Met |", "|---|---:|---|---|---|",
              (f"| {PRIMARY_BRIER} | {brier[0]:+.4f} | {brier[1]:+.4f} to {brier[2]:+.4f} | interval below 0 | "
              f"{'yes' if brier_met else 'no'} |"),
              (f"| {PRIMARY_ACCURACY} | {accuracy[0]:+.4f} | {accuracy[1]:+.4f} to {accuracy[2]:+.4f} | "
              f"lower bound above −0.01 | {'yes' if accuracy_met else 'no'} |"), "",
              (f"**Verdict: M4 {'counts' if brier_met and accuracy_met else 'does not count'} as better than "
               "M2-logit.**")]

    lines += ["", "## Other comparisons, redacted text, clean items", "",
              (f"Paired bootstrap over product families ({args.resamples:,} draws); every draw also refits M2's "
               "Platt map and M2-logit's temperature on a resample of the calibration families. Negative Brier "
               "differences favour the first method."), "",
              "| Split | Difference | Estimate | 95% interval | Undefined draws |", "|---|---|---:|---|---:|"]
    for split in SPLITS:
        for key, (estimate, low, high, undefined) in intervals[split].items():
            lines.append(f"| {SPLITS[split]} | {key} | {estimate:+.4f} | {low:+.4f} to {high:+.4f} | {undefined} |")

    for subset in ("clean", "all labeled"):
        lines += ["", f"## Results, {subset} items", "",
                  "Pooled over both attributes; each attribute and text condition uses its own maps and threshold.", "",
                  ("| Split | Text | Method | n | Accuracy | Invalid | Brier, invalid = 1 | ECE | AUROC | "
                   "Risk–coverage area | Coverage at threshold | Error at threshold |"),
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for split in SPLITS:
            for condition in ("redacted", "full"):
                for name, (source, key) in METHODS.items():
                    records = [r for r in runs[source][split] if r["condition"] == condition
                               and (subset == "all labeled" or r["clean"])]
                    q = methods.confidence(records, key, fitted)
                    s = evaluate(records, q, thresholds[name])
                    if subset == "clean" and condition == "redacted":
                        confidences[(split, name)] = q
                        correct[(split, name)] = [r["correct"] for r in records]
                        operating[(split, name)] = (s["coverage"], s["risk"])
                    lines.append(f"| {SPLITS[split]} | {condition} | {name} | {s['n']:,} | {pct(s['accuracy'])} | "
                                 f"{pct(s['invalid'])} | {num(s['brier'])} | {num(s['ece'])} | {num(s['auroc'])} | "
                                 f"{num(s['aurc'])} | {pct(s['coverage'])} | {pct(s['risk'])} |")

    lines += ["", "## AUROC within each attribute, redacted text, clean items", "",
              ("Pooled AUROC also compares answers across the two attributes, so it partly rewards knowing which "
               "attribute is easier; within one attribute it measures only which answers of the same kind are "
               "riskier."), "",
              "| Split | Attribute | n | " + " | ".join(METHODS) + " |", "|---|---|---:|" + "---:|" * len(METHODS)]
    for split in SPLITS:
        for attribute in ("neckline", "sleeve_length"):
            cells = []
            for name, (source, key) in METHODS.items():
                records = [r for r in runs[source][split]
                           if r["condition"] == "redacted" and r["clean"] and r["attribute"] == attribute]
                cells.append(num(metrics.auroc(methods.confidence(records, key, fitted),
                                               [r["correct"] for r in records])))
            lines.append(f"| {SPLITS[split]} | {attribute} | {len(records):,} | " + " | ".join(cells) + " |")

    lines += ["", "## Evidence sensitivity, clean items", "",
              ("From full to redacted text on the same items: the fall in mean confidence beside the fall in accuracy, "
               "both on the items valid in both conditions."), "",
              "| Split | Method | n | Confidence drop | Accuracy drop | Invalid, full | Invalid, redacted |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for name, (source, key) in METHODS.items():
            scored = {}
            for condition in ("full", "redacted"):
                records = [r for r in runs[source][split] if r["condition"] == condition and r["clean"]]
                scored[condition] = [dict(r, confidence=q)
                                     for r, q in zip(records, methods.confidence(records, key, fitted))]
            s = evaluation.evidence_sensitivity(scored["full"], scored["redacted"])
            lines.append(f"| {SPLITS[split]} | {name} | {s['n']:,} | {s['confidence_drop']:+.3f} | "
                         f"{s['accuracy_drop']:+.3f} | {pct(s['invalid_full'])} | {pct(s['invalid_redacted'])} |")

    lines += ["", "## Stated confidence", "", "Across both test splits and both text conditions:", "",
              "| Model | Distinct values | Most common | Below 0.9 |", "|---|---:|---|---:|"]
    for name, source in (("M1", "m1"), ("M3", "m3"), ("M4", "m4")):
        stated = Counter(r["confidence"] for s in SPLITS for r in runs[source][s] if r["confidence"] is not None)
        total = sum(stated.values())
        common = ", ".join(f"{v:g} ({n / total:.1%})" for v, n in stated.most_common(4))
        below = sum(n for v, n in stated.items() if v < 0.9) / total
        lines.append(f"| {name} | {len(stated)} | {common} | {pct(below)} |")

    lines += ["", "## The Brier score, decomposed (exploratory)", "",
              ("Held-out test, redacted text, clean items. Brier = calibration − resolution + uncertainty (Murphy). "
               "Calibration is the gap between a group's confidence and its hit rate; resolution is how far the "
               "groups' hit rates spread from the overall one; uncertainty is a(1 − a) for accuracy a, so it depends "
               "only on the answers. Written confidences are grouped by value, which makes the decomposition exact; "
               "letter probabilities, which are continuous, are grouped into 50 equal bins."), "",
              "| Method | Brier | Calibration | Resolution | Uncertainty | Groups |", "|---|---:|---:|---:|---:|---:|"]
    for name, (source, key) in METHODS.items():
        records = [r for r in runs[source]["heldout_test"] if r["condition"] == "redacted" and r["clean"]]
        q = methods.confidence(records, key, fitted)
        c = [r["correct"] for r in records]
        part = decompose(q, c, (lambda v: min(int(v * 50), 49)) if key == "letter" else (lambda v: v))
        lines.append(f"| {name} | {metrics.brier(q, c, invalid_loss=1.0):.4f} | {part['calibration']:.4f} | "
                     f"{part['resolution']:.4f} | {part['uncertainty']:.4f} | {part['groups']} |")

    lines += ["", "## How often each written value is right, held-out test, redacted text, clean items", "",
              "Values written fewer than 20 times are left out.", "",
              "| Model | Written confidence | Answers | Right |", "|---|---:|---:|---:|"]
    for name, source in (("M1", "m1"), ("M3", "m3"), ("M4", "m4")):
        by_value = defaultdict(list)
        for r in runs[source]["heldout_test"]:
            if r["condition"] == "redacted" and r["clean"] and r["error"] is None:
                by_value[r["confidence"]].append(r["correct"])
        for value, outcomes in sorted(by_value.items()):
            if len(outcomes) >= 20:
                lines.append(f"| {name} | {value:g} | {len(outcomes):,} | {pct(sum(outcomes) / len(outcomes))} |")

    lines += ["", "## Does the written confidence follow the model's own letter probability? (exploratory)", "",
              ("Rank correlation, on the held-out test (redacted text, clean items, valid outputs), between the "
               "confidence a model writes and the probability it gives the letter it chose. 1 would mean the written "
               "number orders answers exactly as the model's internal probability does. Not part of the "
               "pre-registered comparisons."), "", "| Model | n | Rank correlation |", "|---|---:|---:|"]
    for name, source in (("M1", "m1"), ("M3", "m3"), ("M4", "m4")):
        pairs = [(r["confidence"], evaluation.letter_confidence(r)) for r in runs[source]["heldout_test"]
                 if r["condition"] == "redacted" and r["clean"] and r["error"] is None]
        pairs = [(q, p) for q, p in pairs if p is not None]
        written, letter = zip(*pairs)
        lines.append(f"| {name} | {len(pairs):,} | {num(rank_correlation(written, letter))} |")

    lines += ["", "## Held-out test by category, redacted text, clean items (exploratory)", "",
              ("| Category | n | Accuracy, M1 | Accuracy, M3 | Accuracy, M4 | Brier, M2-logit | Brier, M3 | "
               "Brier, M4 |"), "|---|---:|---:|---:|---:|---:|---:|---:|"]
    categories = Counter(r["category"] for r in runs["m1"]["heldout_test"]
                         if r["condition"] == "redacted" and r["clean"])
    for category, n in categories.most_common():
        cells = []
        for name in ("M1", "M3", "M4"):
            records = [r for r in runs[METHODS[name][0]]["heldout_test"]
                       if r["condition"] == "redacted" and r["clean"] and r["category"] == category]
            cells.append(pct(float(np.mean([r["correct"] for r in records]))))
        for name in ("M2-logit", "M3", "M4"):
            source, key = METHODS[name]
            records = [r for r in runs[source]["heldout_test"]
                       if r["condition"] == "redacted" and r["clean"] and r["category"] == category]
            cells.append(num(metrics.brier(methods.confidence(records, key, fitted), [r["correct"] for r in records],
                                           invalid_loss=1.0)))
        lines.append(f"| {category} | {n:,} | " + " | ".join(cells) + " |")

    lines += ["", "## Thresholds", "",
              ("Each method's frozen auto-fill threshold, chosen on its threshold-half predictions (infinity: accept "
              "nothing)."), "", "| Attribute | Text | " + " | ".join(METHODS) + " |",
              "|---|---|" + "---:|" * len(METHODS)]
    for key in sorted(thresholds["M1"]):
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(f"{thresholds[m][key]:.4f}" for m in METHODS) + " |")

    args.figures.mkdir(parents=True, exist_ok=True)
    reliability_figure(confidences, correct, args.figures / "phase2_reliability.png")
    risk_coverage_figure(confidences, correct, operating, args.figures / "phase2_risk_coverage.png")
    training_figure(histories, args.figures / "phase2_training.png")
    lines += ["", "## Figures", "", "![Training](figures/phase2_training.png)", "",
              "![Reliability](figures/phase2_reliability.png)", "",
              "![Risk–coverage](figures/phase2_risk_coverage.png)", "",
              ("Risk–coverage: lines rank both attributes together, ties in a random order. Dots are the outcome of "
               "each method's frozen thresholds. M0, M1 and M2 are in reports/phase1.md and the tables above.")]

    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
