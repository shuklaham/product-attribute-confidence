"""Phase-3 report: GRPO recipe v2 over three seeds, M5, and the photo swap.

Every method keeps its own auto-fill threshold, chosen on its threshold-half predictions; M2's
Platt map and M2-logit's temperature are fitted on M1's map-fitting half, as before. Nothing is
tuned on test.

The primary comparison was fixed before any Milestone 5 run (docs/plan.md, decision log
2026-10-28): the mean over seeds 0–2 of the M4 v2 minus M2-logit Brier difference on the held-out
test, redacted text, clean items, both attributes pooled. Its 95% interval comes from a paired
family bootstrap that resamples the test families once per draw for all seeds and refits M2's map
and M2-logit's temperature on a resample of the calibration families. M4 counts as better if the
interval lies below zero and the lower bound of the mean accuracy difference is above −1 point.

Usage:
    uv run python -I scripts/report_phase3.py [--artifacts artifacts] [--swaps data/processed/v1/swap_heldout.jsonl]
        [--out reports/phase3.md] [--figures reports/figures] [--resamples 10000]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pac import methods, metrics  # noqa: E402
from pac.calibration import fit_platt, fit_temperature  # noqa: E402
from pac.reporting import arrays, evaluate, family_index, load_run, mapped, num, pct  # noqa: E402

SEEDS = (0, 1, 2)
SPLITS = {"test": "Seen test", "heldout_test": "Held-out test"}
# Method -> (run name prefix, confidence source). The RL arms of Milestone 4 are recipe v1.
METHODS = {"M0": ("m0", "stated"), "M1": ("m1", "stated"), "M2": ("m1", "platt"), "M2-logit": ("m1", "letter"),
           "M3 v1": ("m3", "stated"), "M4 v1": ("m4", "stated")}
METHODS |= {f"M3 v2 s{s}": (f"m3_v2_s{s}", "stated") for s in SEEDS}
METHODS |= {f"M4 v2 s{s}": (f"m4_v2_s{s}", "stated") for s in SEEDS}
METHODS |= {"M5": ("m5", "stated")}
STATED = [m for m, (_, key) in METHODS.items() if key == "stated" and m != "M1"]
# Reference palette, light mode, fixed categorical order (see report_phase2.py); M5 takes slot 7.
COLORS = {"M2-logit": "#1baf7a", "M3": "#e87ba4", "M4": "#008300", "M5": "#6250d6", "v1": "#52514e"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PRIMARY_BRIER = "Brier, M4 v2 minus M2-logit, mean of seeds"
PRIMARY_ACCURACY = "Accuracy, M4 v2 minus M2-logit, mean of seeds"


def mean(values) -> float:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def bootstrap(cal: dict, test: dict, others: dict, cal5: dict, test5: dict, resamples: int, seed: int = 0) -> dict:
    """Paired family bootstrap on one test set; every map is refitted every draw.

    cal and test map attribute -> arrays of M1's records; others maps every stated method to arrays
    of its records for the same test items, in the same order. cal5 and test5 hold M5's records for
    the same calibration and test items, for the exploratory M5 + Platt and M5-logit, whose maps are
    fitted on M5's own calibration predictions.
    """
    cal_index, n_cal = family_index(cal)
    test_index, n_test = family_index(test)

    def statistics(cal_counts, test_counts):
        q = defaultdict(list)
        c = defaultdict(list)
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
            fit5 = cal5[k]
            platt5 = fit_platt(fit5["stated"][cal_rows], fit5["correct"][cal_rows])
            with_letters5 = cal_rows[fit5["has_letters"][cal_rows]]
            t5 = fit_temperature(fit5["letters"][with_letters5], fit5["answer"][with_letters5], iterations=40).t
            m5_platt, m5_logit = mapped(test5[k], test_rows, platt5, t5)
            for name, values in (("M5 + Platt", m5_platt), ("M5-logit", m5_logit)):
                q[name].append(values)
                c[name].append(test5[k]["correct"][test_rows])
            for name in others:
                q[name].append(others[name][k]["stated"][test_rows])
                c[name].append(others[name][k]["correct"][test_rows])
        q = {m: np.concatenate(v) for m, v in q.items()}
        c = {m: np.concatenate(v) for m, v in c.items()}
        brier = {m: metrics.brier(q[m], c[m], invalid_loss=1.0) for m in q}
        accuracy = {m: float(c[m].mean()) for m in q}
        auroc = {m: metrics.auroc(q[m], c[m]) for m in q if m.startswith(("M4 v2", "M3 v2", "M5", "M2-logit"))}
        m4 = [f"M4 v2 s{s}" for s in SEEDS]
        m3 = [f"M3 v2 s{s}" for s in SEEDS]
        out = {
            PRIMARY_BRIER: mean([brier[m] - brier["M2-logit"] for m in m4]),
            PRIMARY_ACCURACY: mean([accuracy[m] - accuracy["M2-logit"] for m in m4]),
            "Brier, M4 v2 minus M3 v2, mean of seeds": mean([brier[a] - brier[b] for a, b in zip(m4, m3)]),
            "AUROC, M4 v2 minus M2-logit, mean of seeds": mean([auroc[m] - auroc["M2-logit"] for m in m4]),
            "Brier, M5 minus M2-logit": brier["M5"] - brier["M2-logit"],
            "Brier, M5 minus M4 v2 (mean of seeds)": brier["M5"] - mean([brier[m] for m in m4]),
            "AUROC, M5 minus M2-logit": auroc["M5"] - auroc["M2-logit"],
            "Accuracy, M5 minus M1": accuracy["M5"] - accuracy["M1"],
        }
        for s in SEEDS:
            out[f"Brier, M4 v2 s{s} minus M2-logit"] = brier[f"M4 v2 s{s}"] - brier["M2-logit"]
        out |= {"Brier, M5 + Platt minus M2-logit (exploratory)": brier["M5 + Platt"] - brier["M2-logit"],
                "Brier, M5-logit minus M2-logit (exploratory)": brier["M5-logit"] - brier["M2-logit"],
                "AUROC, M5-logit minus M2-logit (exploratory)": metrics.auroc(q["M5-logit"], c["M5-logit"])
                - auroc["M2-logit"]}
        return out

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
        low, high = np.nanquantile(values, [0.025, 0.975])
        out[key] = (point[key], float(low), float(high), int(np.isnan(values).sum()))
    return out


def decompose(q: list, correct: list, group) -> dict:
    """Murphy's decomposition of the Brier score, exact when each group holds one value.

    Answers without a confidence are left out; the Brier column counts them separately.
    """
    pairs = [(v, c) for v, c in zip(q, correct) if v is not None]
    correct = [c for _, c in pairs]
    groups = defaultdict(list)
    for v, c in pairs:
        groups[group(v)].append((v, c))
    n, a = len(correct), sum(correct) / len(correct)
    calibration = sum(len(g) / n * (np.mean([v for v, _ in g]) - np.mean([c for _, c in g])) ** 2
                      for g in groups.values())
    resolution = sum(len(g) / n * (np.mean([c for _, c in g]) - a) ** 2 for g in groups.values())
    return {"calibration": float(calibration), "resolution": float(resolution), "uncertainty": a * (1 - a),
            "groups": len(groups)}


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


def rolling(values: list, window: int = 10) -> np.ndarray:
    x = np.array([np.nan if v is None else v for v in values], dtype=float)
    return np.array([np.nanmean(x[max(0, i - window + 1):i + 1]) for i in range(len(x))])


def signal_figure(histories: dict, path: Path) -> None:
    """The share of groups with no correctness signal per step, recipe v1 against v2, both arms."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, arm in zip(axes, ("M4", "M3")):
        if f"{arm} v1" in histories:
            h = histories[f"{arm} v1"]
            ax.plot([x["step"] for x in h], rolling([x["no_correctness_signal"] for x in h]), color=COLORS["v1"],
                    linewidth=1.6, linestyle="--", label="v1 (seed 0)")
        for s in SEEDS:
            h = histories.get(f"{arm} v2 s{s}")
            if h:
                ax.plot([x["step"] for x in h], rolling([x["no_correctness_signal"] for x in h]), color=COLORS[arm],
                        linewidth=1.4, alpha=0.55 + 0.15 * s, label=f"v2 seed {s}")
        ax.axhline(0.8, linestyle=":", color=INK_2, linewidth=0.8)
        ax.text(455, 0.795, "stop line", color=INK_2, fontsize=8, va="top", ha="right")
        style(ax, arm)
        ax.set_xlabel("Step (trailing mean over 10 steps)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[0].set_ylabel("Groups with no correctness signal", color=INK_2, fontsize=9)
    fig.suptitle("Learning signal during GRPO: fixed strata (v1) against re-scored strata (v2)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def reliability_figure(confidences: dict, correct: dict, shown: list, path: Path) -> None:
    fig, axes = plt.subplots(1, len(shown), figsize=(3.3 * len(shown), 3.6), sharey=True, facecolor=SURFACE)
    for ax, method in zip(axes, shown):
        q, c = confidences[method], correct[method]
        bins = [b for b in metrics.reliability(q, c) if b.count]
        color = COLORS["M2-logit"] if method == "M2-logit" else COLORS[method.split()[0]]
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=0.8)
        ax.scatter([b.mean_confidence for b in bins], [b.accuracy for b in bins], color=color,
                   s=[12 + 300 * b.count / len(q) for b in bins], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        style(ax, f"{method}\nECE {metrics.ece(q, c):.3f}")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Confidence", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Accuracy", color=INK_2, fontsize=9)
    fig.suptitle("Reliability, held-out test, redacted text, clean items", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--swaps", type=Path, default=Path("data/processed/v1/swap_heldout.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/phase3.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    art = args.artifacts

    sources = {source for source, _ in METHODS.values()}
    runs = {source: {s: load_run(art, f"{source}_{s}") for s in ("cal_threshold", *SPLITS)} for source in sources}
    runs["m1"]["cal_fit"] = load_run(art, "m1_cal_fit")
    runs["m5"]["cal_fit"] = load_run(art, "m5_cal_fit")
    fitted = methods.fit(runs["m1"]["cal_fit"])
    fitted5 = methods.fit(runs["m5"]["cal_fit"])  # the exploratory M5 + Platt and M5-logit
    extra = {"M5 + Platt (exploratory)": ("platt", methods.thresholds(runs["m5"]["cal_threshold"], "platt", fitted5)),
             "M5-logit (exploratory)": ("letter", methods.thresholds(runs["m5"]["cal_threshold"], "letter", fitted5))}
    thresholds = {name: methods.thresholds(runs[source]["cal_threshold"], key, fitted)
                  for name, (source, key) in METHODS.items()}

    lines = ["# Phase 3: GRPO recipe v2 over three seeds, M5, and the photo swap", "", "## Training", "",
             ("Recipe v2 re-scores every prompt with the current policy every 50 steps; a prompt is mixed if 8 samples "
              "would hold both right and wrong answers with probability at least 0.2. v1 rows are Milestone 4's runs, "
              "whose strata were fixed at the start."), "",
             ("| Arm | Steps | Minutes | Groups with no correctness signal: all, first third, last third | "
              "Groups trained per step | Mixed prompts at the first and last re-scoring | Accuracy, first → last 25 | "
              "Brier, first → last 25 |"), "|---|---:|---:|---|---:|---|---|---|"]
    histories = {}
    for arm in ("M4", "M3"):
        for label, run_dir in [(f"{arm} v1", arm.lower())] + [(f"{arm} v2 s{s}", f"{arm.lower()}_v2_s{s}")
                                                               for s in SEEDS]:
            path = art / run_dir
            if not (path / "log.jsonl").exists():
                continue
            run = json.loads((path / "run.json").read_text(encoding="utf-8"))
            with open(path / "log.jsonl", encoding="utf-8") as f:
                h = [json.loads(line) for line in f]
            histories[label] = h
            third = len(h) // 3
            signal = [x["no_correctness_signal"] for x in h]
            windows = run["pool"].get("windows") if isinstance(run.get("pool"), dict) else None
            mixed = f"{windows[0]['mixed']:,} → {windows[-1]['mixed']:,}" if windows else "–"

            def change(key, h=h):
                return f"{num(mean(x[key] for x in h[:25]))} → {num(mean(x[key] for x in h[-25:]))}"

            lines.append(f"| {label} | {len(h)} | {run['elapsed_seconds'] / 60:.0f} | {pct(mean(signal))}, "
                         f"{pct(mean(signal[:third]))}, {pct(mean(signal[-third:]))} | "
                         f"{mean(x['groups_trained'] for x in h):.1f} | {mixed} | {change('accuracy')} | "
                         f"{change('brier')} |")

    lines += ["", "## Primary comparison (fixed before any Milestone 5 run)", "",
              ("The mean over seeds 0–2 of the M4 v2 minus M2-logit difference, held-out test, redacted text, clean "
               "items, both attributes pooled. M4 counts as better if the 95% interval of the Brier difference lies "
               "below zero and the lower bound of the accuracy difference is above −1 point.")]
    confidences, correct = {}, {}
    intervals = {}
    for split in SPLITS:
        cal, test, others, cal5, test5 = {}, {}, {m: {} for m in STATED}, {}, {}
        m5_cal = {r["example_id"]: r for r in runs["m5"]["cal_fit"]}
        m5_test = {r["example_id"]: r for r in runs["m5"][split]}
        for attribute in ("neckline", "sleeve_length"):
            cal_records = [r for r in runs["m1"]["cal_fit"]
                           if r["condition"] == "redacted" and r["attribute"] == attribute]
            cal[attribute] = arrays(cal_records)
            cal5[attribute] = arrays([m5_cal[r["example_id"]] for r in cal_records])
            test_records = [r for r in runs["m1"][split]
                            if r["condition"] == "redacted" and r["clean"] and r["attribute"] == attribute]
            test[attribute] = arrays(test_records)
            test5[attribute] = arrays([m5_test[r["example_id"]] for r in test_records])
            for name in others:
                by_id = {r["example_id"]: r for r in runs[METHODS[name][0]][split]}
                others[name][attribute] = arrays([by_id[r["example_id"]] for r in test_records])
        intervals[split] = bootstrap(cal, test, others, cal5, test5, args.resamples)
    brier, accuracy = intervals["heldout_test"][PRIMARY_BRIER], intervals["heldout_test"][PRIMARY_ACCURACY]
    brier_met, accuracy_met = brier[2] < 0, accuracy[1] > -0.01
    lines += ["", "| Difference | Estimate | 95% interval | Criterion | Met |", "|---|---:|---|---|---|",
              (f"| {PRIMARY_BRIER} | {brier[0]:+.4f} | {brier[1]:+.4f} to {brier[2]:+.4f} | interval below 0 | "
               f"{'yes' if brier_met else 'no'} |"),
              (f"| {PRIMARY_ACCURACY} | {accuracy[0]:+.4f} | {accuracy[1]:+.4f} to {accuracy[2]:+.4f} | "
               f"lower bound above −0.01 | {'yes' if accuracy_met else 'no'} |"), "",
              (f"**Verdict: M4 v2 {'counts' if brier_met and accuracy_met else 'does not count'} as better than "
               "M2-logit.**")]

    lines += ["", "## Other comparisons, redacted text, clean items", "",
              (f"Paired family bootstrap ({args.resamples:,} draws), the same draws for every seed; every draw also "
               "refits M2's map and M2-logit's temperature. Negative Brier differences favour the first method."), "",
              "| Split | Difference | Estimate | 95% interval | Undefined draws |", "|---|---|---:|---|---:|"]
    for split in SPLITS:
        for key, (estimate, low, high, undefined) in intervals[split].items():
            lines.append(f"| {SPLITS[split]} | {key} | {estimate:+.4f} | {low:+.4f} to {high:+.4f} | {undefined} |")

    lines += ["", "## Results, held-out and seen test, redacted text, clean items", "",
              ("Pooled over both attributes; each attribute and text condition uses its own maps and threshold. The "
               "two exploratory M5 rows were added after the results were seen, and are not part of the pre-registered "
               "comparisons: a Platt map on M5's written confidence, and M5's own letter probabilities with a "
               "temperature, both fitted on M5's map-fitting predictions."), "",
              ("| Split | Method | Accuracy | Brier, invalid = 1 | ECE | AUROC | Risk–coverage area | "
               "Coverage at threshold | Error at threshold |"), "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for name, (source, key) in METHODS.items():
            records = [r for r in runs[source][split] if r["condition"] == "redacted" and r["clean"]]
            q = methods.confidence(records, key, fitted)
            s = evaluate(records, q, thresholds[name])
            if split == "heldout_test":
                confidences[name], correct[name] = q, [r["correct"] for r in records]
            lines.append(f"| {SPLITS[split]} | {name} | {pct(s['accuracy'])} | {num(s['brier'])} | {num(s['ece'])} | "
                         f"{num(s['auroc'])} | {num(s['aurc'])} | {pct(s['coverage'])} | {pct(s['risk'])} |")
        for name, (key, chosen) in extra.items():
            records = [r for r in runs["m5"][split] if r["condition"] == "redacted" and r["clean"]]
            q = methods.confidence(records, key, fitted5)
            s = evaluate(records, q, chosen)
            if split == "heldout_test":
                confidences[name], correct[name] = q, [r["correct"] for r in records]
            lines.append(f"| {SPLITS[split]} | {name} | {pct(s['accuracy'])} | {num(s['brier'])} | {num(s['ece'])} | "
                         f"{num(s['auroc'])} | {num(s['aurc'])} | {pct(s['coverage'])} | {pct(s['risk'])} |")

    lines += ["", "## The Brier score, decomposed (exploratory)", "",
              ("Held-out test, redacted text, clean items. Brier = calibration − resolution + uncertainty; written "
               "confidences grouped by value (exact), letter probabilities in 50 equal bins."), "",
              "| Method | Brier | Calibration | Resolution | Uncertainty | Groups |", "|---|---:|---:|---:|---:|---:|"]
    for name, key in [(m, k) for m, (_, k) in METHODS.items()] + [(m, k) for m, (k, _) in extra.items()]:
        part = decompose(confidences[name], correct[name],
                         (lambda v: min(int(v * 50), 49)) if key == "letter" else (lambda v: v))
        lines.append(f"| {name} | {metrics.brier(confidences[name], correct[name], invalid_loss=1.0):.4f} | "
                     f"{part['calibration']:.4f} | {part['resolution']:.4f} | {part['uncertainty']:.4f} | "
                     f"{part['groups']} |")

    lines += ["", "## How often each written value is right, held-out test, redacted text, clean items", "",
              "Values written fewer than 20 times are left out.", "",
              "| Model | Written confidence | Answers | Right |", "|---|---:|---:|---:|"]
    for name in ["M4 v2 s0", "M5"]:
        by_value = defaultdict(list)
        for q, c in zip(confidences[name], correct[name]):
            if q is not None:
                by_value[q].append(c)
        for value, outcomes in sorted(by_value.items()):
            if len(outcomes) >= 20:
                lines.append(f"| {name} | {value:g} | {len(outcomes):,} | {pct(sum(outcomes) / len(outcomes))} |")

    with open(args.swaps, encoding="utf-8") as f:
        swaps = {e["example_id"]: e for e in (json.loads(line) for line in f)}
    lines += ["", "## Photo swap, held-out test", "",
              ("Each item's text with another product's photo whose label differs (same category, another family). "
               "With full text the two disagree; with redacted text the photo is the only evidence of the label. "
               "Confidence is compared with the same items and condition unswapped."), "",
              ("| Model | Text | n | Answers the text's label | Answers the photo's label | Other or invalid | "
               "Mean confidence, swapped | Mean confidence, unswapped |"),
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, (source, key) in METHODS.items():
        if name in ("M2", "M3 v1", "M4 v1") or not (art / f"{source}_swap").exists():
            continue
        swapped = load_run(art, f"{source}_swap")
        q_swapped = methods.confidence(swapped, key, fitted)
        normal = {r["example_id"]: r for r in runs[source]["heldout_test"]}
        for condition in ("full", "redacted"):
            rows = [(r, q) for r, q in zip(swapped, q_swapped) if r["condition"] == condition]
            base = [normal[r["example_id"].removesuffix(":swap")] for r, _ in rows]
            q_base = methods.confidence(base, key, fitted)
            text = mean([r["choice"] == r["answer"] for r, _ in rows])
            photo = mean([r["choice"] == swaps[r["example_id"]]["photo_answer"] for r, _ in rows])
            lines.append(f"| {name} | {condition} | {len(rows):,} | {pct(text)} | {pct(photo)} | "
                         f"{pct(1 - text - photo)} | {num(mean(q for _, q in rows))} | {num(mean(q_base))} |")

    args.figures.mkdir(parents=True, exist_ok=True)
    signal_figure(histories, args.figures / "phase3_signal.png")
    reliability_figure(confidences, correct, ["M2-logit", "M4 v2 s0", "M4 v2 s1", "M4 v2 s2", "M5"],
                       args.figures / "phase3_reliability.png")
    lines += ["", "## Figures", "", "![Learning signal](figures/phase3_signal.png)", "",
              "![Reliability](figures/phase3_reliability.png)"]

    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
