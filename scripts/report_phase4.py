"""Phase-4 report: three seeds of GRPO recipe v1, the test of M5-logit, and the evidence runs.

Every method keeps its own auto-fill threshold, chosen on its threshold-half predictions. M2's
Platt map and M2-logit's temperature are fitted on M1's map-fitting half, and each M5 seed's
temperature on that seed's own. Nothing is tuned on test.

Fixed before any Milestone 6 run (docs/plan.md, decision log 2026-10-29), on the held-out test,
redacted text, clean items, both attributes pooled:
- primary: the mean over seeds 0–2 of the M4 v1 minus M2-logit Brier difference. M4 counts as
  better if its 95% interval lies below zero and the lower bound of the mean accuracy difference
  is above −1 point. Seed 0 is the Milestone 4 run.
- the test of M5-logit: the mean over seeds 1 and 2 of the M5-logit minus M2-logit Brier
  difference, with the same rule. Seed 0, which suggested the hypothesis, is reported but left out.
Intervals come from a paired family bootstrap that resamples the test families once per draw for
every model, and refits every map on a resample of the calibration families.

The evidence runs are descriptive: the photo-swap subset unswapped, with only the photo or only
the text, beside the same items with both.

Usage:
    uv run python -I scripts/report_phase4.py [--artifacts artifacts] [--swaps data/processed/v1/swap_heldout.jsonl]
        [--subset data/processed/v1/subset_heldout.jsonl] [--out reports/phase4.md] [--figures reports/figures]
        [--resamples 10000]
"""

import argparse
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pac import methods, metrics  # noqa: E402
from pac.calibration import fit_platt, fit_temperature  # noqa: E402
from pac.reporting import arrays, evaluate, family_index, load_run, mapped, num, pct  # noqa: E402

SEEDS = (0, 1, 2)
TEST_SEEDS = (1, 2)  # the test of M5-logit leaves out seed 0, which suggested it
SPLITS = {"test": "Seen test", "heldout_test": "Held-out test"}
ATTRIBUTES = ("neckline", "sleeve_length")


def source(arm: str, seed: int) -> str:
    """One seed's run name. Seed 0 of M3 and M4 is the Milestone 4 run, and of M5 the Milestone 5 run."""
    if seed == 0:
        return arm
    return {"m3": f"m3_v1_s{seed}", "m4": f"m4_v1_s{seed}", "m5": f"m5_s{seed}"}[arm]


# Method -> (run name, confidence source, run whose map-fitting predictions fit its map).
METHODS = {"M0": ("m0", "stated", None), "M1": ("m1", "stated", None), "M2": ("m1", "platt", "m1"),
           "M2-logit": ("m1", "letter", "m1")}
METHODS |= {f"M4 v1 s{s}": (source("m4", s), "stated", None) for s in SEEDS}
METHODS |= {f"M3 v1 s{s}": (source("m3", s), "stated", None) for s in SEEDS}
METHODS |= {f"M5 s{s}": (source("m5", s), "stated", None) for s in SEEDS}
METHODS |= {f"M5-logit s{s}": (source("m5", s), "letter", source("m5", s)) for s in SEEDS}
# The evidence runs: the reference models of earlier milestones, and the modes each was run in.
EVIDENCE = {"M0": "m0", "M1": "m1", "M2-logit": "m1", "M4 v1 s0": "m4", "M5 s0": "m5", "M5-logit s0": "m5"}
MODES = [("photo and full text", "heldout_test", "full"), ("photo and redacted text", "heldout_test", "redacted"),
         ("photo only", "subset_photo_only", "redacted"), ("full text only", "subset_text_only", "full"),
         ("redacted text only", "subset_text_only", "redacted")]
PRIMARY_BRIER = "Brier, M4 v1 minus M2-logit, mean of seeds 0–2"
PRIMARY_ACCURACY = "Accuracy, M4 v1 minus M2-logit, mean of seeds 0–2"
TEST_BRIER = "Brier, M5-logit minus M2-logit, mean of seeds 1–2"
TEST_ACCURACY = "Accuracy, M5-logit minus M2-logit, mean of seeds 1–2"
# Reference palette, light mode, fixed categorical order (see report_phase2.py); M5-logit shares M5's hue.
COLORS = {"M0": "#2a78d6", "M1": "#eb6834", "M2-logit": "#1baf7a", "M3": "#e87ba4", "M4": "#008300", "M5": "#6250d6"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def mean(values) -> float:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else float("nan")


def color(method: str) -> str:
    return COLORS["M2-logit"] if method == "M2-logit" else COLORS[method.split()[0].removesuffix("-logit")]


def bootstrap(cal: dict, test: dict, resamples: int, seed: int = 0) -> tuple[dict, Counter]:
    """Paired family bootstrap on one test set; every map is refitted every draw.

    cal maps M1's run and each M5 seed's run to attribute -> arrays of its map-fitting records, and
    test maps every method's run to attribute -> arrays of its records; all runs hold the same items
    in the same order.

    A temperature refitted on a resample with almost no wrong answers runs to the edge of its search
    range. Such draws stay in the intervals, as the procedure was fixed, but are counted per run, and
    each interval is also given without them.
    """
    cal_index, n_cal = family_index(cal["m1"])
    test_index, n_test = family_index(test["m1"])
    at_edge = set()  # the runs whose temperature hit a search bound in the current draw

    def temperature(a, rows, run):
        rows = rows[a["has_letters"][rows]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t = fit_temperature(a["letters"][rows], a["answer"][rows], iterations=40).t
        if caught:
            at_edge.add(run)
        return t

    def statistics(cal_counts, test_counts):
        q, c = defaultdict(list), defaultdict(list)
        for k in ATTRIBUTES:
            cal_rows = np.repeat(np.arange(len(cal_index[k])), cal_counts[cal_index[k]])
            test_rows = np.repeat(np.arange(len(test_index[k])), test_counts[test_index[k]])
            fit = cal["m1"][k]
            platt = fit_platt(fit["stated"][cal_rows], fit["correct"][cal_rows])
            m2, m2_logit = mapped(test["m1"][k], test_rows, platt, temperature(fit, cal_rows, "m1"))
            fitted = {"M2": m2, "M2-logit": m2_logit}
            for s in SEEDS:
                run = source("m5", s)
                # Only the letter half is used: M5's letter probability with this seed's own temperature.
                t5 = temperature(cal[run][k], cal_rows, run)
                fitted[f"M5-logit s{s}"] = mapped(test[run][k], test_rows, platt, t5)[1]
            for name, (run, key, _) in METHODS.items():
                q[name].append(test[run][k]["stated"][test_rows] if key == "stated" else fitted[name])
                c[name].append(test[run][k]["correct"][test_rows])
        q = {m: np.concatenate(v) for m, v in q.items()}
        c = {m: np.concatenate(v) for m, v in c.items()}
        brier = {m: metrics.brier(q[m], c[m], invalid_loss=1.0) for m in q}
        accuracy = {m: float(c[m].mean()) for m in q}
        auroc = {m: metrics.auroc(q[m], c[m]) for m in q if m not in ("M0", "M1", "M2")}
        m4, m3, m5 = ([f"{arm} s{s}" for s in SEEDS] for arm in ("M4 v1", "M3 v1", "M5"))
        tested = [f"M5-logit s{s}" for s in TEST_SEEDS]
        out = {
            PRIMARY_BRIER: mean([brier[m] - brier["M2-logit"] for m in m4]),
            PRIMARY_ACCURACY: mean([accuracy[m] - accuracy["M2-logit"] for m in m4]),
            TEST_BRIER: mean([brier[m] - brier["M2-logit"] for m in tested]),
            TEST_ACCURACY: mean([accuracy[m] - accuracy["M2-logit"] for m in tested]),
            "Brier, M4 v1 minus M3 v1, mean of seeds": mean([brier[a] - brier[b] for a, b in zip(m4, m3)]),
            "Brier, M4 v1 minus M2, mean of seeds": mean([brier[m] - brier["M2"] for m in m4]),
            "AUROC, M4 v1 minus M2-logit, mean of seeds": mean([auroc[m] - auroc["M2-logit"] for m in m4]),
            "Accuracy, M4 v1 minus M3 v1, mean of seeds": mean([accuracy[a] - accuracy[b] for a, b in zip(m4, m3)]),
            "Accuracy, M3 v1 minus M1, mean of seeds": mean([accuracy[m] - accuracy["M1"] for m in m3]),
            "Brier, M5 minus M2-logit, mean of seeds": mean([brier[m] - brier["M2-logit"] for m in m5]),
            "AUROC, M5 minus M2-logit, mean of seeds": mean([auroc[m] - auroc["M2-logit"] for m in m5]),
            "AUROC, M5-logit minus M2-logit, mean of seeds 1–2": mean([auroc[m] - auroc["M2-logit"] for m in tested]),
        }
        out |= {f"Brier, M4 v1 s{s} minus M2-logit": brier[f"M4 v1 s{s}"] - brier["M2-logit"] for s in SEEDS}
        out |= {f"Brier, M5-logit s{s} minus M2-logit": brier[f"M5-logit s{s}"] - brier["M2-logit"] for s in SEEDS}
        return out

    point = statistics(np.ones(n_cal, dtype=int), np.ones(n_test, dtype=int))
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in point}
    edges, clean = Counter(), []
    for _ in range(resamples):
        at_edge.clear()
        values = statistics(np.bincount(rng.integers(0, n_cal, n_cal), minlength=n_cal),
                            np.bincount(rng.integers(0, n_test, n_test), minlength=n_test))
        for key, value in values.items():
            draws[key].append(value)
        edges.update(at_edge)
        clean.append(not at_edge)
    clean = np.array(clean)
    out = {}
    for key, values in draws.items():
        values = np.array(values)
        low, high = np.nanquantile(values, [0.025, 0.975])
        clean_low, clean_high = np.nanquantile(values[clean], [0.025, 0.975])
        out[key] = (point[key], float(low), float(high), int(np.isnan(values).sum()), float(clean_low),
                    float(clean_high))
    return out, edges


def verdict(intervals: dict, brier_key: str, accuracy_key: str) -> tuple[list[str], bool]:
    """The table rows of a pre-registered comparison and whether it counts as better."""
    brier, accuracy = intervals[brier_key][:3], intervals[accuracy_key][:3]
    brier_met, accuracy_met = brier[2] < 0, accuracy[1] > -0.01
    rows = ["| Difference | Estimate | 95% interval | Criterion | Met |", "|---|---:|---|---|---|",
            (f"| {brier_key} | {brier[0]:+.4f} | {brier[1]:+.4f} to {brier[2]:+.4f} | interval below 0 | "
             f"{'yes' if brier_met else 'no'} |"),
            (f"| {accuracy_key} | {accuracy[0]:+.4f} | {accuracy[1]:+.4f} to {accuracy[2]:+.4f} | "
             f"lower bound above −0.01 | {'yes' if accuracy_met else 'no'} |")]
    return rows, brier_met and accuracy_met


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
    """The share of groups with no correctness signal per step, for each seed of recipe v1."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    for ax, arm in zip(axes, ("M4", "M3")):
        for s in SEEDS:
            h = histories[f"{arm} v1 s{s}"]
            ax.plot([x["step"] for x in h], rolling([x["no_correctness_signal"] for x in h]), color=COLORS[arm],
                    linewidth=1.4, alpha=0.55 + 0.15 * s, label=f"seed {s}")
        ax.axhline(0.8, linestyle=":", color=INK_2, linewidth=0.8)
        ax.text(455, 0.795, "stop line", color=INK_2, fontsize=8, va="top", ha="right")
        style(ax, f"{arm} v1")
        ax.set_xlabel("Step (trailing mean over 10 steps)", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[0].set_ylabel("Groups with no correctness signal", color=INK_2, fontsize=9)
    fig.suptitle("Learning signal during GRPO recipe v1, three seeds", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def reliability_figure(confidences: dict, correct: dict, shown: list, path: Path) -> None:
    rows = math.ceil(len(shown) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.4 * rows + 0.4), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, method in zip(axes.flat, shown):
        q, c = confidences[method], correct[method]
        bins = [b for b in metrics.reliability(q, c) if b.count]
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=0.8)
        ax.scatter([b.mean_confidence for b in bins], [b.accuracy for b in bins], color=color(method),
                   s=[12 + 300 * b.count / len(q) for b in bins], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        style(ax, f"{method}, ECE {metrics.ece(q, c):.3f}")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Confidence", color=INK_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy", color=INK_2, fontsize=9)
    fig.suptitle("Reliability, held-out test, redacted text, clean items", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def evidence_figure(points: dict, path: Path) -> None:
    """Mean confidence against accuracy in each evidence mode; on the diagonal, confidence follows the evidence.

    Each panel is one model, in its colour; the marker shape is the evidence mode, named in one legend.
    """
    markers = dict(zip([mode for mode, _, _ in MODES], ("o", "s", "^", "D", "v")))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4), sharex=True, sharey=True, facecolor=SURFACE)
    values = [v for model in points.values() for mode in model.values() for v in mode]
    low = max(0.0, math.floor(min(values) * 10) / 10 - 0.05)
    for ax, (model, modes) in zip(axes.flat, points.items()):
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=0.8)
        for mode, (accuracy, confidence) in modes.items():
            ax.scatter([accuracy], [confidence], color=color(model), marker=markers[mode], s=64, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
        style(ax, model)
        ax.set_xlim(low, 1.02)
        ax.set_ylim(low, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("Accuracy", color=INK_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean confidence", color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], linestyle="", marker=marker, markersize=7, color=INK_2, label=mode)
               for mode, marker in markers.items()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=5, frameon=False, fontsize=8,
               labelcolor=INK)
    fig.suptitle("Confidence against accuracy as evidence is removed, 1,000 held-out items",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--swaps", type=Path, default=Path("data/processed/v1/swap_heldout.jsonl"))
    parser.add_argument("--subset", type=Path, default=Path("data/processed/v1/subset_heldout.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/phase4.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    art = args.artifacts

    runs = {run: {s: load_run(art, f"{run}_{s}") for s in ("cal_threshold", *SPLITS)}
            for run in {run for run, _, _ in METHODS.values()}}
    fitted = {}
    for run in {fit for _, _, fit in METHODS.values() if fit}:
        runs[run]["cal_fit"] = load_run(art, f"{run}_cal_fit")
        fitted[run] = methods.fit(runs[run]["cal_fit"])
    thresholds = {name: methods.thresholds(runs[run]["cal_threshold"], key, fitted.get(fit))
                  for name, (run, key, fit) in METHODS.items()}

    lines = ["# Phase 4: three seeds of GRPO recipe v1, the test of M5-logit, and the evidence runs", "",
             "## Training", "",
             ("Recipe v1: the strata come from M1's sampling pass and stay fixed. Seed 0 of M3 and M4 is the Milestone "
              "4 run, and seed 0 of M5 the Milestone 5 run."), "",
             ("| Arm | Steps | Minutes | Groups with no correctness signal: all, first third, last third | "
              "Groups trained per step | Accuracy, first → last 25 | Brier, first → last 25 |"),
             "|---|---:|---:|---|---:|---|---|"]
    histories = {}
    for arm in ("M4", "M3"):
        for s in SEEDS:
            path = art / source(arm.lower(), s)
            run = json.loads((path / "run.json").read_text(encoding="utf-8"))
            with open(path / "log.jsonl", encoding="utf-8") as f:
                h = [json.loads(line) for line in f]
            histories[f"{arm} v1 s{s}"] = h
            third = len(h) // 3
            signal = [x["no_correctness_signal"] for x in h]

            def change(key, h=h):
                return f"{num(mean(x[key] for x in h[:25]))} → {num(mean(x[key] for x in h[-25:]))}"

            lines.append(f"| {arm} v1 s{s} | {len(h)} | {run['elapsed_seconds'] / 60:.0f} | {pct(mean(signal))}, "
                         f"{pct(mean(signal[:third]))}, {pct(mean(signal[-third:]))} | "
                         f"{mean(x['groups_trained'] for x in h):.1f} | {change('accuracy')} | {change('brier')} |")
    lines += ["", "| M5 seed | Steps | Minutes | Final training loss |", "|---|---:|---:|---:|"]
    for s in SEEDS:
        run = json.loads((art / source("m5", s) / "run.json").read_text(encoding="utf-8"))
        lines.append(f"| {s} | {run['steps']} | {run['elapsed_seconds'] / 60:.0f} | {run['final_train_loss']:.4f} |")

    intervals, edges = {}, {}
    for split in SPLITS:
        cal = {run: {} for run in fitted}
        test = {run: {} for run in runs}
        by_id = {run: {r["example_id"]: r for r in runs[run][split]} for run in runs}
        cal_by_id = {run: {r["example_id"]: r for r in runs[run]["cal_fit"]} for run in fitted}
        for attribute in ATTRIBUTES:
            cal_records = [r for r in runs["m1"]["cal_fit"]
                           if r["condition"] == "redacted" and r["attribute"] == attribute]
            test_records = [r for r in runs["m1"][split]
                            if r["condition"] == "redacted" and r["clean"] and r["attribute"] == attribute]
            for run in cal:
                cal[run][attribute] = arrays([cal_by_id[run][r["example_id"]] for r in cal_records])
            for run in test:
                test[run][attribute] = arrays([by_id[run][r["example_id"]] for r in test_records])
        intervals[split], edges[split] = bootstrap(cal, test, args.resamples)

    rows, better = verdict(intervals["heldout_test"], PRIMARY_BRIER, PRIMARY_ACCURACY)
    lines += ["", "## Primary comparison (fixed before any Milestone 6 run)", "",
              ("The mean over seeds 0–2 of the M4 v1 minus M2-logit difference, held-out test, redacted text, clean "
               "items, both attributes pooled. M4 counts as better if the 95% interval of the Brier difference lies "
               "below zero and the lower bound of the accuracy difference is above −1 point."), "", *rows, "",
              f"**Verdict: M4 v1 {'counts' if better else 'does not count'} as better than M2-logit.**"]
    rows, better = verdict(intervals["heldout_test"], TEST_BRIER, TEST_ACCURACY)
    lines += ["", "## The test of M5-logit (fixed before any Milestone 6 run)", "",
              ("M5's own letter probabilities with a temperature fitted on its map-fitting predictions had the best "
               "held-out Brier score of any method in Milestone 5, a result found after the results were seen. It is "
               "tested here on two new training runs: the mean over seeds 1 and 2 of the M5-logit minus M2-logit "
               "difference, with the same rule. Seed 0 is reported below but left out of the test. The test set is the "
               "one the hypothesis was found on."), "", *rows, "",
              f"**Verdict: M5-logit {'counts' if better else 'does not count'} as better than M2-logit.**", "",
              ("Sensitivity: in some draws a refitted temperature runs to the edge of its search range, because the "
               "resampled map-fitting families hold almost no wrong answers. Draws per run: "
               + (", ".join(f"{run} {n:,}" for run, n in sorted(edges["heldout_test"].items())) or "none") + ". "
               "Without those draws, the Brier intervals are "
               + "; ".join(f"{key.split(',')[1].strip()}: {intervals['heldout_test'][key][4]:+.4f} to "
                           f"{intervals['heldout_test'][key][5]:+.4f}" for key in (PRIMARY_BRIER, TEST_BRIER)) + ".")]

    lines += ["", "## Other comparisons, redacted text, clean items", "",
              (f"Paired family bootstrap ({args.resamples:,} draws), the same draws for every model; every draw refits "
               "M2's map, M2-logit's temperature and each M5 seed's temperature. Negative Brier differences favour the "
               "first method."), "",
              "| Split | Difference | Estimate | 95% interval | Undefined draws |", "|---|---|---:|---|---:|"]
    for split in SPLITS:
        for key, (estimate, low, high, undefined, *_) in intervals[split].items():
            lines.append(f"| {SPLITS[split]} | {key} | {estimate:+.4f} | {low:+.4f} to {high:+.4f} | {undefined} |")

    lines += ["", "## Results, held-out and seen test, redacted text, clean items", "",
              "Pooled over both attributes; each attribute and text condition uses its own maps and threshold.", "",
              ("| Split | Method | Accuracy | Brier, invalid = 1 | ECE | AUROC | Risk–coverage area | "
               "Coverage at threshold | Error at threshold |"), "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    confidences, correct, scores = {}, {}, {}
    for split in SPLITS:
        for name, (run, key, fit) in METHODS.items():
            records = [r for r in runs[run][split] if r["condition"] == "redacted" and r["clean"]]
            q = methods.confidence(records, key, fitted.get(fit))
            s = evaluate(records, q, thresholds[name])
            if split == "heldout_test":
                confidences[name], correct[name], scores[name] = q, [r["correct"] for r in records], s
            lines.append(f"| {SPLITS[split]} | {name} | {pct(s['accuracy'])} | {num(s['brier'])} | {num(s['ece'])} | "
                         f"{num(s['auroc'])} | {num(s['aurc'])} | {pct(s['coverage'])} | {pct(s['risk'])} |")

    lines += ["", "## Spread over seeds, held-out test, redacted text, clean items", "",
              "Mean over the three seeds, with the lowest and highest seed in brackets.", "",
              "| Method | Accuracy | Brier, invalid = 1 | ECE | AUROC |", "|---|---|---|---|---|"]
    for family in ("M4 v1", "M3 v1", "M5", "M5-logit"):
        cells = []
        for key, show in (("accuracy", pct), ("brier", num), ("ece", num), ("auroc", num)):
            values = [scores[f"{family} s{s}"][key] for s in SEEDS]
            cells.append(f"{show(mean(values))} ({show(min(values))} to {show(max(values))})")
        lines.append(f"| {family} | {' | '.join(cells)} |")

    lines += ["", "## Written values, held-out test, redacted text, clean items", "",
              "| Model | Distinct values | Most common value | Its share | Share below 0.9 |",
              "|---|---:|---:|---:|---:|"]
    for name in [f"{arm} s{s}" for arm in ("M4 v1", "M3 v1", "M5") for s in SEEDS]:
        written = [q for q in confidences[name] if q is not None]
        value, count = Counter(written).most_common(1)[0]
        lines.append(f"| {name} | {len(set(written))} | {value:g} | {pct(count / len(written))} | "
                     f"{pct(mean([q < 0.9 for q in written]))} |")
    lines += ["", "How often each written value is right (values written at least 20 times):", "",
              "| Model | Written confidence | Answers | Right |", "|---|---:|---:|---:|"]
    for name in [f"M4 v1 s{s}" for s in SEEDS]:
        by_value = defaultdict(list)
        for q, c in zip(confidences[name], correct[name]):
            if q is not None:
                by_value[q].append(c)
        for value, outcomes in sorted(by_value.items()):
            if len(outcomes) >= 20:
                lines.append(f"| {name} | {value:g} | {len(outcomes):,} | {pct(sum(outcomes) / len(outcomes))} |")

    with open(args.subset, encoding="utf-8") as f:
        subset = {(e["product_id"], e["attribute"], e["condition"]) for e in (json.loads(line) for line in f)}
    lines += ["", "## Evidence runs, 1,000 clean held-out items (the photo-swap subset, unswapped)", "",
              ("Each model answers the same items with both photo and text, with the photo alone (text removed), and "
               "with the text alone (photo removed), in each text condition. A confidence that follows the evidence "
               "falls as far as accuracy does, so the gap stays near zero. M2-logit and M5-logit use the maps fitted "
               "on full evidence, as they would be in use. Descriptive; no decision rule."), "",
              "| Model | Evidence | n | Accuracy | Invalid | Mean confidence | Confidence minus accuracy | AUROC |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    points = {}
    for name, run in EVIDENCE.items():
        _, key, fit = METHODS[name]
        points[name] = {}
        for mode, split, condition in MODES:
            records = [r for r in (runs[run][split] if split in SPLITS else load_run(art, f"{run}_{split}"))
                       if r["condition"] == condition and (r["product_id"], r["attribute"], r["condition"]) in subset]
            q = methods.confidence(records, key, fitted.get(fit))
            c = [r["correct"] for r in records]
            accuracy, confidence = mean(c), mean(q)
            points[name][mode] = (accuracy, confidence)
            lines.append(f"| {name} | {mode} | {len(records):,} | {pct(accuracy)} | "
                         f"{pct(mean([r['error'] is not None for r in records]))} | {num(confidence)} | "
                         f"{confidence - accuracy:+.3f} | {num(metrics.auroc(q, c))} |")

    with open(args.swaps, encoding="utf-8") as f:
        swaps = {e["example_id"]: e for e in (json.loads(line) for line in f)}
    lines += ["", "## Photo swap, held-out test", "",
              ("Each item's text with another product's photo whose label differs (same category, another family). "
               "With full text the two disagree; with redacted text the photo is the only evidence of the label. "
               "Confidence is compared with the same items and condition unswapped. M4 v1 seed 0 has no swap run."), "",
              ("| Model | Text | n | Answers the text's label | Answers the photo's label | Other or invalid | "
               "Mean confidence, swapped | Mean confidence, unswapped |"),
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, (run, key, fit) in METHODS.items():
        if name == "M2" or not (art / f"{run}_swap").exists():
            continue
        swapped = load_run(art, f"{run}_swap")
        q_swapped = methods.confidence(swapped, key, fitted.get(fit))
        normal = {r["example_id"]: r for r in runs[run]["heldout_test"]}
        for condition in ("full", "redacted"):
            rows = [(r, q) for r, q in zip(swapped, q_swapped) if r["condition"] == condition]
            base = [normal[r["example_id"].removesuffix(":swap")] for r, _ in rows]
            q_base = methods.confidence(base, key, fitted.get(fit))
            text = mean([r["choice"] == r["answer"] for r, _ in rows])
            photo = mean([r["choice"] == swaps[r["example_id"]]["photo_answer"] for r, _ in rows])
            lines.append(f"| {name} | {condition} | {len(rows):,} | {pct(text)} | {pct(photo)} | "
                         f"{pct(1 - text - photo)} | {num(mean(q for _, q in rows))} | {num(mean(q_base))} |")

    args.figures.mkdir(parents=True, exist_ok=True)
    signal_figure(histories, args.figures / "phase4_signal.png")
    reliability_figure(confidences, correct, ["M2-logit", "M4 v1 s0", "M4 v1 s1", "M4 v1 s2", "M5-logit s1",
                                              "M5-logit s2"], args.figures / "phase4_reliability.png")
    evidence_figure(points, args.figures / "phase4_evidence.png")
    lines += ["", "## Figures", "", "![Learning signal](figures/phase4_signal.png)", "",
              "![Reliability](figures/phase4_reliability.png)", "", "![Evidence](figures/phase4_evidence.png)"]

    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
