"""Phase-1 report: the base model (M0), plain SFT (M1), M1 with a Platt map on its stated
confidence (M2), and M1's temperature-scaled letter probabilities (M2-logit), on the seen and
held-out test splits.

Maps are fitted on M1's predictions for the map-fitting half of the calibration split, per
attribute and text condition, and every method's auto-fill threshold on the threshold half;
nothing is tuned on test. Intervals for differences come from a paired bootstrap over product
families that also refits the maps on a resample of the calibration families.

Usage:
    uv run python -I scripts/report_phase1.py [--artifacts artifacts] [--out reports/phase1.md]
        [--figures reports/figures] [--resamples 10000]
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pac import evaluation, metrics, methods  # noqa: E402
from pac.calibration import EPSILON, Temperature, fit_platt, fit_temperature, logit, sigmoid  # noqa: E402

SPLITS = {"test": "Seen test", "heldout_test": "Held-out test"}
METHODS = {"M0": ("m0", "stated"), "M1": ("m1", "stated"), "M2": ("m1", "platt"), "M2-logit": ("m1", "letter")}
# Reference palette, light mode (validated: all checks pass for the first three slots, all pairs).
COLORS = {"M0": "#2a78d6", "M1": "#eb6834", "M2-logit": "#1baf7a", "M2": "#eda100"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_run(artifacts: Path, name: str) -> list[dict]:
    path = artifacts / name
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    with open(path / "predictions.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if len(records) != run["n"]:
        raise SystemExit(f"{name}: {len(records)} records but the run record says {run['n']}")
    return records


def pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def num(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def evaluate(records: list[dict], q: list, thresholds: dict) -> dict:
    """Metrics for one method on one set of records, pooled over attributes."""
    correct = np.array([r["correct"] for r in records], dtype=float)
    accepted = np.array([v is not None and v >= thresholds[methods.group_key(r)] for r, v in zip(records, q)])
    n_accepted = int(accepted.sum())
    return {
        "n": len(records),
        "accuracy": float(correct.mean()),
        "invalid": float(np.mean([r["error"] is not None for r in records])),
        "brier": metrics.brier(q, correct, invalid_loss=1.0),
        "ece": metrics.ece(q, correct),
        "auroc": metrics.auroc(q, correct),
        "aurc": metrics.aurc(q, correct),
        "coverage": n_accepted / len(records),
        "risk": float((1 - correct[accepted]).mean()) if n_accepted else float("nan"),
    }


def arrays(records: list[dict]) -> dict:
    """Per-record arrays for the bootstrap: stated confidence, correctness, letter log-probabilities."""
    width = max(len(r["letter_logprobs"]) for r in records if r["letter_logprobs"])
    rows, answers, chosen = [], [], []
    for r in records:
        letters = list(r["letter_logprobs"] or {})
        rows.append([r["letter_logprobs"][k] for k in letters] if letters else [0.0] * width)
        answers.append(letters.index(r["answer"]) if letters else 0)
        chosen.append(letters.index(r["choice"]) if r["error"] is None and r["choice"] in letters else -1)
    return {"stated": np.array([np.nan if r["confidence"] is None else r["confidence"] for r in records]),
            "correct": np.array([r["correct"] for r in records], dtype=float),
            "letters": np.array(rows), "answer": np.array(answers), "chosen": np.array(chosen),
            "has_letters": np.array([bool(r["letter_logprobs"]) for r in records]),
            "family": np.array([r["family_id"] for r in records])}


def mapped(a: dict, rows: np.ndarray, platt, t: float) -> tuple[np.ndarray, np.ndarray]:
    """M2 and M2-logit confidences for the given rows, NaN where a method has no confidence."""
    stated = a["stated"][rows]
    m2 = np.where(np.isnan(stated), np.nan, sigmoid(platt.a * logit(np.nan_to_num(stated, nan=0.5)) + platt.b))
    probabilities = Temperature(t).probabilities(a["letters"][rows])
    chosen = a["chosen"][rows]
    m2_logit = np.where(chosen >= 0, probabilities[np.arange(len(rows)), np.maximum(chosen, 0)], np.nan)
    return m2, m2_logit


def bootstrap(cal: dict, test: dict, m0_test: dict, resamples: int, seed: int = 0) -> dict:
    """Paired family bootstrap of the key differences on one test set, refitting the maps every draw.

    cal and test map attribute -> arrays; m0_test holds M0's arrays for the same test items.
    """
    def families(by_attribute):
        values = np.concatenate([by_attribute[k]["family"] for k in by_attribute])
        _, index = np.unique(values.astype(str), return_inverse=True)
        out, start = {}, 0
        for k in by_attribute:
            n = len(by_attribute[k]["family"])
            out[k], start = index[start:start + n], start + n
        return out, int(index.max()) + 1

    cal_index, n_cal = families(cal)
    test_index, n_test = families(test)

    def statistics(cal_counts, test_counts):
        q = {"M0": [], "M1": [], "M2": [], "M2-logit": []}
        correct, correct_m0 = [], []
        for k in test:
            cal_rows = np.repeat(np.arange(len(cal_index[k])), cal_counts[cal_index[k]])
            test_rows = np.repeat(np.arange(len(test_index[k])), test_counts[test_index[k]])
            c = cal[k]
            platt = fit_platt(c["stated"][cal_rows], c["correct"][cal_rows])
            with_letters = cal_rows[c["has_letters"][cal_rows]]
            t = fit_temperature(c["letters"][with_letters], c["answer"][with_letters], iterations=40).t
            m2, m2_logit = mapped(test[k], test_rows, platt, t)
            q["M0"].append(m0_test[k]["stated"][test_rows])
            q["M1"].append(test[k]["stated"][test_rows])
            q["M2"].append(m2)
            q["M2-logit"].append(m2_logit)
            correct.append(test[k]["correct"][test_rows])
            correct_m0.append(m0_test[k]["correct"][test_rows])
        q = {m: np.concatenate(v) for m, v in q.items()}
        c, c0 = np.concatenate(correct), np.concatenate(correct_m0)
        brier = {m: metrics.brier(q[m], c0 if m == "M0" else c, invalid_loss=1.0) for m in q}
        return {
            "Brier, M2-logit minus M2": brier["M2-logit"] - brier["M2"],
            "Brier, M2-logit minus M1": brier["M2-logit"] - brier["M1"],
            "Brier, M2 minus M1": brier["M2"] - brier["M1"],
            "AUROC, M2-logit minus M1": metrics.auroc(q["M2-logit"], c) - metrics.auroc(q["M1"], c),
            "Accuracy, M1 minus M0": float(c.mean() - c0.mean()),
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
    """Accuracy against stated confidence per bin; marker area follows the bin's item count."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2), sharex=True, sharey=True, facecolor=SURFACE)
    for row, split in enumerate(SPLITS):
        for col, method in enumerate(METHODS):
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
    """Expected error rate among the most confident items as coverage grows, with each method's frozen threshold.

    Ties are taken in a random order (metrics.risk_curve), so a method that states one value for
    every item draws a flat line at its overall error rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, split in zip(axes, SPLITS):
        for method in ("M0", "M1", "M2-logit"):
            coverage, risk = metrics.risk_curve(confidences[(split, method)], correct[(split, method)])
            ax.plot(coverage, risk, color=COLORS[method], linewidth=1.8, label=method)
            at, error = operating[(split, method)]
            if not math.isnan(error):
                ax.scatter([at], [error], color=COLORS[method], s=48, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.axhline(0.05, linestyle="--", color=INK_2, linewidth=0.8)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("reports/phase1.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    art = args.artifacts

    runs = {"m0": {s: load_run(art, f"m0_{s}") for s in ("cal_threshold", *SPLITS)},
            "m1": {s: load_run(art, f"m1_{s}") for s in ("cal_fit", "cal_threshold", *SPLITS)}}
    fitted = methods.fit(runs["m1"]["cal_fit"])
    thresholds = {name: methods.thresholds(runs[source]["cal_threshold"], key, fitted)
                  for name, (source, key) in METHODS.items()}

    lines = ["# Phase 1: base model, SFT and the cheap fixes", ""]
    best = (art / "m1_best_lr.txt").read_text().strip()
    lines += ["## M1: learning rate chosen on dev", "",
              ("Rule fixed before training: highest accuracy on redacted dev (invalid outputs count as wrong), then "
              "fewer invalid outputs, then the lower learning rate."), "",
              ("| Learning rate | Steps | Final training loss | Minutes | Redacted dev accuracy | Full dev accuracy | "
              "Invalid | Chosen |"), "|---|---:|---:|---:|---:|---:|---:|---|"]
    for lr in ("5e-5", "1e-4", "2e-4"):
        train = json.loads((art / f"m1_lr{lr}" / "run.json").read_text())
        dev = load_run(art, f"m1_lr{lr}_dev")
        accuracy = {c: np.mean([r["correct"] for r in dev if r["condition"] == c]) for c in ("redacted", "full")}
        invalid = np.mean([r["error"] is not None for r in dev])
        minutes = train["elapsed_seconds"] / 60
        lines.append(f"| {lr} | {train['steps']} | {train['final_train_loss']:.4f} | {minutes:.0f} | "
                     f"{pct(accuracy['redacted'])} | {pct(accuracy['full'])} | {pct(invalid)} | "
                     f"{'yes' if lr == best else ''} |")

    lines += ["", "## Fitted maps and thresholds", "",
              ("Fitted on M1's calibration predictions, per attribute and text condition. Thresholds: the lowest "
              "confidence whose accepted items have at most 5% errors on the threshold half; infinity means accept "
              "nothing."), "",
              "| Attribute | Text | Platt a | Platt b | Temperature | Threshold M0 | M1 | M2 | M2-logit |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(fitted.platt):
        p, t = fitted.platt[key], fitted.temperature[key].t
        lines.append(f"| {key[0]} | {key[1]} | {p.a:.3f} | {p.b:.3f} | {t:.3f} | "
                     + " | ".join(f"{thresholds[m][key]:.4f}" for m in METHODS) + " |")

    confidences, correct, operating = {}, {}, {}
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

    increasing = all(p.a > 0 for p in fitted.platt.values())
    lines += ["", "## AUROC within each attribute, redacted text, clean items", "",
              ("The pooled AUROC above also compares answers across the two attributes, so it partly rewards knowing "
               "which attribute is easier. Within one attribute it measures only which answers of the same kind are "
               "riskier" + (", and M2 equals M1 there, because each of its maps is increasing (a > 0 above)."
                            if increasing else ".")), "",
              "| Split | Attribute | n | M0 | M1 | M2 | M2-logit |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for attribute in ("neckline", "sleeve_length"):
            cells = []
            for name, (source, key) in METHODS.items():
                records = [r for r in runs[source][split]
                           if r["condition"] == "redacted" and r["clean"] and r["attribute"] == attribute]
                cells.append(num(metrics.auroc(methods.confidence(records, key, fitted),
                                               [r["correct"] for r in records])))
            lines.append(f"| {SPLITS[split]} | {attribute} | {len(records):,} | " + " | ".join(cells) + " |")

    lines += ["", "## Accuracy by category, redacted text, clean items", "", "| Split | Category | n | M0 | M1 |",
              "|---|---|---:|---:|---:|"]
    for split in SPLITS:
        cats = Counter(r["category"] for r in runs["m1"][split] if r["condition"] == "redacted" and r["clean"])
        for category, n in cats.most_common():
            acc = {m: np.mean([r["correct"] for r in runs[m][split] if r["condition"] == "redacted" and r["clean"]
                               and r["category"] == category]) for m in ("m0", "m1")}
            lines.append(f"| {SPLITS[split]} | {category} | {n:,} | {pct(acc['m0'])} | {pct(acc['m1'])} |")

    lines += ["", "## Evidence sensitivity, clean items", "",
              ("From full to redacted text on the same items: the fall in mean confidence beside the fall in accuracy, "
              "both on the items valid in both conditions. A model that notices the missing evidence lowers its "
              "confidence about as much as its accuracy falls."), "",
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

    stated = Counter(r["confidence"] for s in SPLITS for r in runs["m1"][s] if r["confidence"] is not None)
    total = sum(stated.values())
    lines += ["", "## M1's stated confidence", "",
              f"Across both test splits, M1 states {len(stated)} distinct values. Most common: "
              + ", ".join(f"{v:g} ({n / total:.1%})" for v, n in stated.most_common(5)) + "."]
    fitted_on = [r["confidence"] for s in ("cal_fit", "cal_threshold", *SPLITS) for r in runs["m1"][s]
                 if r["confidence"] is not None]
    clipped = sum(not EPSILON <= v <= 1 - EPSILON for v in fitted_on)
    lines += ["", (f"Clip check: {clipped} of {len(fitted_on):,} stated values in M1's calibration and test runs lie "
                   f"outside [{EPSILON}, {1 - EPSILON}], the range the Platt map clips to"
                   + (", so the clip never binds and M2 does not depend on it." if not clipped
                      else ", so M2 depends on where the clip puts them."))]

    m0_fitted = methods.fit(load_run(art, "m0_cal_fit"))
    lines += ["", "## Letter probabilities before and after SFT", "",
              ("Temperatures fitted the same way on each model's own map-fitting predictions: 1 means the letter "
               "probabilities need no correction, above 1 means they are overconfident."), "",
              "| Attribute | Text | Temperature, M0 | Temperature, M1 |", "|---|---|---:|---:|"]
    for key in sorted(fitted.temperature):
        lines.append(f"| {key[0]} | {key[1]} | {m0_fitted.temperature[key].t:.2f} | {fitted.temperature[key].t:.2f} |")
    untempered = methods.Fitted(fitted.platt, {key: Temperature(1.0) for key in fitted.temperature})
    lines += ["", "M1's letter probabilities without and with the fitted temperature, redacted text, clean items:", "",
              "| Split | Brier, temperature 1 | Brier, fitted | ECE, temperature 1 | ECE, fitted |",
              "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        records = [r for r in runs["m1"][split] if r["condition"] == "redacted" and r["clean"]]
        c = [r["correct"] for r in records]
        raw, scaled = (methods.confidence(records, "letter", f) for f in (untempered, fitted))
        lines.append(f"| {SPLITS[split]} | {num(metrics.brier(raw, c, invalid_loss=1.0))} | "
                     f"{num(metrics.brier(scaled, c, invalid_loss=1.0))} | {num(metrics.ece(raw, c))} | "
                     f"{num(metrics.ece(scaled, c))} |")

    args.figures.mkdir(parents=True, exist_ok=True)
    reliability_figure(confidences, correct, args.figures / "phase1_reliability.png")
    risk_coverage_figure(confidences, correct, operating, args.figures / "phase1_risk_coverage.png")
    lines += ["", "## Figures", "", "![Reliability](figures/phase1_reliability.png)", ""]
    top_bins = {min(int(v * 10), 9) for v in stated}
    if len(top_bins) == 1:
        lines += [(f"Reliability: M1's stated values ({', '.join(f'{v:g}' for v in sorted(stated))}) all fall in one "
                   "bin, so its diagram shows a single point and its ECE pools them; the Brier score and AUROC still "
                   "tell them apart."), ""]
    lines += ["![Risk–coverage](figures/phase1_risk_coverage.png)", "",
              ("Risk–coverage: lines rank both attributes together, with ties in a random order, so a method that "
              "states one value for every item draws a flat line at its overall error rate. Dots are the outcome of "
              "each method's frozen thresholds, which are set per attribute, so a dot can sit off its line. M2 is left "
              "out to keep the figure readable: within each attribute it ranks answers exactly as M1 does, and its "
              "pooled risk–coverage area is in the results table.")]

    lines += ["", "## Differences with 95% intervals, redacted text, clean items", "",
              (f"Paired bootstrap over product families ({args.resamples:,} draws); every draw also refits M2's "
              "Platt map and M2-logit's temperature on a resample of the calibration families. Negative Brier "
              "differences favour "
              "the first method."), "",
              "| Split | Difference | Estimate | 95% interval | Undefined draws |", "|---|---|---:|---|---:|"]
    for split in SPLITS:
        cal, test, m0_test = {}, {}, {}
        for attribute in ("neckline", "sleeve_length"):
            cal[attribute] = arrays([r for r in runs["m1"]["cal_fit"]
                                     if r["condition"] == "redacted" and r["attribute"] == attribute])
            test_records = [r for r in runs["m1"][split]
                            if r["condition"] == "redacted" and r["clean"] and r["attribute"] == attribute]
            by_id = {r["example_id"]: r for r in runs["m0"][split]}
            test[attribute] = arrays(test_records)
            m0_test[attribute] = arrays([by_id[r["example_id"]] for r in test_records])
        for key, (estimate, low, high, undefined) in bootstrap(cal, test, m0_test, args.resamples).items():
            lines.append(f"| {SPLITS[split]} | {key} | {estimate:+.4f} | {low:+.4f} to {high:+.4f} | {undefined} |")

    text = "\n".join(lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    (art / "phase1_fitted.json").write_text(json.dumps({
        "platt": {"/".join(k): {"a": v.a, "b": v.b} for k, v in fitted.platt.items()},
        "temperature": {"/".join(k): v.t for k, v in fitted.temperature.items()},
        "thresholds": {m: {"/".join(k): v for k, v in t.items()} for m, t in thresholds.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
