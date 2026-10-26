"""Data audit for the Myntra catalog under ontology v0.

Writes label coverage, every exclusion by reason, included labels per split, the label-noise
estimate by category, a category-majority baseline, a redaction check, and colour-variant
family statistics.

Usage:
    uv run python -I scripts/audit.py <styles_dir> <report.md> [photos.csv]
"""

import collections
import csv
import statistics
import sys
from pathlib import Path

from pac import ontology as ontology_module
from pac.records import directory_fingerprint, load_products
from pac.redact import redact_description, redact_title
from pac.splits import (CROSSING_SPLIT, EVAL_SPLITS, HELD_OUT, HELD_OUT_SPLIT, SEEN, assign_splits, families,
                        label_rows, usable_photos)
from pac.text import mask_brand

SPLIT_ORDER = ["train", "dev", "cal_fit", "cal_threshold", "test", HELD_OUT_SPLIT]
REASONS = ["", "text_conflict", "family_disagreement", "roll_up_sleeves", "dropped_label", "unmapped_value",
           "crossing_family", "no_usable_photo", "unsupported_label"]


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "–"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def main(styles_dir: str, report_path: str, photos_csv: str | None = None) -> None:
    ontology = ontology_module.load()
    products, failures = load_products(styles_dir)
    by_id = {p.id: p for p in products}
    family_of = families(products)
    splits = assign_splits(products, family_of)
    photo_ok = None
    if photos_csv:
        with open(photos_csv, newline="", encoding="utf-8") as f:
            photo_ok = usable_photos(list(csv.DictReader(f)))
    rows = label_rows(products, splits, ontology, photo_ok)
    side = {pid: ("seen" if s.category in SEEN else "held-out") for pid, s in splits.items()}

    out = ["# Data audit: Myntra Fashion Product Images, ontology v0", "",
           f"Parsed {len(products):,} records ({failures} unreadable); source fingerprint "
           f"`{directory_fingerprint(styles_dir)[:12]}`, ontology `{ontology.version}` (`{ontology.sha256[:12]}`). "
           f"{len(splits):,} products are in the {len(SEEN)} seen and {len(HELD_OUT)} held-out categories.", ""]

    coverage = []
    for category in SEEN + HELD_OUT:
        ps = [by_id[pid] for pid, s in splits.items() if s.category == category]
        coverage.append([category, "seen" if category in SEEN else "held-out", len(ps)]
                        + [sum(1 for p in ps if attr.canonical(p.raw_labels.get(attr.source_field))[1] != "unlabeled")
                           for attr in ontology.attributes.values()])
    out += ["## Coverage by category", "",
            table(["Category", "Side", "Products"] + [f"{k} values" for k in ontology.attributes], coverage), ""]

    for key, attribute in ontology.attributes.items():
        attr_rows = [r for r in rows if r.attribute == key]
        out += [f"## {key.replace('_', ' ').capitalize()}", ""]

        decided = collections.Counter((r.reason, side[r.product_id]) for r in attr_rows)
        out += ["### Decisions", "", table(
            ["Outcome", "Seen", "Held-out"],
            [[reason or "included", decided[(reason, "seen")], decided[(reason, "held-out")]] for reason in REASONS]), ""]

        per_split = collections.Counter((r.label, splits[r.product_id].split) for r in attr_rows if r.included)
        out += ["### Included labels by split", "", table(
            ["Label"] + SPLIT_ORDER,
            [[label] + [per_split[(label, s)] for s in SPLIT_ORDER] for label in attribute.label_names]), ""]

        mapped = [r for r in attr_rows if r.label and splits[r.product_id].split != CROSSING_SPLIT]
        noise = collections.defaultdict(lambda: [0, 0])
        for r in mapped:
            noise[splits[r.product_id].category][1] += 1
            noise[splits[r.product_id].category][0] += r.text_status == "conflicts"
        out += ["### Label-noise estimate", "",
                "Share of mapped labels whose listing text names a different label. These are excluded from "
                "training and the clean test sets.", "",
                table(["Category", "Mapped labels", "Text names another label", "Rate"],
                      [[c, n, k, pct(k, n)] for c, (k, n) in sorted(noise.items(), key=lambda kv: -kv[1][0] / kv[1][1])]),
                ""]

        status_rows = []
        for group in ("seen", "held-out"):
            included = [r for r in attr_rows if r.included and side[r.product_id] == group]
            counts = collections.Counter(r.text_status for r in included)
            status_rows.append([group, len(included), pct(counts["agrees"], len(included)),
                                pct(counts["silent"], len(included))])
        out += ["### Included labels: does the text name the answer?", "",
                table(["Side", "Included", "Text names the label", "Text names no label"], status_rows), ""]

        majority_rows = []
        for split in ("test", HELD_OUT_SPLIT):
            included = [r for r in attr_rows if r.included and splits[r.product_id].split == split]
            by_category = collections.defaultdict(collections.Counter)
            for r in included:
                by_category[splits[r.product_id].category][r.label] += 1
            hits = sum(c.most_common(1)[0][1] for c in by_category.values())
            majority_rows.append([split, len(included), pct(hits, len(included))])
        out += ["### Guessing the most common label per category (no photo, no text)", "",
                table(["Split", "Included", "Accuracy"], majority_rows), ""]

        evaluated = [r for r in attr_rows if r.included and splits[r.product_id].split in EVAL_SPLITS]
        residual = empty = 0
        ratios = []
        for r in evaluated:
            p = by_id[r.product_id]
            title = redact_title(p.title, attribute, p.brand)
            description = redact_description(p.description, attribute, p.brand)
            text = mask_brand(f"{title} {description}", p.brand)  # brand names are kept on purpose (rule 4)
            residual += bool(attribute.attribute_words.search(text) or any(x.search(text) for x in attribute.all_patterns))
            empty += p.description != "" and description == ""
            if p.description:
                ratios.append(len(description) / len(p.description))
        out += ["### Redaction check (evaluation splits)", "",
                f"- Redacted texts that still contain an attribute word or label phrase: {residual} of {len(evaluated):,}.",
                f"- Descriptions that become empty: {empty} ({pct(empty, len(evaluated))}).",
                f"- Median share of description length kept: {statistics.median(ratios):.0%}." if ratios else "",
                "- The stronger check, a text-only model on redacted dev text scoring near the category guess, "
                "runs in Milestone 2.", ""]

    members = collections.defaultdict(list)
    for pid, s in splits.items():
        members[s.family_id].append(pid)
    sizes = collections.Counter(min(len(m), 5) for m in members.values())
    shared = sum(len(m) for m in members.values() if len(m) > 1)
    crossing = sum(1 for m in members.values() if splits[m[0]].split == CROSSING_SPLIT)
    out += ["## Colour-variant families", "",
            f"- {len(members):,} families cover {len(splits):,} in-scope products; {shared:,} products "
            f"({pct(shared, len(splits))}) share a family.",
            "- Family sizes (5 means 5 or more): " + ", ".join(f"{k}: {v:,}" for k, v in sorted(sizes.items())),
            f"- Families spanning seen and held-out categories, excluded: {crossing}.", ""]

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
