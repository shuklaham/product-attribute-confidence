"""Coverage and leakage audit for the Myntra Fashion Product Images catalog.

Reads the per-product style JSONs and writes a Markdown report covering label
coverage by category, label distributions, how often the text names the label,
colour-variant families, and the proposed seen / held-out category split.

Usage:
    uv run python -I src/audit.py <styles_dir> <report.md>
"""

import collections
import glob
import html
import json
import os
import re
import sys

SEEN = ["Tshirts", "Tops", "Shirts"]
HELD_OUT = ["Kurtas", "Kurtis", "Tunics", "Dresses", "Nightdress", "Night suits", "Sweatshirts", "Sweaters"]
ATTRIBUTES = {"neckline": "Neck", "sleeve_length": "Sleeve Length"}
MISSING = {"", "NA", "N/A", "None", "null"}
ALIASES = {"sleeve_length": {"No Sleeves": "Sleeveless"}}
MIN_TRAIN_SUPPORT = 30

# Words that name the attribute at all, and phrases that name a specific label.
ATTRIBUTE_WORD = {
    "neckline": re.compile(r"neck|collar", re.I),
    "sleeve_length": re.compile(r"sleeve", re.I),
}
LABEL_PHRASES = {
    "neckline": {
        "Round Neck": r"round[- ]?neck|crew[- ]?neck",
        "V-Neck": r"\bv[- ]?neck",
        "Polo Collar": r"\bpolo\b",
        "Mandarin Collar": r"mandarin|chinese collar|band collar",
        "Scoop Neck": r"\bscoop",
        "Boat Neck": r"boat[- ]?neck|bateau",
        "Square Neck": r"square[- ]?neck",
        "Shirt Collar": r"shirt collar|spread collar|collared",
        "Shoulder Straps": r"shoulder strap|spaghetti|strappy",
        "Halter Neck": r"\bhalter",
        "Henley Neck": r"\bhenley",
        "Strapless": r"strapless|\btube\b",
        "Off-Shoulder": r"off[- ]?shoulder",
        "High Neck": r"high[- ]?neck|turtle|mock neck",
        "Hood": r"\bhood",
        "Cowl Neck": r"\bcowl",
        "Sweetheart Neck": r"sweetheart",
        "Keyhole Neck": r"keyhole",
        "Tie-Up Neck": r"tie[- ]?up",
    },
    "sleeve_length": {
        "Short Sleeves": r"short[- ]sleeve|half[- ]sleeve|cap sleeve",
        "Long Sleeves": r"long[- ]sleeve|full[- ]sleeve",
        "Three-Quarter Sleeves": r"three[- ]quarter|3/4",
        "Sleeveless": r"sleeveless|no sleeve|\btank\b|\bcami",
    },
}
LABEL_PATTERNS = {a: {lab: re.compile(p, re.I) for lab, p in d.items()} for a, d in LABEL_PHRASES.items()}


def clean_html(raw):
    text = re.sub(r"<div[^>]*_mcePaste[^>]*>.*?</div>", " ", raw or "", flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def canonical(attribute, value):
    if value is None or str(value).strip() in MISSING:
        return None
    value = str(value).strip()
    return ALIASES.get(attribute, {}).get(value, value)


def sibling_ids(record):
    """Style IDs of the product's other colour variants, from `colours.colors.<id>.styleid`."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in ("styleid", "styleId") and isinstance(val, (int, str)) and str(val).isdigit():
                    found.add(int(val))
                else:
                    walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(record.get("colours"))
    return found


def load(styles_dir):
    products, failures = [], 0
    for path in glob.glob(os.path.join(styles_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f).get("data") or {}
        except (OSError, ValueError):
            failures += 1
            continue
        attrs = d.get("articleAttributes") or {}
        descriptors = d.get("productDescriptors") or {}
        products.append({
            "id": int(d.get("id") or os.path.basename(path).split(".")[0]),
            "category": (d.get("articleType") or {}).get("typeName"),
            "title": d.get("productDisplayName") or "",
            "variant": d.get("variantName") or "",
            "description": clean_html((descriptors.get("description") or {}).get("value")),
            "labels": {a: canonical(a, attrs.get(field)) for a, field in ATTRIBUTES.items()},
            "views": sorted((d.get("styleImages") or {}).keys()),
            "siblings": sibling_ids(d),
        })
    return products, failures


def families(products):
    parent = {p["id"]: p["id"] for p in products}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in products:
        for s in p["siblings"]:
            if s in parent:
                parent[find(s)] = find(p["id"])
    return {p["id"]: find(p["id"]) for p in products}


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(styles_dir, report_path):
    products, failures = load(styles_dir)
    family_of = families(products)
    in_scope = [p for p in products if p["category"] in SEEN + HELD_OUT]
    lines = ["# Data audit: Myntra Fashion Product Images", ""]
    lines += [f"Parsed {len(products):,} product records ({failures} unreadable). "
              f"{len(in_scope):,} are in the {len(SEEN)} seen and {len(HELD_OUT)} held-out categories.", ""]

    # Coverage by category
    rows = []
    for cat in SEEN + HELD_OUT:
        ps = [p for p in in_scope if p["category"] == cat]
        rows.append([cat, "seen" if cat in SEEN else "held-out", len(ps)]
                    + [sum(p["labels"][a] is not None for p in ps) for a in ATTRIBUTES]
                    + [pct(sum(len(p["views"]) > 1 for p in ps), len(ps))])
    lines += ["## Label coverage by category", "",
              table(["Category", "Split", "Products", "Neckline labels", "Sleeve labels", "Extra views listed"], rows), ""]

    for attribute in ATTRIBUTES:
        labeled = [p for p in in_scope if p["labels"][attribute] is not None]
        seen = [p for p in labeled if p["category"] in SEEN]
        held = [p for p in labeled if p["category"] in HELD_OUT]
        lines += [f"## {attribute.replace('_', ' ').capitalize()}", ""]

        # Label support across the split
        seen_c = collections.Counter(p["labels"][attribute] for p in seen)
        held_c = collections.Counter(p["labels"][attribute] for p in held)
        rows = []
        for lab in sorted(set(seen_c) | set(held_c), key=lambda l: -(seen_c[l] + held_c[l])):
            note = "too little training support" if held_c[lab] and seen_c[lab] < MIN_TRAIN_SUPPORT else ""
            rows.append([lab, seen_c[lab], held_c[lab], note])
        lines += ["### Label counts", "", table(["Label", "Seen categories", "Held-out categories", "Note"], rows), ""]

        # Category-majority guess: accuracy with no image and no attribute words
        rows = []
        for group, ps in (("seen", seen), ("held-out", held)):
            by_cat = collections.defaultdict(collections.Counter)
            for p in ps:
                by_cat[p["category"]][p["labels"][attribute]] += 1
            hits = sum(c.most_common(1)[0][1] for c in by_cat.values())
            rows.append([group, len(ps), pct(hits, len(ps))])
        lines += ["### Guessing the most common label per category", "",
                  table(["Split", "Labeled products", "Accuracy"], rows), ""]

        # Text leakage and label-text agreement
        rows = []
        for group, ps in (("seen", seen), ("held-out", held)):
            n = len(ps)
            word = {f: sum(bool(ATTRIBUTE_WORD[attribute].search(p[f])) for p in ps) for f in ("title", "variant", "description")}
            agree = conflict = silent = 0
            for p in ps:
                text = " ".join((p["title"], p["variant"], p["description"]))
                named = {lab for lab, pat in LABEL_PATTERNS[attribute].items() if pat.search(text)}
                if p["labels"][attribute] in named:
                    agree += 1
                elif named:
                    conflict += 1
                else:
                    silent += 1
            rows.append([group, n, pct(word["title"], n), pct(word["variant"], n), pct(word["description"], n),
                         pct(agree, n), pct(conflict, n), pct(silent, n)])
        lines += ["### How often the text gives the label away", "",
                  table(["Split", "N", "Attribute word in title", "in variant name", "in description",
                         "Text names the label", "Text names another label", "Text names none"], rows), "",
                  "\"Text names another label\" marks possible label noise or a phrase-matching miss; inspect samples before excluding.", ""]

    # Families
    fam_members = collections.defaultdict(list)
    for p in in_scope:
        fam_members[family_of[p["id"]]].append(p)
    sizes = collections.Counter(min(len(m), 5) for m in fam_members.values())
    multi = sum(len(m) for m in fam_members.values() if len(m) > 1)
    crossing = sum(1 for m in fam_members.values()
                   if {p["category"] in SEEN for p in m} == {True, False})
    conflicts = {a: sum(1 for m in fam_members.values()
                        if len({p["labels"][a] for p in m if p["labels"][a]}) > 1) for a in ATTRIBUTES}
    lines += ["## Colour-variant families", "",
              f"- {len(fam_members):,} families cover {len(in_scope):,} in-scope products; "
              f"{multi:,} products ({pct(multi, len(in_scope))}) share a family with another product.",
              "- Family sizes (5 means 5 or more): " + ", ".join(f"{k}: {v:,}" for k, v in sorted(sizes.items())),
              f"- Families spanning seen and held-out categories: {crossing} (keep together or exclude).",
              "- Families with conflicting labels: " + ", ".join(f"{a}: {n}" for a, n in conflicts.items()), ""]

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}: {len(products):,} records, {len(in_scope):,} in scope, {len(fam_members):,} families")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
