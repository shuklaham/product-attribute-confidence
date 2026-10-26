"""Write model-facing examples for every included label, under both text conditions.

Examples contain product text, so they go under data/ (gitignored), never into the repo. The
training split is capped per (category, attribute, label) cell, in a seeded order. The script
refuses to run if the style JSONs do not match the fingerprint stored with the manifests.

Usage:
    uv run python -I scripts/build_examples.py <styles_dir> <manifest_dir> <out_dir> [--train-cap 1500]
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from pac import ontology as ontology_module
from pac.examples import CONDITIONS, make_example
from pac.records import directory_fingerprint, load_products

CAP_SALT = "train-cap-v0"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("styles_dir")
    parser.add_argument("manifest_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--train-cap", type=int, default=1500)
    args = parser.parse_args()

    manifest = json.loads((args.manifest_dir / "manifest.json").read_text(encoding="utf-8"))
    if directory_fingerprint(args.styles_dir) != manifest["source_fingerprint"]:
        raise SystemExit("style JSONs do not match the manifest fingerprint; rebuild the manifests first")
    ontology = ontology_module.load()
    if ontology.sha256 != manifest["ontology_sha256"]:
        raise SystemExit("configs/ontology.yaml changed since the manifests were built; rebuild them first")

    products = {p.id: p for p in load_products(args.styles_dir)[0]}
    splits = {int(r["product_id"]): r for r in read_csv(args.manifest_dir / "products.csv")}
    labels = [r for r in read_csv(args.manifest_dir / "labels.csv") if r["included"] == "1"]

    cells = defaultdict(list)
    for r in labels:
        s = splits[int(r["product_id"])]
        if s["split"] == "train":
            cells[(s["category"], r["attribute"], r["label"])].append(r)
    capped = set()
    for members in cells.values():
        members.sort(key=lambda r: hashlib.sha256(f"{CAP_SALT}:{r['product_id']}:{r['attribute']}".encode()).hexdigest())
        capped.update((r["product_id"], r["attribute"]) for r in members[args.train_cap:])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles, counts = {}, Counter()
    try:
        for r in labels:
            if (r["product_id"], r["attribute"]) in capped:
                counts[("train", r["attribute"], "capped")] += 1
                continue
            pid = int(r["product_id"])
            s = splits[pid]
            if s["split"] not in handles:
                handles[s["split"]] = open(args.out_dir / f"{s['split']}.jsonl", "w", encoding="utf-8")
            for condition in CONDITIONS:
                example = make_example(products[pid], ontology.attributes[r["attribute"]], r["label"],
                                       s["split"], int(s["family_id"]), condition)
                handles[s["split"]].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[(s["split"], r["attribute"], "items")] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for (split, attribute, kind), n in sorted(counts.items()):
        print(f"{split:>14}  {attribute:<14} {kind:<7} {n:>6,}" + ("  (x2 conditions)" if kind == "items" else ""))


if __name__ == "__main__":
    main()
