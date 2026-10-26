"""Write model-facing examples for every included label, under both text conditions.

Examples contain product text, so they go under data/ (gitignored), never into the repo. The
training split is capped per (category, attribute, label) cell, in a seeded order. The test
splits also get the items excluded only as label noise (a contradicting description or a family
whose variants disagree), marked clean = false, for the "all labeled items" results. The script
refuses to run if the style JSONs, the label set or the manifest files differ from what the
manifest recorded, and it writes a build record next to the examples.

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
from pac.splits import TEST_CONFLICT_REASONS, TEST_SPLITS

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
    for name, digest in manifest["files"].items():
        if hashlib.sha256((args.manifest_dir / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{name} differs from the hash recorded in manifest.json; rebuild the manifests")

    products = {p.id: p for p in load_products(args.styles_dir)[0]}
    splits = {int(r["product_id"]): r for r in read_csv(args.manifest_dir / "products.csv")}
    rows = read_csv(args.manifest_dir / "labels.csv")
    labels = [r for r in rows if r["included"] == "1"]
    supported = Counter((r["attribute"], r["label"]) for r in labels
                        if splits[int(r["product_id"])]["split"] == "train")
    noisy = [r for r in rows if r["reason"] in TEST_CONFLICT_REASONS
             and splits[int(r["product_id"])]["split"] in TEST_SPLITS
             and supported[(r["attribute"], r["label"])] >= ontology.min_train_support]

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
        for r in sorted(labels + noisy, key=lambda r: (int(r["product_id"]), r["attribute"])):
            if (r["product_id"], r["attribute"]) in capped:
                counts[("train", r["attribute"], "capped")] += 1
                continue
            pid = int(r["product_id"])
            s = splits[pid]
            clean = r["included"] == "1"
            if s["split"] not in handles:
                handles[s["split"]] = open(args.out_dir / f"{s['split']}.jsonl", "w", encoding="utf-8")
            for condition in CONDITIONS:
                example = make_example(products[pid], ontology.attributes[r["attribute"]], r["label"],
                                       s["split"], int(s["family_id"]), condition, text_status=r["text_status"],
                                       clean=clean)
                handles[s["split"]].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[(s["split"], r["attribute"], "items" if clean else "noisy")] += 1
    finally:
        for handle in handles.values():
            handle.close()

    record = {"manifest": {name: manifest["files"][name] for name in sorted(manifest["files"])},
              "ontology_sha256": ontology.sha256, "source_fingerprint": manifest["source_fingerprint"],
              "train_cap": args.train_cap,
              "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(args.out_dir.glob("*.jsonl"))},
              "counts": {"/".join(key): n for key, n in sorted(counts.items())}}
    (args.out_dir / "build.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    for (split, attribute, kind), n in sorted(counts.items()):
        print(f"{split:>14}  {attribute:<14} {kind:<7} {n:>6,}" + ("  (x2 conditions)" if kind != "capped" else ""))


if __name__ == "__main__":
    main()
