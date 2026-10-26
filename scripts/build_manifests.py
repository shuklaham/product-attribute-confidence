"""Build the frozen split and label manifests from the style JSONs.

The manifests hold product IDs, splits and label decisions only, never product text, so they
can be committed. A fingerprint of the source files and the ontology hash are stored with
them, so a later change to the data or the label set is visible.

Usage:
    uv run python scripts/build_manifests.py <styles_dir> <out_dir>
"""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from pac import ontology as ontology_module
from pac.records import directory_fingerprint, load_products
from pac.splits import BUCKETS, MIN_PHOTO_LONG_SIDE, SALT, assign_splits, families, label_rows, usable_photos


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(styles_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ontology = ontology_module.load()
    products, failures = load_products(styles_dir)
    splits = assign_splits(products, families(products))
    photos_path = out / "photos.csv"
    photo_ok = None
    if photos_path.exists():  # written by prepare_images.py
        with open(photos_path, newline="", encoding="utf-8") as f:
            photo_ok = usable_photos(list(csv.DictReader(f)))
    rows = label_rows(products, splits, ontology, photo_ok)

    with open(out / "products.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "family_id", "category", "split"])
        for r in sorted(splits.values(), key=lambda r: r.product_id):
            writer.writerow([r.product_id, r.family_id, r.category, r.split])
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "attribute", "raw_value", "label", "text_status", "included", "reason"])
        for r in sorted(rows, key=lambda r: (r.product_id, r.attribute)):
            writer.writerow([r.product_id, r.attribute, r.raw_value, r.label, r.text_status, int(r.included), r.reason])

    included = Counter((r.attribute, splits[r.product_id].split) for r in rows if r.included)
    manifest = {
        "ontology_version": ontology.version,
        "ontology_sha256": ontology.sha256,
        "split_salt": SALT,
        "bucket_percent": dict(BUCKETS),
        "source_fingerprint": directory_fingerprint(styles_dir),
        "records": len(products),
        "unreadable_records": failures,
        "products_in_scope": len(splits),
        "photo_rule": f"decodes and long side >= {MIN_PHOTO_LONG_SIDE} px" if photo_ok is not None else "not applied",
        "included_labels": {f"{a}/{s}": n for (a, s), n in sorted(included.items())},
        "files": {name: sha256_file(out / name) for name in ("products.csv", "labels.csv", "photos.csv")
                  if (out / name).exists()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(splits):,} products in scope, {sum(included.values()):,} included labels")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
