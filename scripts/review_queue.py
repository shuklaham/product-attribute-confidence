"""Prepare the hand-check queue for review: a text sheet and photo contact sheets.

The queue is the frozen sample in annotations/hand_check_queue_v0.csv. For each queued product
it writes the catalog label, the label the description names, and the
sentence that names it to <out_dir>/queue.csv. It also writes contact sheets of the photos,
six per sheet and numbered to match the CSV, to <out_dir>/sheet_NN.jpg. Output stays under
data/ (gitignored) because it contains product text and photos.

Usage:
    uv run python -I scripts/review_queue.py <styles_dir> <manifest_dir> <image_dir> <out_dir> [queue.csv]
"""

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from pac import ontology as ontology_module
from pac.records import load_products
from pac.text import mask_brand

TILE_W, TILE_H, CAPTION = 384, 512, 30
COLUMNS, ROWS = 3, 2


def snippet(text: str, attribute, label: str, width: int = 70) -> str:
    """The text around the first phrase of another label that the description names."""
    masked = mask_brand(text, None)
    for kept in attribute.labels:
        if kept.name == label:
            continue
        for pattern in kept.patterns:
            match = pattern.search(masked)
            if match:
                start, end = max(0, match.start() - width), min(len(text), match.end() + width)
                return f"…{text[start:end]}…"
    return ""


def main(styles_dir: str, manifest_dir: str, image_dir: str, out_dir: str,
         queue_csv: str = "annotations/hand_check_queue_v0.csv") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ontology = ontology_module.load()
    products = {p.id: p for p in load_products(styles_dir)[0]}
    with open(Path(manifest_dir) / "products.csv", newline="", encoding="utf-8") as f:
        splits = {int(r["product_id"]): r for r in csv.DictReader(f)}
    with open(Path(manifest_dir) / "labels.csv", newline="", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    rows = [dict(r, split=splits[int(r["product_id"])]["split"]) for r in labels]
    by_key = {(int(r["product_id"]), r["attribute"]): r for r in rows}
    with open(queue_csv, newline="", encoding="utf-8") as f:
        queue = [(int(r["product_id"]), r["attribute"]) for r in csv.DictReader(f)]

    with open(out / "queue.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "product_id", "attribute", "category", "split", "catalog_label", "text_names", "title",
                         "snippet"])
        for k, (pid, key) in enumerate(queue, start=1):
            p, r, attribute = products[pid], by_key[(pid, key)], ontology.attributes[key]
            text = mask_brand(" ".join((p.title, p.description)), p.brand)
            named = sorted(attribute.named_labels(text) - {r["label"]})
            writer.writerow([k, pid, key, splits[pid]["category"], r["split"], r["label"], ", ".join(named), p.title,
                             snippet(" ".join((p.title, p.description)), attribute, r["label"])])

    per_sheet = COLUMNS * ROWS
    for sheet_index in range(0, len(queue), per_sheet):
        sheet = Image.new("RGB", (COLUMNS * TILE_W, ROWS * (TILE_H + CAPTION)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (pid, key) in enumerate(queue[sheet_index:sheet_index + per_sheet]):
            x, y = (i % COLUMNS) * TILE_W, (i // COLUMNS) * (TILE_H + CAPTION)
            photo = Path(image_dir) / f"{pid}.jpg"
            if photo.exists():
                with Image.open(photo) as image:
                    image.thumbnail((TILE_W, TILE_H))
                    sheet.paste(image, (x + (TILE_W - image.width) // 2, y + CAPTION))
            draw.text((x + 8, y + 8), f"#{sheet_index + i + 1}  {pid}  {key}", fill="black")
        sheet.save(out / f"sheet_{sheet_index // per_sheet + 1:02d}.jpg", quality=88)
    print(f"wrote {out}: {len(queue)} queued items, {-(-len(queue) // per_sheet)} contact sheets")


if __name__ == "__main__":
    main(*sys.argv[1:6])
