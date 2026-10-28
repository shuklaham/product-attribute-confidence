"""Verify and resize the photos of in-scope products.

Reads each photo straight from the Kaggle zip (no full extraction), checks that it decodes,
and writes a copy whose long side is at most 512 px to <out_dir>/<id>.jpg (smaller photos are
not enlarged). The status, original size and sha256 of every original photo go to
<manifest_dir>/photos.csv, which build_manifests.py uses to exclude products without a usable
photo and to put byte-identical photos in one family. Missing and unreadable photos are listed
in the report.

Usage:
    uv run python -I scripts/prepare_images.py <kaggle_zip> <manifest_dir> <out_dir> <report.md>
"""

import collections
import csv
import hashlib
import io
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

LONG_SIDE = 512
PREFIX = "fashion-dataset/images/"


def _process(job: tuple[str, list[int], str]) -> list[tuple[int, str, tuple[int, int] | None, str]]:
    zip_path, ids, out_dir = job
    results = []
    with zipfile.ZipFile(zip_path) as archive:
        for pid in ids:
            try:
                data = archive.read(f"{PREFIX}{pid}.jpg")
            except KeyError:
                results.append((pid, "missing", None, ""))
                continue
            digest = hashlib.sha256(data).hexdigest()
            try:
                with Image.open(io.BytesIO(data)) as image:
                    size = image.size
                    image.draft("RGB", (LONG_SIDE, LONG_SIDE))
                    image = image.convert("RGB")
                    image.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
                    image.save(Path(out_dir) / f"{pid}.jpg", "JPEG", quality=90)
                results.append((pid, "ok", size, digest))
            except (OSError, ValueError):
                results.append((pid, "unreadable", None, digest))
    return results


def main(zip_path: str, manifest_dir: str, out_dir: str, report_path: str) -> None:
    with open(Path(manifest_dir) / "products.csv", newline="", encoding="utf-8") as f:
        ids = sorted(int(r["product_id"]) for r in csv.DictReader(f))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    workers = os.cpu_count() or 4
    chunks = [ids[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(workers) as pool:
        results = [r for chunk in pool.map(_process, [(zip_path, c, out_dir) for c in chunks]) for r in chunk]

    with open(Path(manifest_dir) / "photos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["product_id", "status", "width", "height", "sha256"])
        for pid, s, size, digest in sorted(results):
            writer.writerow([pid, s, *(size or ("", "")), digest])

    status = collections.Counter(s for _, s, _, _ in results)
    sizes = collections.Counter(size for _, s, size, _ in results if s == "ok")
    problems = sorted((pid, s) for pid, s, _, _ in results if s != "ok")
    shared = collections.Counter(digest for _, s, _, digest in results if s == "ok")
    lines = ["# Image check", "",
             (f"In-scope products: {len(ids):,}. Photos written: {status['ok']:,}. "
              f"Missing from the zip: {status['missing']}. Unreadable: {status['unreadable']}."), "",
             (f"Each photo is resized so its long side is at most {LONG_SIDE} px and saved as JPEG quality 90; "
              "smaller photos are not enlarged."), "",
             (f"Byte-identical photos: {sum(n for n in shared.values() if n > 1):,} products share a photo file "
              f"with another product ({sum(1 for n in shared.values() if n > 1):,} groups). build_manifests.py "
              "puts each group in one family."), "",
             "Most common original sizes: " + ", ".join(f"{w}×{h} ({n:,})" for (w, h), n in sizes.most_common(5)), ""]
    if problems:
        lines += ["Products without a usable photo: " + ", ".join(f"{pid} ({s})" for pid, s in problems), ""]
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:5])
