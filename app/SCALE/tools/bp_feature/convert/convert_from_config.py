"""Build SCALE LMDB from TSV region feature files."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from tensorpack.dataflow import LMDBSerializer, PrefetchDataZMQ, RNGDataFlow

FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features", "title"]
MIN_BOXES = 36
MAX_BOXES = 36
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TsvRegionFlow(RNGDataFlow):
    def __init__(self, tsv_files: list[Path], allowed_ids: set[str] | None = None):
        self.tsv_files = tsv_files
        self.allowed_ids = allowed_ids
        self._count = 0
        for tsv in tsv_files:
            with tsv.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t", fieldnames=FIELDNAMES)
                for row in reader:
                    if self.allowed_ids is None or row["image_id"] in self.allowed_ids:
                        self._count += 1

    def __len__(self):
        return self._count

    def __iter__(self):
        for tsv in self.tsv_files:
            with tsv.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t", fieldnames=FIELDNAMES)
                for item in reader:
                    image_id = item["image_id"]
                    if self.allowed_ids is not None and image_id not in self.allowed_ids:
                        continue
                    try:
                        image_h = int(item["image_h"])
                        image_w = int(item["image_w"])
                        num_boxes = MAX_BOXES
                        boxes = np.frombuffer(
                            base64.b64decode(item["boxes"]), dtype=np.float32
                        ).reshape(MAX_BOXES, 4)
                        features = np.frombuffer(
                            base64.b64decode(item["features"]), dtype=np.float32
                        ).reshape(MAX_BOXES, 2048)
                        caption = item["title"]
                    except Exception as exc:
                        print(f"skip {image_id}: {exc}")
                        continue
                    yield [features, boxes, num_boxes, image_h, image_w, image_id, caption]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv-dir", type=Path, required=True)
    parser.add_argument("--output-lmdb", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--tsv-name", type=str, default="features.tsv")
    args = parser.parse_args()

    tsv_files = sorted(args.tsv_dir.glob("*.tsv"))
    if args.tsv_name:
        single = args.tsv_dir / args.tsv_name
        if not single.is_file():
            raise FileNotFoundError(f"TSV not found: {single}")
        tsv_files = [single]
    elif not tsv_files:
        raise FileNotFoundError(f"No TSV files in {args.tsv_dir}")

    allowed_ids = None
    if args.ids_file and args.ids_file.is_file():
        ids = read_json(args.ids_file)
        if isinstance(ids, dict):
            ids = ids.get("ids", list(ids.keys()))
        allowed_ids = set(str(i) for i in ids)

    ds = TsvRegionFlow(tsv_files, allowed_ids)
    print(f"Building LMDB with {len(ds)} records -> {args.output_lmdb}")
    args.output_lmdb.parent.mkdir(parents=True, exist_ok=True)
    for path in (args.output_lmdb, Path(str(args.output_lmdb) + "-lock")):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
    if os.name == "nt":
        LMDBSerializer.save(ds, str(args.output_lmdb))
    else:
        LMDBSerializer.save(PrefetchDataZMQ(ds, 1), str(args.output_lmdb))


if __name__ == "__main__":
    main()
