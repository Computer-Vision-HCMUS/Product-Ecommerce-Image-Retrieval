"""Build reproducible SCALE metadata and splits from downloaded_2k.

The subset has no official M5Product query/gallery ground truth.  This script
therefore defines positives by product label and writes that decision into each
split manifest so evaluation cannot be mistaken for the official benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def serialize_pv(value: str) -> str:
    """Make the proposal's key/value boundaries explicit for the BERT encoder."""
    entities: list[str] = []
    for pair in value.split("#;#"):
        if "#:#" not in pair:
            continue
        key, raw_value = pair.split("#:#", 1)
        key, raw_value = key.strip(), raw_value.strip()
        if key and raw_value:
            entities.append(f"[ENT] {key} [VAL] {raw_value} [SEP]")
    return " ".join(entities)


def make_splits(ids: list[str], labels: dict[str, str], seed: int) -> dict[str, list[str]]:
    """Split within label groups, preserving labels for custom positive matching."""
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = {}
    for product_id in ids:
        grouped.setdefault(labels[product_id], []).append(product_id)
    train, gallery, query = [], [], []
    for group_ids in grouped.values():
        rng.shuffle(group_ids)
        if len(group_ids) == 1:
            train.extend(group_ids)
            continue
        query_count = max(1, round(len(group_ids) * 0.1))
        gallery_count = max(1, round(len(group_ids) * 0.2))
        query.extend(group_ids[:query_count])
        gallery.extend(group_ids[query_count : query_count + gallery_count])
        train.extend(group_ids[query_count + gallery_count :])
    return {"train": train, "gallery": gallery, "query": query}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Decode every image before inclusion (slow on a network/disk-mounted dataset).",
    )
    args = parser.parse_args()

    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = read_manifest(args.dataset_dir / "manifest.jsonl")
    records: dict[str, dict[str, Any]] = {}
    for product_id, item in metadata.items():
        row = manifest.get(product_id, {})
        image_path = Path(str(row.get("image_path") or ""))
        if not image_path.is_file() or (args.verify_images and not readable_image(image_path)):
            continue
        video_path = row.get("video_path")
        records[product_id] = {
            "id": product_id,
            "title": str(item.get("title", "")).strip(),
            "label": str(item.get("label", "")).strip() or "__unknown__",
            "pv": str(item.get("pv", "")).strip(),
            "table_serialized": serialize_pv(str(item.get("pv", ""))),
            "image_path": str(image_path.resolve()),
            "video_path": str(video_path) if video_path and Path(video_path).is_file() else None,
            "has_video": bool(video_path and Path(video_path).is_file()),
            "description": "",
        }

    if not records:
        raise SystemExit("No readable images found. Check manifest paths and dataset location.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = {product_id: row["label"] for product_id, row in records.items()}
    splits = make_splits(sorted(records), labels, args.seed)
    (args.output_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, ids in splits.items():
        payload = {
            "split": name,
            "seed": args.seed,
            "positive_definition": "same label (custom subset evaluation; not official M5Product ground truth)",
            "ids": ids,
        }
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (args.output_dir / "labels.json").write_text(
        json.dumps(sorted(Counter(labels.values())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(records)} readable images in {args.output_dir}")


if __name__ == "__main__":
    main()
