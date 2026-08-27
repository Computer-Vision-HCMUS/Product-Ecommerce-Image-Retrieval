"""Build reproducible, retrieval-valid SCALE metadata and splits.

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
    """Split within labels: 70% train, 10% val, 10% query, 10% gallery.

    The final two partitions are the 20% held-out test pool.  Keeping query
    and gallery separate prevents query leakage and guarantees that every
    retained test label has at least one possible gallery positive.
    """
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = {}
    for product_id in ids:
        grouped.setdefault(labels[product_id], []).append(product_id)
    train, val, test, gallery = [], [], [], []
    for group_ids in grouped.values():
        rng.shuffle(group_ids)
        if len(group_ids) < 4:
            raise ValueError("make_splits received a label with fewer than four records")
        val_count = max(1, round(len(group_ids) * 0.1))
        test_count = max(1, round(len(group_ids) * 0.1))
        gallery_count = max(1, round(len(group_ids) * 0.1))
        train_count = len(group_ids) - val_count - test_count - gallery_count
        # The caller filters to >=4, but retain an explicit guard if rounding
        # changes in the future.
        if train_count < 1:
            train_count, val_count, test_count, gallery_count = 1, 1, 1, len(group_ids) - 3
        train.extend(group_ids[:train_count])
        val.extend(group_ids[train_count : train_count + val_count])
        test.extend(group_ids[train_count + val_count : train_count + val_count + test_count])
        gallery.extend(group_ids[train_count + val_count + test_count :])
    return {"train": train, "val": val, "test": test, "gallery": gallery}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-items-per-label", type=int, default=4,
                        help="Drop labels too small for train/val/query/gallery (minimum: 4).")
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Decode every image before inclusion (slow on a network/disk-mounted dataset).",
    )
    args = parser.parse_args()
    if args.min_items_per_label < 4:
        raise SystemExit("--min-items-per-label must be at least 4 for retrieval evaluation.")

    metadata = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = read_manifest(args.dataset_dir / "manifest.jsonl")
    records: dict[str, dict[str, Any]] = {}
    for product_id, item in metadata.items():
        row = manifest.get(product_id, {})
        # Respect the controlled modality cohort written by the downloader.
        # Raw media remains on disk for future ablations but must not leak into
        # this run when the modality was intentionally hidden.
        availability = row.get("modality_present") or item.get("modality_present") or {}
        video_available = availability.get("video_url", True)
        image_path = Path(str(row.get("image_path") or ""))
        if not image_path.is_file() or (args.verify_images and not readable_image(image_path)):
            continue
        video_path = row.get("video_path") if video_available else None
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
            "modality_source": str(row.get("modality_source") or item.get("modality_source") or "unknown"),
            "masked_modalities": list(row.get("masked_modalities") or item.get("masked_modalities") or []),
            "modality_present": availability,
        }

    if not records:
        raise SystemExit("No readable images found. Check manifest paths and dataset location.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_counts = Counter(row["label"] for row in records.values())
    dropped_labels = {label for label, count in label_counts.items() if count < args.min_items_per_label}
    records = {product_id: row for product_id, row in records.items() if row["label"] not in dropped_labels}
    if not records:
        raise SystemExit("No records remain after enforcing --min-items-per-label.")
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
    split_summary = {
        "seed": args.seed,
        "protocol": "70% train, 10% validation, 10% retrieval query, 10% retrieval gallery; split within label",
        "positive_definition": "same label, restricted to the gallery; custom subset evaluation, not official M5Product ground truth",
        "min_items_per_label": args.min_items_per_label,
        "dropped_label_count": len(dropped_labels),
        "dropped_record_count": sum(label_counts[label] for label in dropped_labels),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "cohort_counts": {
            name: dict(Counter(records[item_id].get("modality_source", "unknown") for item_id in ids))
            for name, ids in splits.items()
        },
    }
    (args.output_dir / "split_protocol.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(records)} readable images in {args.output_dir}; dropped {split_summary['dropped_record_count']} records from undersized labels")


if __name__ == "__main__":
    main()
