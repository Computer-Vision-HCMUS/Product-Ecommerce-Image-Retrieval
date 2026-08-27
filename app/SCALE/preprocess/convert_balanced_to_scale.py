"""Convert downloaded_m5product_balanced into SCALE paper layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True,
                        help="Directory with train.json, val.json, test.json, gallery.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-media", action="store_true",
                        help="Copy images/videos into output dir (default: write path manifest only)")
    args = parser.parse_args()

    metadata_path = args.dataset_dir / "metadata.json"
    manifest_path = args.dataset_dir / "manifest.jsonl"
    metadata = read_json(metadata_path)
    manifest = load_manifest(manifest_path)

    # Only materialize records actually retained by the split. This prevents
    # failed downloads and labels dropped by prepare_subset from reaching
    # feature extraction or training.
    included_ids: set[str] = set()
    for split_name in ("train", "val", "test", "gallery"):
        split_file = args.splits_dir / f"{split_name}.json"
        if split_file.is_file():
            split_data = read_json(split_file)
            included_ids.update(str(product_id) for product_id in split_data.get("ids", split_data))
    if not included_ids:
        raise SystemExit("No split IDs found. Run prepare_subset.py before conversion.")

    id_label: dict[str, dict[str, str]] = {}
    path_manifest: dict[str, dict[str, str | None]] = {}
    labels: set[str] = set()

    for product_id, meta in metadata.items():
        if product_id not in included_ids:
            continue
        row = manifest.get(product_id, meta)
        # The downloader retains raw media for future ablations, but this
        # conversion must expose only the modality view selected for this run.
        # Older datasets without this field remain backward compatible.
        availability = row.get("modality_present") or meta.get("modality_present") or {}
        title_available = availability.get("title", True)
        pv_available = availability.get("pv", True)
        video_available = availability.get("video_url", True)
        audio_available = availability.get("audio", True)
        title = str(row.get("title") or meta.get("title") or "").strip()
        pv = str(row.get("pv") or meta.get("pv") or "").strip()
        label = str(row.get("label") or meta.get("label") or "").strip()
        super_category = str(row.get("super_category") or meta.get("super_category") or "").strip()
        if not title_available:
            title = ""
        if not pv_available:
            pv = ""
        if not title and not label:
            continue
        raw_video_path = row.get("video_path")
        video_path = str(raw_video_path) if video_available and raw_video_path and Path(raw_video_path).is_file() else None
        id_label[product_id] = {
            "title": title,
            "pv": pv,
            "label": label,
            "super_category": super_category,
            "modality_source": str(row.get("modality_source") or meta.get("modality_source") or "unknown"),
            "masked_modalities": list(row.get("masked_modalities") or meta.get("masked_modalities") or []),
            "audio_available": bool(audio_available),
        }
        labels.add(label)
        path_manifest[product_id] = {
            "image_path": row.get("image_path"),
            "video_path": video_path,
        }

    out = args.output_dir
    dirs = [
        out / "tsv_features",
        out / "video_feature",
        out / "audio_feature",
        out / "audios",
        out / "lmdb_features",
        out / "checkpoints",
        out / "features",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    write_json(out / "id_label.json", id_label)
    write_json(out / "label_list.json", sorted(labels))
    write_json(out / "path_manifest.json", path_manifest)

    for split_name in ("train", "val", "test", "gallery"):
        split_file = args.splits_dir / f"{split_name}.json"
        if split_file.is_file():
            split_data = read_json(split_file)
            ids = split_data.get("ids", split_data)
            write_json(out / f"{split_name}_ids.json", ids)

    if args.copy_media:
        images_out = out / "images"
        videos_out = out / "videos"
        images_out.mkdir(exist_ok=True)
        videos_out.mkdir(exist_ok=True)
        for product_id, paths in path_manifest.items():
            img = paths.get("image_path")
            if img and Path(img).is_file():
                dest = images_out / f"{product_id}.jpg"
                if not dest.is_file():
                    shutil.copy2(img, dest)
            vid = paths.get("video_path")
            if vid and Path(vid).is_file():
                dest = videos_out / f"{product_id}.mp4"
                if not dest.is_file():
                    shutil.copy2(vid, dest)

    print(f"id_label: {len(id_label)} products")
    print(f"labels: {len(labels)}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
