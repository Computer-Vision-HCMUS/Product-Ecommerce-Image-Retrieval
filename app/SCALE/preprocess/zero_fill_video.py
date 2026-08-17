"""Zero-fill video features only for products without a local video file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VIDEO_LEN = 12
VIDEO_DIM = 1024


def resolve_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    repo_path = Path(__file__).resolve().parents[3] / raw_path
    return repo_path if repo_path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.path_manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for product_id, paths in manifest.items():
        if resolve_path(paths.get("video_path")) is not None:
            continue  # has video — do not zero-fill
        out = args.output_dir / f"{product_id}.npy"
        if not out.is_file():
            np.save(out, np.zeros((VIDEO_LEN, VIDEO_DIM), dtype=np.float32))
            created += 1
    print(f"zero-filled {created} products without video (skipped those with video_path)")


if __name__ == "__main__":
    main()
