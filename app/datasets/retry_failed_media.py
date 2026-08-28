"""Retry failed M5Product image/video downloads without changing selected IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"


def folder_name(super_category: str) -> str:
    readable = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", super_category).strip(" ._")[:80]
    suffix = hashlib.sha1(super_category.encode("utf-8")).hexdigest()[:8]
    return f"{readable or 'uncategorized'}__{suffix}"


def fetch(url: str, target: Path, timeout: int, retries: int) -> tuple[str | None, str]:
    if not url:
        return None, "missing_url"
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = "unknown"
    for attempt in range(retries + 1):
        temp_path: Path | None = None
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".part") as stream:
                    temp_path = Path(stream.name)
                    while block := response.read(1024 * 256):
                        stream.write(block)
            os.replace(temp_path, target)
            return str(target), "downloaded_retry"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if temp_path:
                temp_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    return None, f"error_retry: {last_error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--media", choices=("image", "video", "both"), default="both")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    jobs: list[tuple[int, str, str, Path]] = []
    for index, row in enumerate(rows):
        category = Path(str(row.get("image_path") or row.get("video_path") or "")).parent.parent
        if not category.name:
            # Failed downloads have no path; their media folders follow the
            # folder layout created by the initial downloader.
            category = args.dataset_dir / folder_name(str(row["super_category"]))
        if args.media in ("image", "both") and not str(row.get("image_status", "")).startswith("downloaded"):
            jobs.append((index, "image", str(row.get("source_image_url") or row.get("image_url") or ""), category / "images" / f"{row['id']}.jpg"))
        if args.media in ("video", "both") and not str(row.get("video_status", "")).startswith("downloaded"):
            jobs.append((index, "video", str(row.get("source_video_url") or row.get("video_url") or ""), category / "videos" / f"{row['id']}.mp4"))
    print(f"Retrying {len(jobs)} failed media downloads", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, url, target, args.timeout, args.retries): (index, kind) for index, kind, url, target in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            index, kind = futures[future]
            path, status = future.result()
            rows[index][f"{kind}_path"] = path
            rows[index][f"{kind}_status"] = status
            if completed % 25 == 0 or completed == len(jobs):
                print(f"Retried {completed}/{len(jobs)}", flush=True)
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Updated {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
