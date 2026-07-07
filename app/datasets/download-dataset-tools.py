"""Download a small media subset from the M5Product metadata JSON.

The source JSON is a large top-level object:

{
  "<product_id>": {
    "title": "...",
    "label": "...",
    "url": "https://...jpg",
    "video": "https://...mp4",
    "pv": "key#:#value#;#..."
  }
}

This script streams the first N products instead of loading the whole file into
memory, then downloads image/video files into a local dataset layout:

downloaded_2k/
  images/<product_id>.<ext>
  videos/<product_id>.<ext>
  metadata.json
  manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_INPUT = Path(__file__).with_name("product1m_product5m_id_label.json")
DEFAULT_OUTPUT = Path(__file__).with_name("downloaded_2k")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)


@dataclass(frozen=True)
class ProductRecord:
    product_id: str
    title: str
    label: str
    image_url: str
    video_url: str
    pv: str


def iter_top_level_object(path: Path, limit: int, start: int = 0) -> Iterator[ProductRecord]:
    """Yield product records from a huge top-level JSON object."""
    decoder = json.JSONDecoder()
    emitted = 0
    seen = 0
    buffer = ""
    pos = 0

    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk

            while True:
                pos = _skip_ws_and_commas(buffer, pos)
                if pos >= len(buffer):
                    break
                if buffer[pos] == "{":
                    pos += 1
                    continue
                if buffer[pos] == "}":
                    return

                try:
                    product_id, key_end = decoder.raw_decode(buffer, pos)
                    colon = _skip_ws(buffer, key_end)
                    if colon >= len(buffer) or buffer[colon] != ":":
                        break
                    value_start = _skip_ws(buffer, colon + 1)
                    data, value_end = decoder.raw_decode(buffer, value_start)
                except json.JSONDecodeError:
                    break

                seen += 1
                pos = value_end
                if seen <= start:
                    continue

                emitted += 1
                yield ProductRecord(
                    product_id=str(product_id),
                    title=str(data.get("title", "")),
                    label=str(data.get("label", "")),
                    image_url=str(data.get("url", "")),
                    video_url=str(data.get("video", "")),
                    pv=str(data.get("pv", "")),
                )
                if emitted >= limit:
                    return

            if pos:
                buffer = buffer[pos:]
                pos = 0


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _skip_ws_and_commas(text: str, pos: int) -> int:
    while pos < len(text) and (text[pos].isspace() or text[pos] == ","):
        pos += 1
    return pos


def guess_extension(url: str, content_type: Optional[str], fallback: str) -> str:
    parsed = urlparse(url)
    suffix = Path(unquote(posixpath.basename(parsed.path))).suffix.lower()
    if suffix and len(suffix) <= 8:
        return suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed
    return fallback


def normalize_url(url: str) -> str:
    url = url.strip()
    while url.startswith("#"):
        url = url[1:].lstrip()
    if url.startswith("//"):
        url = "https:" + url
    return url


def download_url(
    url: str,
    output_stem: Path,
    fallback_ext: str,
    timeout: int,
    retries: int,
    overwrite: bool,
) -> Tuple[Optional[str], str]:
    url = normalize_url(url)
    if not url:
        return None, "missing"

    parsed_ext = guess_extension(url, None, fallback_ext)
    candidate_path = output_stem.with_suffix(parsed_ext)
    if candidate_path.exists() and not overwrite:
        return str(candidate_path), "exists"

    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://www.taobao.com/"})
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type")
                ext = guess_extension(url, content_type, fallback_ext)
                final_path = output_stem.with_suffix(ext)
                if final_path.exists() and not overwrite:
                    return str(final_path), "exists"

                tmp_path = final_path.with_suffix(final_path.suffix + ".part")
                with tmp_path.open("wb") as fh:
                    while True:
                        block = response.read(1024 * 256)
                        if not block:
                            break
                        fh.write(block)
                os.replace(tmp_path, final_path)
                return str(final_path), "downloaded"
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 * attempt, 8))

    return None, f"error: {last_error}"


def download_product(
    record: ProductRecord,
    images_dir: Path,
    videos_dir: Path,
    timeout: int,
    retries: int,
    overwrite: bool,
) -> Dict[str, object]:
    image_path, image_status = download_url(
        record.image_url,
        images_dir / record.product_id,
        ".jpg",
        timeout,
        retries,
        overwrite,
    )
    video_path, video_status = download_url(
        record.video_url,
        videos_dir / record.product_id,
        ".mp4",
        timeout,
        retries,
        overwrite,
    )

    return {
        "id": record.product_id,
        "title": record.title,
        "label": record.label,
        "pv": record.pv,
        "image_url": record.image_url,
        "video_url": record.video_url,
        "image_path": image_path,
        "video_path": video_path,
        "image_status": image_status,
        "video_status": video_status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download image/video media for a subset of M5Product.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to product1m_product5m_id_label.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dataset directory.")
    parser.add_argument("--limit", type=int, default=2000, help="Number of products to process.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many products before downloading.")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent download workers.")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per media URL after the first attempt.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files that already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_top_level_object(input_path, limit=args.limit, start=args.start))
    metadata = {
        record.product_id: {
            "title": record.title,
            "label": record.label,
            "url": record.image_url,
            "video": record.video_url,
            "pv": record.pv,
        }
        for record in records
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Loaded {len(records)} products from {input_path}", flush=True)
    print(f"Dataset output: {output_dir}", flush=True)

    done = 0
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    download_product,
                    record,
                    images_dir,
                    videos_dir,
                    args.timeout,
                    args.retries,
                    args.overwrite,
                )
                for record in records
            ]
            for future in as_completed(futures):
                row = future.result()
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest.flush()
                done += 1
                if done % 50 == 0 or done == len(records):
                    print(f"Processed {done}/{len(records)} products", flush=True)

    print(f"Wrote manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
