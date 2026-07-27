"""Build and download a balanced, high-quality M5Product subset.

The metadata file is streamed three times, so it is safe to use with the full
M5Product JSON object.  Categories are selected in a round-robin manner across
human-curated super-categories, rather than globally by frequency.  This avoids
one domain (for example facial skincare) dominating the subset.  The default
selection has 30 head, 40 medium, 20 tail, and 10 rare categories; each
selected category contributes up to 200 products.

Products are ranked with:
    Score = .4 * Completeness + .3 * TextQuality + .2 * MerchantScore
            + .1 * Diversity

The first three components are intrinsic product quality.  Diversity rewards
distinct titles, image hosts, and merchants while greedily selecting a
category's final products; this avoids downloading 200 near-duplicate listings.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import mimetypes
import os
import posixpath
import re
import socket
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_INPUT = Path(__file__).with_name("product1m_product5m_id_label.json")
DEFAULT_OUTPUT = Path(__file__).with_name("downloaded_m5product_balanced")
DEFAULT_TAXONOMY = Path(__file__).with_name("m5product_label_taxonomy.csv")
DEFAULT_CATEGORY_COUNTS = Path(__file__).with_name("m5product_category_counts.csv")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
MERCHANT_KEYS = ("merchant", "seller", "shop", "store", "vendor")


@dataclass(frozen=True)
class ProductRecord:
    product_id: str
    title: str
    label: str
    image_url: str
    video_url: str
    pv: str


@dataclass(frozen=True)
class ScoredRecord:
    record: ProductRecord
    completeness: float
    text_quality: float
    merchant_score: float
    base_score: float
    title_tokens: frozenset[str]
    merchant: str
    image_host: str


def iter_top_level_object(path: Path) -> Iterator[ProductRecord]:
    """Yield product records from the huge top-level JSON object."""
    decoder = json.JSONDecoder()
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
                    data, value_end = decoder.raw_decode(buffer, _skip_ws(buffer, colon + 1))
                except json.JSONDecodeError:
                    break
                pos = value_end
                yield ProductRecord(str(product_id), str(data.get("title", "")), str(data.get("label", "")),
                                    str(data.get("url", "")), str(data.get("video", "")), str(data.get("pv", "")))
            if pos:
                buffer, pos = buffer[pos:], 0


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _skip_ws_and_commas(text: str, pos: int) -> int:
    while pos < len(text) and (text[pos].isspace() or text[pos] == ","):
        pos += 1
    return pos


def nonempty(value: str) -> bool:
    return bool(value and value.strip() and value.strip().lower() not in {"null", "none", "nan"})


def parse_pv(pv: str) -> Dict[str, str]:
    """Parse M5Product's `key#:#value#;#...` property-value format."""
    values: Dict[str, str] = {}
    for part in pv.split("#;#"):
        if "#:#" in part:
            key, value = part.split("#:#", 1)
            values[key.strip().lower()] = value.strip()
    return values


def merchant_value(properties: Dict[str, str]) -> str:
    for key, value in properties.items():
        if any(marker in key for marker in MERCHANT_KEYS) and nonempty(value):
            return value.casefold()
    return ""


def score_record(record: ProductRecord) -> ScoredRecord:
    properties = parse_pv(record.pv)
    title_tokens = frozenset(token.casefold() for token in TOKEN_RE.findall(record.title))
    completeness = sum(map(nonempty, (record.title, record.label, record.image_url, record.video_url, record.pv))) / 5
    # A useful listing has a non-trivial, non-repetitive title (ideal: 4--20 tokens).
    token_count = len(title_tokens)
    length_score = min(1.0, len(record.title.strip()) / 30)
    token_score = min(1.0, token_count / 4)
    repetition_score = min(1.0, token_count / max(1, len(TOKEN_RE.findall(record.title))))
    text_quality = (length_score + token_score + repetition_score) / 3 if title_tokens else 0.0
    merchant = merchant_value(properties)
    numeric_signal = any(
        any(term in key for term in ("rating", "score", "sale", "sold", "credit")) and any(char.isdigit() for char in value)
        for key, value in properties.items()
    )
    merchant_score = min(1.0, 0.75 * bool(merchant) + 0.25 * numeric_signal)
    base_score = 0.4 * completeness + 0.3 * text_quality + 0.2 * merchant_score
    return ScoredRecord(record, completeness, text_quality, merchant_score, base_score, title_tokens,
                        merchant, urlparse(normalize_url(record.image_url)).netloc.casefold())


def load_taxonomy(path: Path) -> Dict[str, str]:
    """Load a curated mapping with columns `label` and `super_category`.

    `sieu_danh_muc_tieng_viet` is also accepted as the super-category column,
    making it convenient to use a Vietnamese taxonomy.
    """
    if not path.is_file():
        raise SystemExit(
            f"Taxonomy file not found: {path}. Create a UTF-8 CSV with columns "
            "`label,super_category` (or `label,sieu_danh_muc_tieng_viet`)."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"{path} must contain a `label` column.")
        super_column = next((name for name in ("super_category", "sieu_danh_muc_tieng_viet") if name in reader.fieldnames), None)
        if not super_column:
            raise SystemExit(f"{path} must contain `super_category` or `sieu_danh_muc_tieng_viet`.")
        return {
            row["label"].strip(): row[super_column].strip()
            for row in reader
            if nonempty(row.get("label", "")) and nonempty(row.get(super_column, ""))
        }


def _frequency_band(index: int, total: int) -> str:
    """Return a category's frequency quartile within its super-category."""
    return ("head", "medium", "tail", "rare")[min(3, 4 * index // total)]


def _round_robin_pick(
    grouped: Dict[str, list[Tuple[str, int]]],
    quota: int,
    used_per_super: Counter[str],
    maximum_per_super: int,
) -> list[Tuple[str, int, str]]:
    """Take the best available label from each super-category in turn."""
    picked: list[Tuple[str, int, str]] = []
    while len(picked) < quota:
        progressed = False
        for super_category in sorted(grouped):
            if len(picked) == quota:
                break
            if used_per_super[super_category] >= maximum_per_super or not grouped[super_category]:
                continue
            label, count = grouped[super_category].pop(0)
            picked.append((label, count, super_category))
            used_per_super[super_category] += 1
            progressed = True
        if not progressed:
            break
    return picked


def select_categories(
    counts: Counter[str], taxonomy: Dict[str, str], quotas: Tuple[int, int, int, int],
    min_samples: int, maximum_per_super: int,
) -> Tuple[Dict[str, str], list[Dict[str, object]]]:
    """Choose frequency bands within each super-category, then round-robin."""
    candidates: Dict[str, list[Tuple[str, int]]] = defaultdict(list)
    for label, count in counts.items():
        super_category = taxonomy.get(label)
        if super_category and count >= min_samples:
            candidates[super_category].append((label, count))
    if not candidates:
        raise SystemExit("No taxonomy labels have enough samples. Check the label spelling and --min-category-samples.")
    banded: Dict[str, Dict[str, list[Tuple[str, int]]]] = {band: defaultdict(list) for band in ("head", "medium", "tail", "rare")}
    for super_category, labels in candidates.items():
        labels.sort(key=lambda item: (-item[1], item[0]))
        for index, entry in enumerate(labels):
            banded[_frequency_band(index, len(labels))][super_category].append(entry)
    names = ("head", "medium", "tail", "rare")
    tier_by_label: Dict[str, str] = {}
    category_rows: list[Dict[str, object]] = []
    used_per_super: Counter[str] = Counter()
    for tier, quota in zip(names, quotas):
        picked = _round_robin_pick(banded[tier], quota, used_per_super, maximum_per_super)
        if len(picked) != quota:
            raise SystemExit(
                f"Could select {quota} {tier} categories with the current taxonomy/caps; got {len(picked)}. "
                "Add taxonomy labels or increase --max-labels-per-super-category."
            )
        for label, count, super_category in picked:
            tier_by_label[label] = tier
            category_rows.append({"label": label, "count": count, "tier": tier, "super_category": super_category})
    return tier_by_label, category_rows


def diversity(candidate: ScoredRecord, selected: list[ScoredRecord], title_counts: Counter[str], merchants: set[str], hosts: set[str]) -> float:
    if not selected:
        return 1.0
    # Novel title terms are most important; merchant and host add catalog-source variety.
    token_novelty = sum(token not in title_counts for token in candidate.title_tokens) / max(1, len(candidate.title_tokens))
    merchant_novelty = float(bool(candidate.merchant and candidate.merchant not in merchants))
    host_novelty = float(bool(candidate.image_host and candidate.image_host not in hosts))
    return 0.6 * token_novelty + 0.25 * merchant_novelty + 0.15 * host_novelty


def select_products(candidates: Dict[str, list[ScoredRecord]], per_category: int) -> list[Tuple[ScoredRecord, float, float]]:
    selected: list[Tuple[ScoredRecord, float, float]] = []
    for label in sorted(candidates):
        remaining = candidates[label][:]
        chosen: list[ScoredRecord] = []
        token_counts: Counter[str] = Counter()
        merchants: set[str] = set()
        hosts: set[str] = set()
        while remaining and len(chosen) < per_category:
            best = max(remaining, key=lambda item: (item.base_score + 0.1 * diversity(item, chosen, token_counts, merchants, hosts), item.record.product_id))
            div = diversity(best, chosen, token_counts, merchants, hosts)
            score = best.base_score + 0.1 * div
            selected.append((best, score, div))
            chosen.append(best)
            token_counts.update(best.title_tokens)
            if best.merchant:
                merchants.add(best.merchant)
            if best.image_host:
                hosts.add(best.image_host)
            remaining.remove(best)
    return selected


def normalize_url(url: str) -> str:
    url = url.strip()
    while url.startswith("#"):
        url = url[1:].lstrip()
    return "https:" + url if url.startswith("//") else url


def guess_extension(url: str, content_type: Optional[str], fallback: str) -> str:
    suffix = Path(unquote(posixpath.basename(urlparse(url).path))).suffix.lower()
    if suffix and len(suffix) <= 8:
        return suffix
    if content_type:
        return mimetypes.guess_extension(content_type.split(";")[0].strip()) or fallback
    return fallback


def download_url(url: str, output_stem: Path, fallback_ext: str, timeout: int, retries: int, overwrite: bool) -> Tuple[Optional[str], str]:
    url = normalize_url(url)
    if not url:
        return None, "missing"
    candidate_path = output_stem.with_suffix(guess_extension(url, None, fallback_ext))
    if candidate_path.exists() and not overwrite:
        return str(candidate_path), "exists"
    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://www.taobao.com/"})
            with urlopen(request, timeout=timeout) as response:
                final_path = output_stem.with_suffix(guess_extension(url, response.headers.get("Content-Type"), fallback_ext))
                if final_path.exists() and not overwrite:
                    return str(final_path), "exists"
                tmp_path = final_path.with_suffix(final_path.suffix + ".part")
                with tmp_path.open("wb") as fh:
                    while block := response.read(1024 * 256):
                        fh.write(block)
                os.replace(tmp_path, final_path)
                return str(final_path), "downloaded"
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 * attempt, 8))
    return None, f"error: {last_error}"


def download_product(scored: ScoredRecord, score: float, diversity_score: float, tier: str, images_dir: Path, videos_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    record = scored.record
    image_path, image_status = download_url(record.image_url, images_dir / record.product_id, ".jpg", args.timeout, args.retries, args.overwrite)
    video_path, video_status = download_url(record.video_url, videos_dir / record.product_id, ".mp4", args.timeout, args.retries, args.overwrite)
    return {"id": record.product_id, "title": record.title, "label": record.label, "tier": tier, "pv": record.pv,
            "image_url": record.image_url, "video_url": record.video_url, "score": round(score, 6),
            "score_components": {"completeness": round(scored.completeness, 6), "text_quality": round(scored.text_quality, 6), "merchant_score": round(scored.merchant_score, 6), "diversity": round(diversity_score, 6)},
            "image_path": image_path, "video_path": video_path, "image_status": image_status, "video_status": video_status}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a balanced, quality-ranked M5Product subset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY,
                        help="UTF-8 CSV mapping `label` to `super_category` (or `sieu_danh_muc_tieng_viet`).")
    parser.add_argument("--category-counts-output", type=Path, default=DEFAULT_CATEGORY_COUNTS,
                        help="CSV statistics for every M5Product label.")
    parser.add_argument("--category-counts-only", action="store_true",
                        help="Only scan metadata and write --category-counts-output; do not select or download.")
    parser.add_argument("--per-category", type=int, default=200, help="Maximum selected products per category.")
    parser.add_argument("--candidate-pool", type=int, default=1000, help="Best pre-diversification candidates retained per category.")
    parser.add_argument("--min-category-samples", type=int, default=None,
                        help="Minimum metadata rows per label; defaults to --per-category.")
    parser.add_argument("--max-labels-per-super-category", type=int, default=3,
                        help="Maximum selected labels from a super-category across all frequency bands.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_category <= 0 or args.candidate_pool < args.per_category:
        raise SystemExit("--per-category must be positive and --candidate-pool must be at least --per-category.")
    min_samples = args.min_category_samples or args.per_category
    if min_samples < args.per_category or args.max_labels_per_super_category <= 0:
        raise SystemExit("--min-category-samples must be at least --per-category and the super-category cap must be positive.")
    input_path, output_dir = args.input.resolve(), args.output.resolve()
    category_counts_path = args.category_counts_output.resolve()
    print("Pass 1/3: counting category frequencies...", flush=True)
    counts = Counter(record.label for record in iter_top_level_object(input_path) if nonempty(record.label))
    category_counts_path.parent.mkdir(parents=True, exist_ok=True)
    if args.category_counts_only:
        with category_counts_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=("rank", "label", "sample_count"))
            writer.writeheader()
            for rank, (label, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), 1):
                writer.writerow({"rank": rank, "label": label, "sample_count": count})
        print(f"Wrote category statistics: {category_counts_path} ({len(counts)} categories)", flush=True)
        return 0
    taxonomy = load_taxonomy(args.taxonomy.resolve())
    tiers, category_rows = select_categories(counts, taxonomy, (30, 40, 20, 10), min_samples, args.max_labels_per_super_category)
    output_dir.mkdir(parents=True, exist_ok=True)
    with category_counts_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=("rank", "label", "sample_count", "selected", "tier", "super_category"))
        writer.writeheader()
        for rank, (label, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), 1):
            writer.writerow({"rank": rank, "label": label, "sample_count": count,
                             "selected": label in tiers, "tier": tiers.get(label, ""), "super_category": taxonomy.get(label, "")})
    print(f"Wrote category statistics: {category_counts_path} ({len(counts)} categories)", flush=True)
    print("Pass 2/3: scoring products in the selected 100 categories...", flush=True)
    candidate_heaps: Dict[str, list[Tuple[float, str, ScoredRecord]]] = defaultdict(list)
    for record in iter_top_level_object(input_path):
        if record.label in tiers:
            scored = score_record(record)
            heap = candidate_heaps[record.label]
            entry = (scored.base_score, scored.record.product_id, scored)
            if len(heap) < args.candidate_pool:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    # Keep the strongest quality pool before applying greedy diversity selection.
    candidates = {
        label: [entry[2] for entry in sorted(heap, key=lambda entry: (-entry[0], entry[1]))]
        for label, heap in candidate_heaps.items()
    }
    selected = select_products(candidates, args.per_category)
    for row in category_rows:
        row["selected"] = sum(item.record.label == row["label"] for item, _, _ in selected)
    images_dir, videos_dir = output_dir / "images", output_dir / "videos"
    images_dir.mkdir(exist_ok=True); videos_dir.mkdir(exist_ok=True)
    (output_dir / "category_selection.json").write_text(json.dumps(category_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {item.record.product_id: {"title": item.record.title, "label": item.record.label, "tier": tiers[item.record.label], "url": item.record.image_url, "video": item.record.video_url, "pv": item.record.pv, "score": score} for item, score, _ in selected}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} products across {len(tiers)} categories. Pass 3/3: downloading...", flush=True)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_product, item, score, div, tiers[item.record.label], images_dir, videos_dir, args) for item, score, div in selected]
        for done, future in enumerate(as_completed(futures), 1):
            manifest.write(json.dumps(future.result(), ensure_ascii=False) + "\n"); manifest.flush()
            if done % 50 == 0 or done == len(selected):
                print(f"Processed {done}/{len(selected)} products", flush=True)
    print(f"Wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
