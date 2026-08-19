"""Build and download a balanced, high-quality M5Product subset.

The metadata file is streamed three times, so it is safe to use with the full
M5Product JSON object. The default selection contains the 50 largest curated
super-categories, with up to 200 products from each super-category.

Within each super-category, products are split by modality completeness:
    80% full modality (title, label, image, video, pv all present)
    20% incomplete modality (naturally missing fields, or full samples with
    1-2 modalities artificially masked when natural incomplete samples run out)

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
import hashlib
import json
import mimetypes
import os
import posixpath
import random
import re
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
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
MODALITY_FIELDS = ("title", "label", "image_url", "video_url", "pv")
MASKABLE_MODALITIES = ("title", "pv", "video_url")


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


@dataclass(frozen=True)
class SelectionMeta:
    modality_complete: bool
    modality_source: str
    masked_modalities: tuple[str, ...]
    modality_present: dict[str, bool]


class NetworkInterrupted(RuntimeError):
    """Raised when connectivity fails and the download must stop immediately."""


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


def modality_presence(record: ProductRecord) -> dict[str, bool]:
    return {
        "title": nonempty(record.title),
        "label": nonempty(record.label),
        "image_url": nonempty(record.image_url),
        "video_url": nonempty(record.video_url),
        "pv": nonempty(record.pv),
    }


def is_selectable(record: ProductRecord) -> bool:
    """Require image and label so a product can be downloaded and evaluated."""
    flags = modality_presence(record)
    return flags["label"] and flags["image_url"]


def is_full_modality(record: ProductRecord) -> bool:
    return all(modality_presence(record).values())


def missing_modalities(record: ProductRecord) -> tuple[str, ...]:
    return tuple(name for name, present in modality_presence(record).items() if not present)


def mask_rng(product_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{product_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def choose_masked_modalities(record: ProductRecord, seed: int) -> tuple[str, ...]:
    """Pick 1-2 maskable modalities to hide on an otherwise complete sample."""
    present = [name for name in MASKABLE_MODALITIES if modality_presence(record)[name]]
    if not present:
        return ()
    rng = mask_rng(record.product_id, seed)
    mask_count = 1 if len(present) == 1 or rng.random() < 0.7 else 2
    shuffled = present[:]
    rng.shuffle(shuffled)
    return tuple(sorted(shuffled[:mask_count]))


def apply_modality_masks(record: ProductRecord, masked_modalities: tuple[str, ...]) -> ProductRecord:
    if not masked_modalities:
        return record
    updates = {}
    if "title" in masked_modalities:
        updates["title"] = ""
    if "pv" in masked_modalities:
        updates["pv"] = ""
    if "video_url" in masked_modalities:
        updates["video_url"] = ""
    return replace(record, **updates)


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


def _greedy_select(
    remaining: list[ScoredRecord], count: int,
) -> tuple[list[tuple[ScoredRecord, float, float]], list[ScoredRecord]]:
    """Pick `count` diverse, high-scoring products from `remaining`."""
    chosen: list[ScoredRecord] = []
    selected: list[tuple[ScoredRecord, float, float]] = []
    token_counts: Counter[str] = Counter()
    merchants: set[str] = set()
    hosts: set[str] = set()
    pool = remaining[:]
    while pool and len(chosen) < count:
        best = max(
            pool,
            key=lambda item: (
                item.base_score + 0.1 * diversity(item, chosen, token_counts, merchants, hosts),
                item.record.product_id,
            ),
        )
        div = diversity(best, chosen, token_counts, merchants, hosts)
        score = best.base_score + 0.1 * div
        selected.append((best, score, div))
        chosen.append(best)
        token_counts.update(best.title_tokens)
        if best.merchant:
            merchants.add(best.merchant)
        if best.image_host:
            hosts.add(best.image_host)
        pool.remove(best)
    return selected, pool


def select_products(
    candidates: Dict[str, list[ScoredRecord]],
    per_category: int,
    full_modality_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[tuple[ScoredRecord, float, float, SelectionMeta]], list[dict[str, object]]]:
    """Select products per super-category with an 80/20 full/incomplete modality mix."""
    if not 0 < full_modality_ratio < 1:
        raise ValueError("full_modality_ratio must be in (0, 1).")
    full_quota = round(per_category * full_modality_ratio)
    incomplete_quota = per_category - full_quota
    selected: list[tuple[ScoredRecord, float, float, SelectionMeta]] = []
    summary_rows: list[dict[str, object]] = []

    for super_category in sorted(candidates):
        pool = [item for item in candidates[super_category] if is_selectable(item.record)]
        full_pool = [item for item in pool if is_full_modality(item.record)]
        incomplete_pool = [item for item in pool if not is_full_modality(item.record)]

        full_selected, full_remaining = _greedy_select(full_pool, full_quota)
        selected_ids = {item.record.product_id for item, _, _ in full_selected}

        incomplete_candidates = [
            item for item in incomplete_pool if item.record.product_id not in selected_ids
        ]
        incomplete_selected, _ = _greedy_select(incomplete_candidates, incomplete_quota)
        selected_ids.update(item.record.product_id for item, _, _ in incomplete_selected)

        masked_selected: list[tuple[ScoredRecord, float, float, SelectionMeta]] = []
        gap = incomplete_quota - len(incomplete_selected)
        if gap > 0:
            fill_pool = [
                item for item in full_remaining if item.record.product_id not in selected_ids
            ]
            fill_selected, _ = _greedy_select(fill_pool, gap)
            for item, score, div in fill_selected:
                masked = choose_masked_modalities(item.record, seed)
                masked_selected.append(
                    (
                        item,
                        score,
                        div,
                        SelectionMeta(
                            modality_complete=False,
                            modality_source="masked",
                            masked_modalities=masked,
                            modality_present=modality_presence(item.record),
                        ),
                    )
                )
                selected_ids.add(item.record.product_id)

        for item, score, div in full_selected:
            selected.append(
                (
                    item,
                    score,
                    div,
                    SelectionMeta(
                        modality_complete=True,
                        modality_source="natural_full",
                        masked_modalities=(),
                        modality_present=modality_presence(item.record),
                    ),
                )
            )
        for item, score, div in incomplete_selected:
            selected.append(
                (
                    item,
                    score,
                    div,
                    SelectionMeta(
                        modality_complete=False,
                        modality_source="natural_incomplete",
                        masked_modalities=missing_modalities(item.record),
                        modality_present=modality_presence(item.record),
                    ),
                )
            )
        selected.extend(masked_selected)

        summary_rows.append(
            {
                "super_category": super_category,
                "requested": per_category,
                "selected": len(full_selected) + len(incomplete_selected) + len(masked_selected),
                "full_modality": len(full_selected),
                "natural_incomplete": len(incomplete_selected),
                "masked_incomplete": len(masked_selected),
                "full_quota": full_quota,
                "incomplete_quota": incomplete_quota,
            }
        )
    return selected, summary_rows


def folder_name(super_category: str) -> str:
    """Create a stable Windows-safe folder name for a super-category."""
    readable = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", super_category).strip(" ._")[:80]
    suffix = hashlib.sha1(super_category.encode("utf-8")).hexdigest()[:8]
    return f"{readable or 'uncategorized'}__{suffix}"


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


def download_url(
    url: str, output_stem: Path, fallback_ext: str, timeout: int, retries: int,
    overwrite: bool, stop_event: threading.Event, stop_on_network_error: bool,
) -> Tuple[Optional[str], str]:
    if stop_event.is_set():
        raise NetworkInterrupted("Download stopped after a connectivity failure.")
    url = normalize_url(url)
    if not url:
        return None, "missing"
    candidate_path = output_stem.with_suffix(guess_extension(url, None, fallback_ext))
    if candidate_path.exists() and not overwrite:
        return str(candidate_path), "exists"
    last_error = ""
    for attempt in range(1, retries + 2):
        if stop_event.is_set():
            raise NetworkInterrupted("Download stopped after a connectivity failure.")
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
        except HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            break
        except (URLError, TimeoutError, socket.timeout) as exc:
            if stop_on_network_error:
                stop_event.set()
                raise NetworkInterrupted(f"Connectivity failure: {type(exc).__name__}: {exc}") from exc
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                time.sleep(min(2 * attempt, 8))
        except (OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            break
    return None, f"error: {last_error}"


def download_product(
    scored: ScoredRecord,
    score: float,
    diversity_score: float,
    selection: SelectionMeta,
    super_category: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    stop_event: threading.Event,
) -> Dict[str, object]:
    record = apply_modality_masks(scored.record, selection.masked_modalities)
    directory = folder_name(super_category)
    category_dir = dataset_dir / directory
    category_images, category_videos = category_dir / "images", category_dir / "videos"
    category_images.mkdir(parents=True, exist_ok=True)
    category_videos.mkdir(parents=True, exist_ok=True)
    image_path, image_status = download_url(
        record.image_url, category_images / record.product_id, ".jpg",
        args.timeout, args.retries, args.overwrite, stop_event, args.stop_on_network_error,
    )
    video_path, video_status = download_url(
        record.video_url, category_videos / record.product_id, ".mp4",
        args.timeout, args.retries, args.overwrite, stop_event, args.stop_on_network_error,
    )
    return {
        "id": record.product_id,
        "title": record.title,
        "label": record.label,
        "super_category": super_category,
        "pv": record.pv,
        "image_url": record.image_url,
        "video_url": record.video_url,
        "score": round(score, 6),
        "score_components": {
            "completeness": round(scored.completeness, 6),
            "text_quality": round(scored.text_quality, 6),
            "merchant_score": round(scored.merchant_score, 6),
            "diversity": round(diversity_score, 6),
        },
        "modality_complete": selection.modality_complete,
        "modality_source": selection.modality_source,
        "masked_modalities": list(selection.masked_modalities),
        "modality_present": selection.modality_present,
        "image_path": image_path,
        "video_path": video_path,
        "image_status": image_status,
        "video_status": video_status,
    }


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
    parser.add_argument("--super-categories", type=int, default=50, help="Number of largest super-categories to include.")
    parser.add_argument("--per-super-category", "--per-category", dest="per_super_category", type=int, default=200,
                        help="Maximum selected products per super-category.")
    parser.add_argument("--full-modality-ratio", type=float, default=0.8,
                        help="Share of each super-category that must have all modalities present.")
    parser.add_argument("--selection-seed", type=int, default=42,
                        help="Seed for deterministic artificial modality masking.")
    parser.add_argument("--candidate-pool", type=int, default=1000, help="Best pre-diversification candidates retained per super-category.")
    parser.add_argument("--min-category-samples", type=int, default=None,
                        help="Minimum metadata rows per label; defaults to --per-category.")
    parser.add_argument("--max-labels-per-super-category", type=int, default=3,
                        help="Maximum selected labels from a super-category across all frequency bands.")
    parser.add_argument("--max-products", type=int, default=None,
                        help="Stop after this many manifest records; defaults to all selected products.")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing manifest and skip product IDs already recorded.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed URL after the first request.")
    parser.add_argument("--stop-on-network-error", action="store_true",
                        help="Stop the entire download on DNS, connection, or timeout errors (disabled by default).")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.super_categories <= 0 or args.per_super_category <= 0 or args.candidate_pool < args.per_super_category:
        raise SystemExit("--super-categories and --per-super-category must be positive; --candidate-pool must be at least --per-super-category.")
    if not 0 < args.full_modality_ratio < 1:
        raise SystemExit("--full-modality-ratio must be between 0 and 1.")
    if args.max_products is not None and args.max_products <= 0:
        raise SystemExit("--max-products must be positive.")
    min_samples = args.min_category_samples or args.per_super_category
    if min_samples < args.per_super_category:
        raise SystemExit("--min-category-samples must be at least --per-super-category.")
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
    super_counts: Counter[str] = Counter()
    for label, count in counts.items():
        if taxonomy.get(label):
            super_counts[taxonomy[label]] += count
    selected_super_categories = [name for name, _ in sorted(super_counts.items(), key=lambda item: (-item[1], item[0]))[:args.super_categories]]
    if len(selected_super_categories) != args.super_categories:
        raise SystemExit(f"Only found {len(selected_super_categories)} mapped super-categories; need {args.super_categories}.")
    selected_super_set = set(selected_super_categories)
    category_rows = [{"super_category": name, "sample_count": super_counts[name], "selected": 0, "folder": folder_name(name)} for name in selected_super_categories]
    output_dir.mkdir(parents=True, exist_ok=True)
    with category_counts_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=("rank", "label", "sample_count", "selected", "tier", "super_category"))
        writer.writeheader()
        for rank, (label, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0])), 1):
            writer.writerow({"rank": rank, "label": label, "sample_count": count,
                             "selected": taxonomy.get(label, "") in selected_super_set, "tier": "", "super_category": taxonomy.get(label, "")})
    print(f"Wrote category statistics: {category_counts_path} ({len(counts)} categories)", flush=True)
    print(f"Pass 2/3: scoring products in {len(selected_super_categories)} selected super-categories...", flush=True)
    candidate_heaps: Dict[str, list[Tuple[float, str, ScoredRecord]]] = defaultdict(list)
    for record in iter_top_level_object(input_path):
        super_category = taxonomy.get(record.label)
        if super_category in selected_super_set:
            scored = score_record(record)
            heap = candidate_heaps[super_category]
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
    selected, modality_summary = select_products(
        candidates,
        args.per_super_category,
        full_modality_ratio=args.full_modality_ratio,
        seed=args.selection_seed,
    )
    for row in category_rows:
        row["selected"] = sum(
            taxonomy.get(item.record.label) == row["super_category"]
            for item, _, _, _ in selected
        )
    (output_dir / "category_selection.json").write_text(
        json.dumps(category_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "modality_selection.json").write_text(
        json.dumps(modality_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = output_dir / "manifest.jsonl"
    existing_ids: set[str] = set()
    if args.resume and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as manifest:
            for line in manifest:
                try:
                    product_id = json.loads(line).get("id")
                except json.JSONDecodeError:
                    continue
                if product_id:
                    existing_ids.add(str(product_id))
    target_count = min(args.max_products or len(selected), len(selected))
    target_selected = [entry for entry in selected if entry[0].record.product_id in existing_ids]
    for entry in selected:
        if len(target_selected) >= target_count:
            break
        if entry[0].record.product_id not in existing_ids:
            target_selected.append(entry)
    if len(target_selected) < target_count:
        raise SystemExit(f"Only {len(target_selected)} selected records are available; cannot reach --max-products={target_count}.")
    pending = [entry for entry in target_selected if entry[0].record.product_id not in existing_ids]
    metadata = {}
    for item, score, _, selection in target_selected:
        export_record = apply_modality_masks(item.record, selection.masked_modalities)
        metadata[item.record.product_id] = {
            "title": export_record.title,
            "label": export_record.label,
            "super_category": taxonomy[item.record.label],
            "url": export_record.image_url,
            "video": export_record.video_url,
            "pv": export_record.pv,
            "score": score,
            "modality_complete": selection.modality_complete,
            "modality_source": selection.modality_source,
            "masked_modalities": list(selection.masked_modalities),
            "modality_present": selection.modality_present,
        }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    modality_counts = Counter(selection.modality_source for _, _, _, selection in target_selected)
    print(
        f"Selected {len(selected)} products across {len(selected_super_categories)} super-categories "
        f"(full={modality_counts.get('natural_full', 0)}, "
        f"natural_incomplete={modality_counts.get('natural_incomplete', 0)}, "
        f"masked={modality_counts.get('masked', 0)}). "
        f"Pass 3/3: {len(existing_ids)} existing, {len(pending)} pending, target {target_count}...",
        flush=True,
    )
    manifest_mode = "a" if args.resume else "w"
    stop_event = threading.Event()
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                download_product, item, score, div, selection,
                taxonomy[item.record.label], output_dir, args, stop_event,
            )
            for item, score, div, selection in pending
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except NetworkInterrupted as exc:
                stop_event.set()
                for pending_future in futures:
                    pending_future.cancel()
                raise SystemExit(
                    "Network connection was interrupted. Download stopped without retrying; "
                    "run again with --resume after the connection is stable. "
                    f"Details: {exc}"
                ) from exc
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n"); manifest.flush()
            done = len(existing_ids) + completed
            if completed % 50 == 0 or completed == len(pending):
                print(f"Processed {done}/{target_count} products", flush=True)
    print(f"Wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
