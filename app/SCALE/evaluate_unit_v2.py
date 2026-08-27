#!/usr/bin/env python
"""Evaluate custom subset retrieval without leaking non-gallery positives.

This differs from the official M5Product protocol: the local subset declares
same-label relevance, and requires an explicit gallery ID list for valid AP.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURE_TYPES = (
    "t", "p", "i", "v", "a", "tp", "ti", "tv", "pi", "pv", "iv", "ta", "pa", "ia", "va",
    "tpi", "tpv", "tiv", "piv", "tpiv", "tpiva", "dense",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_metric_dir", required=True)
    parser.add_argument("--retrieval_result_dir", required=True)
    parser.add_argument("--GT_file", required=True)
    parser.add_argument("--gallery-ids", required=True,
                        help="Complete gallery ID array (.npy); required for valid subset metrics.")
    for feature_type in FEATURE_TYPES:
        parser.add_argument(f"--{feature_type}", action="store_true")
    return parser.parse_args()


def precision_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    return sum(item in positives for item in ranked[:k]) / k


def average_precision_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    hits = 0
    score = 0.0
    for position, item in enumerate(ranked[:k], start=1):
        if item in positives:
            hits += 1
            score += hits / position
    return score / min(len(positives), k) if positives else 0.0


def summarize(rows: list[tuple[float, float, float]]) -> dict[str, float | int]:
    return {
        "mAP": 100 * float(np.mean([row[0] for row in rows])),
        "Prec": 100 * float(np.mean([row[1] for row in rows])),
        "mHitRate": 100 * float(np.mean([row[2] for row in rows])),
        "query_count": len(rows),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_metric_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(Path(args.GT_file).read_text(encoding="utf-8"))
    gallery_ids = {str(item) for item in np.load(args.gallery_ids, allow_pickle=True).tolist()}
    if not gallery_ids:
        raise SystemExit("--gallery-ids is empty.")

    label_to_gallery: dict[str, set[str]] = defaultdict(set)
    for product_id in gallery_ids:
        info = catalog.get(product_id)
        if info and info.get("label"):
            label_to_gallery[str(info["label"])].add(product_id)

    selected_types = [name for name in FEATURE_TYPES if getattr(args, name)]
    if not selected_types:
        raise SystemExit("Select at least one feature type, for example --tpiva.")
    results: dict[str, dict] = {}
    for feature_type in selected_types:
        result_path = Path(args.retrieval_result_dir) / f"{feature_type}_feature_retrieval_id_list.txt"
        records: list[tuple[str, list[str]]] = []
        for line in result_path.read_text(encoding="utf-8").splitlines():
            parts = [part for part in line.split(",") if part]
            if parts:
                records.append((parts[0], [item for item in parts[1:] if item in gallery_ids]))
        per_type: dict[str, dict] = {}
        for k in (1, 5, 10):
            rows: list[tuple[float, float, float]] = []
            cohorts: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
            skipped = 0
            for query_id, ranked in records:
                query = catalog.get(query_id)
                if not query:
                    continue
                positives = label_to_gallery.get(str(query.get("label", "")), set())
                if not positives:
                    skipped += 1
                    continue
                row = (
                    average_precision_at_k(ranked, positives, k),
                    precision_at_k(ranked, positives, k),
                    float(any(item in positives for item in ranked[:k])),
                )
                rows.append(row)
                cohorts[str(query.get("modality_source", "unknown"))].append(row)
            if not rows:
                raise SystemExit(f"No evaluable queries for {feature_type} at K={k}.")
            summary = summarize(rows)
            summary["skipped_no_gallery_positive"] = skipped
            summary["positive_definition"] = "same label, restricted to gallery IDs; custom subset protocol, not official M5Product ground truth"
            summary["cohort_metrics"] = {name: summarize(values) for name, values in cohorts.items()}
            per_type[f"top{k}"] = summary
            print(f"{feature_type} top{k}: mAP={summary['mAP']:.2f} Prec={summary['Prec']:.2f} Hit={summary['mHitRate']:.2f}")
        results[feature_type] = per_type

    (output_dir / "metric_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
