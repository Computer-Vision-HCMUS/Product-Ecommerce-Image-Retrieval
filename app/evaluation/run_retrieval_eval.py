"""Compare Faiss results with FlatIP and report retrieval/system metrics."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from evaluation.metrics import metrics_at_k
except ModuleNotFoundError:
    from metrics import metrics_at_k


def normalized(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def mean(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys() if rows else []
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--gallery-embeddings", type=Path, required=True)
    parser.add_argument("--gallery-ids", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import faiss

    query = normalized(np.load(args.query_embeddings).astype("float32"))
    gallery = normalized(np.load(args.gallery_embeddings).astype("float32"))
    query_ids = np.load(args.query_ids, allow_pickle=True).astype(str)
    gallery_ids = np.load(args.gallery_ids, allow_pickle=True).astype(str)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    labels = {product_id: row["label"] for product_id, row in records.items()}
    label_members: dict[str, set[str]] = defaultdict(set)
    for product_id in gallery_ids:
        label_members[labels[product_id]].add(product_id)

    started = time.perf_counter()
    exact_positions = np.argsort(-(query @ gallery.T), axis=1)[:, : args.k]
    exact_seconds = time.perf_counter() - started
    index = faiss.read_index(str(args.index_dir / "index.faiss"))
    started = time.perf_counter()
    _, ann_positions = index.search(query, args.k)
    ann_seconds = time.perf_counter() - started

    exact_metrics, ann_metrics, recalls = [], [], []
    for row, product_id in enumerate(query_ids):
        positives = label_members[labels[product_id]] - {product_id}
        exact_ids = [gallery_ids[pos] for pos in exact_positions[row] if gallery_ids[pos] != product_id]
        ann_ids = [gallery_ids[pos] for pos in ann_positions[row] if pos >= 0 and gallery_ids[pos] != product_id]
        exact_metrics.append(metrics_at_k(exact_ids, positives, args.k))
        ann_metrics.append(metrics_at_k(ann_ids, positives, args.k))
        recalls.append(len(set(exact_ids) & set(ann_ids)) / max(len(exact_ids), 1))
    result = {
        "positive_definition": "same label; custom downloaded_2k split, not official M5Product ground truth",
        "k": args.k,
        "flat": mean(exact_metrics),
        "ann": mean(ann_metrics),
        "ann_recall_vs_flat": float(np.mean(recalls)),
        "flat_latency_ms_per_query": 1000 * exact_seconds / max(len(query), 1),
        "ann_latency_ms_per_query": 1000 * ann_seconds / max(len(query), 1),
        "ann_qps": len(query) / max(ann_seconds, 1e-12),
        "index_bytes": (args.index_dir / "index.faiss").stat().st_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
