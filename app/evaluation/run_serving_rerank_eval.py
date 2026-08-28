"""Evaluate the deployed HNSW + label-free metadata reranking protocol."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from indexing.catalog_hybrid import MetadataReranker
from indexing.improved_search import FaissSearchBackend


def precision_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    return sum(product_id in positives for product_id in ranked[:k]) / k


def average_precision_at_k(ranked: list[str], positives: set[str], k: int) -> float:
    hits = 0
    score = 0.0
    for position, product_id in enumerate(ranked[:k], start=1):
        if product_id in positives:
            hits += 1
            score += hits / position
    return score / min(len(positives), k) if positives else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--candidate-n", type=int, default=100)
    parser.add_argument("--lambda-emb", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_n < 10:
        raise SystemExit("--candidate-n must be at least 10")

    work = args.work_dir
    metadata = json.loads((work / "id_label.json").read_text(encoding="utf-8"))
    query_vectors = np.load(work / "features" / "test" / "tpiva_feature_np.npy").astype(np.float32)
    query_ids = np.load(work / "features" / "test" / "id.npy", allow_pickle=True).astype(str)
    gallery_ids = np.load(work / "features" / "gallery" / "id.npy", allow_pickle=True).astype(str)
    backend = FaissSearchBackend(work / "index_hnsw")
    reranker = MetadataReranker(work, lambda_emb=args.lambda_emb)

    by_label: dict[str, set[str]] = defaultdict(set)
    for product_id in gallery_ids:
        label = str(metadata.get(product_id, {}).get("label", ""))
        if label:
            by_label[label].add(product_id)

    embedding_ranked_lists: list[list[str]] = []
    reranked_lists: list[list[str]] = []
    for product_id, vector in zip(query_ids, query_vectors):
        item = metadata.get(product_id, {})
        candidates = backend.search(vector, args.candidate_n)
        embedding_ranked_lists.append([candidate_id for candidate_id, _ in candidates])
        reranked_lists.append([
            candidate_id for candidate_id, _ in reranker.rerank(
                candidates,
                title=str(item.get("title", "")),
                pv=str(item.get("pv", "")),
                top_k=args.candidate_n,
            )
        ])

    def score(ranked_lists: list[list[str]]) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for k in (1, 5, 10):
            ap_rows, precision_rows, hit_rows = [], [], []
            for product_id, ranked in zip(query_ids, ranked_lists):
                positives = by_label.get(str(metadata.get(product_id, {}).get("label", "")), set())
                ap_rows.append(average_precision_at_k(ranked, positives, k))
                precision_rows.append(precision_at_k(ranked, positives, k))
                hit_rows.append(float(any(candidate in positives for candidate in ranked[:k])))
            results[f"top{k}"] = {
                "mAP": 100 * float(np.mean(ap_rows)),
                "Precision": 100 * float(np.mean(precision_rows)),
                "HitRate": 100 * float(np.mean(hit_rows)),
            }
        results["average_over_top1_top5_top10"] = {
            "mAP": float(np.mean([results[f"top{k}"]["mAP"] for k in (1, 5, 10)])),
            "Precision": float(np.mean([results[f"top{k}"]["Precision"] for k in (1, 5, 10)])),
        }
        return results

    payload = {
        "protocol": "offline TPIVA test -> HNSW candidate set -> title/PV TF-IDF rerank; labels are used only to score same-label positives",
        "candidate_n": args.candidate_n,
        "lambda_emb": args.lambda_emb,
        "num_queries": int(len(query_ids)),
        "num_gallery": int(len(gallery_ids)),
        "embedding_only_metrics_percent": score(embedding_ranked_lists),
        "serving_reranked_metrics_percent": score(reranked_lists),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
