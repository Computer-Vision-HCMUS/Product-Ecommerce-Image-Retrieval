"""Retrieval metrics for the custom label-positive downloaded_2k split."""

from __future__ import annotations

import math
from collections.abc import Sequence


def metrics_at_k(retrieved: Sequence[str], positives: set[str], k: int) -> dict[str, float]:
    ranked = list(retrieved[:k])
    hits = [item in positives for item in ranked]
    hit_count = sum(hits)
    precision = hit_count / k if k else 0.0
    recall = hit_count / len(positives) if positives else 0.0
    average_precision = 0.0
    for position, hit in enumerate(hits, start=1):
        if hit:
            average_precision += sum(hits[:position]) / position
    average_precision /= min(len(positives), k) if positives else 1
    dcg = sum(hit / math.log2(position + 1) for position, hit in enumerate(hits, start=1))
    ideal = sum(1 / math.log2(position + 1) for position in range(1, min(k, len(positives)) + 1))
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"map@{k}": average_precision,
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
    }
