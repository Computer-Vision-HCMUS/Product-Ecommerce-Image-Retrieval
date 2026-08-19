"""Run retrieval evaluation using SCALE tpiva features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compute_ap(rank_list: list[str], pos_set: set[str], topk: int) -> float:
    hits = 0
    score = 0.0
    for i, item_id in enumerate(rank_list[:topk]):
        if item_id in pos_set:
            hits += 1
            score += hits / (i + 1)
    if not pos_set:
        return 0.0
    return score / min(len(pos_set), topk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--gallery-features", type=Path, required=True)
    parser.add_argument("--gallery-ids", type=Path, required=True)
    parser.add_argument("--id-label", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    q_feat = np.load(args.query_features).astype(np.float32)
    q_ids = np.load(args.query_ids, allow_pickle=True).tolist()
    g_feat = np.load(args.gallery_features).astype(np.float32)
    g_ids = np.load(args.gallery_ids, allow_pickle=True).tolist()
    id_label = read_json(args.id_label)

    q_norm = q_feat / (np.linalg.norm(q_feat, axis=1, keepdims=True) + 1e-9)
    g_norm = g_feat / (np.linalg.norm(g_feat, axis=1, keepdims=True) + 1e-9)
    sim = q_norm @ g_norm.T

    label_map = {pid: id_label[str(pid)]["label"] for pid in id_label}
    gallery_by_label: dict[str, list[str]] = {}
    for gid in g_ids:
        label = label_map.get(str(gid), "")
        gallery_by_label.setdefault(label, []).append(str(gid))

    aps, hits = [], []
    for qi, qid in enumerate(q_ids):
        q_label = label_map.get(str(qid), "")
        pos_set = set(gallery_by_label.get(q_label, [])) - {str(qid)}
        order = np.argsort(-sim[qi])
        ranked = [str(g_ids[i]) for i in order]
        aps.append(compute_ap(ranked, pos_set, args.topk))
        hits.append(1.0 if any(r in pos_set for r in ranked[: args.topk]) else 0.0)

    results = {
        "metric": "same_label_proxy",
        "topk": args.topk,
        "mAP": float(np.mean(aps)),
        "hit_rate": float(np.mean(hits)),
        "num_queries": len(q_ids),
        "num_gallery": len(g_ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
