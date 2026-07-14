"""Build FlatIP, HNSW, or IVF-PQ indexes from normalized product embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32", copy=False)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-type", choices=("flat", "hnsw", "ivfpq"), default="hnsw")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--pq-bytes", type=int, default=16)
    args = parser.parse_args()

    try:
        import faiss
    except ImportError as exc:
        raise SystemExit("Install faiss-cpu from app/requirements-windows.txt.") from exc

    vectors = normalize(np.load(args.embeddings))
    ids = np.load(args.ids, allow_pickle=True)
    if len(vectors) != len(ids):
        raise SystemExit("Embedding and ID counts differ.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dimension = vectors.shape[1]
    started = time.perf_counter()
    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction
    else:
        if len(vectors) < args.nlist * 40:
            raise SystemExit("IVF-PQ needs more vectors; lower --nlist for this small subset.")
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(
            quantizer, dimension, args.nlist, args.pq_bytes, 8, faiss.METRIC_INNER_PRODUCT
        )
        index.train(vectors)
    index.add(vectors)
    faiss.write_index(index, str(args.output_dir / "index.faiss"))
    np.save(args.output_dir / "ids.npy", ids)
    metadata = {
        "index_type": args.index_type,
        "metric": "inner_product_on_l2_normalized_vectors",
        "dimension": dimension,
        "count": len(vectors),
        "build_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "index.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
