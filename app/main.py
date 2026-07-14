"""Query a built SigLIP/Faiss product index.

The offline exporter creates normalized `embedding.npy`/`id.npy` artifacts.
Use `--image` (and optional `--text`) for a live SigLIP query, `--query-id`
for reproducible evaluation, or `--embedding` for a precomputed vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from encoding.siglip_encoder import SiglipEncoder
from indexing.search import ProductIndex


def load_query_embedding(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        with Image.open(args.image) as image:
            image_embedding = args.encoder.encode_images([image.convert("RGB")])
        if not args.text:
            return image_embedding[0]
        text_embedding = args.encoder.encode_texts([args.text])
        return args.encoder.fuse(image_embedding, text_embedding)[0]
    if args.embedding:
        return np.load(args.embedding)
    if args.query_id:
        ids = np.load(args.query_ids, allow_pickle=True)
        matches = np.where(ids.astype(str) == args.query_id)[0]
        if not len(matches):
            raise SystemExit(f"Query ID {args.query_id!r} was not found in {args.query_ids}.")
        return np.load(args.query_embeddings)[matches[0]]
    raise SystemExit("Provide --image, --embedding, or --query-id.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search fused SigLIP product embeddings with Faiss.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True, help="prepared records.json")
    parser.add_argument("--top-k", type=int, default=10)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Live query image.")
    source.add_argument("--embedding", type=Path, help="Path to one query embedding .npy")
    source.add_argument("--query-id", help="ID from an exported query split")
    parser.add_argument("--text", help="Optional title, PV, or caption accompanying --image.")
    parser.add_argument("--query-embeddings", type=Path, default=Path("query/embedding.npy"))
    parser.add_argument("--query-ids", type=Path, default=Path("query/id.npy"))
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--image-weight", type=float, default=0.7)
    parser.add_argument("--device", help="cuda or cpu; auto-selects when omitted")
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    args.encoder = (
        SiglipEncoder(args.model, args.device, args.image_weight) if args.image else None
    )
    embedding = load_query_embedding(args)
    results = []
    for product_id, score in ProductIndex(args.index_dir).search(embedding, args.top_k):
        item = records.get(product_id, {})
        results.append(
            {
                "id": product_id,
                "score": score,
                "title": item.get("title", ""),
                "label": item.get("label", ""),
                "image_path": item.get("image_path", ""),
                "has_video": item.get("has_video", False),
            }
        )
    print(json.dumps({"top_k": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()