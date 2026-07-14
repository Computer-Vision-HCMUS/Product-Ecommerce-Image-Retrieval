"""Load a persisted Faiss index and map neighbors to product IDs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype="float32")
    if vector.ndim == 1:
        vector = vector[None, :]
    return vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)


class ProductIndex:
    def __init__(self, index_dir: str | Path) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("Faiss is required to search the product index.") from exc
        index_dir = Path(index_dir)
        self._index = faiss.read_index(str(index_dir / "index.faiss"))
        self._ids = np.load(index_dir / "ids.npy", allow_pickle=True)
        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = max(64, self._index.hnsw.efSearch)

    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        scores, positions = self._index.search(normalize(embedding), top_k)
        return [
            (str(self._ids[position]), float(score))
            for position, score in zip(positions[0], scores[0])
            if position >= 0
        ]
