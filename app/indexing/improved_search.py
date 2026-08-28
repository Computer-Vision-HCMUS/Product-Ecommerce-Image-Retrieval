"""Faiss-backed retrieval service used by the API.

The baseline is exact to the gallery embedding index.  ``improved`` currently
uses the same candidate index until attribute reranking is trained/validated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from indexing.pipeline_config import PipelineMode, PipelinePaths
from indexing.rerank import RerankWeights


class FaissSearchBackend:
    def __init__(self, index_dir: Path) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss is required for retrieval.") from exc
        index_path = index_dir / "index.faiss"
        ids_path = index_dir / "ids.npy"
        if not index_path.is_file() or not ids_path.is_file():
            raise FileNotFoundError(
                f"Missing retrieval index in {index_dir}. Build it from gallery embeddings first."
            )
        self._index = faiss.read_index(str(index_path))
        self._ids = np.load(ids_path, allow_pickle=True)

    def search(self, embedding: np.ndarray, top_k: int, **_: object) -> list[tuple[str, float]]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        requested_k = min(int(top_k), len(self._ids))
        # HNSW otherwise exposes only a small candidate set even when the
        # serving layer asks to inspect the complete 1,054-item gallery for
        # catalog-aware label reranking.
        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = max(int(self._index.hnsw.efSearch), requested_k)
        scores, positions = self._index.search(vector, requested_k)
        return [
            (str(self._ids[pos]), float(score))
            for score, pos in zip(scores[0], positions[0])
            if pos >= 0
        ]


def build_search_backend(
    work_dir: Path | str,
    mode: PipelineMode | str,
    *,
    candidate_n: int = 100,
    weights: RerankWeights | None = None,
) -> FaissSearchBackend:
    del candidate_n, weights
    paths = PipelinePaths(work_dir, mode)
    # Improved reranking is not validated for this subset yet.  Reuse the
    # baseline index rather than silently serving an unrelated index.
    index_dir = paths.index_dir
    if not (index_dir / "index.faiss").is_file():
        index_dir = PipelinePaths(work_dir, PipelineMode.BASELINE).index_dir
    return FaissSearchBackend(index_dir)
