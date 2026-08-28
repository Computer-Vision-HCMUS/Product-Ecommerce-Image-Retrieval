"""Shared, label-free metadata reranker for the serving endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class MetadataReranker:
    """Apply the report's S_total blend to an HNSW candidate set only."""

    def __init__(self, work_dir: Path, lambda_emb: float = 0.7) -> None:
        if not 0.0 <= lambda_emb <= 1.0:
            raise ValueError("lambda_emb must be between 0 and 1")
        self.work_dir = Path(work_dir)
        self.lambda_emb = float(lambda_emb)
        self._ids = np.load(self.work_dir / "index_hnsw" / "ids.npy", allow_pickle=True).astype(str)
        self._id_to_position = {product_id: index for index, product_id in enumerate(self._ids)}
        self._metadata = json.loads((self.work_dir / "id_label.json").read_text(encoding="utf-8"))
        self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=60_000, sublinear_tf=True)
        self._lexical_matrix = self._vectorizer.fit_transform([self._document(product_id) for product_id in self._ids])

    def _document(self, product_id: str) -> str:
        item = self._metadata.get(str(product_id), {})
        return f"{item.get('title', '')} {item.get('pv', '')}".strip()

    def rerank(
        self,
        candidates: list[tuple[str, float]],
        *,
        title: str,
        pv: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        query = f"{title} {pv}".strip()
        embedding_scores = np.asarray([score for _, score in candidates], dtype=np.float32)
        if query:
            positions = [self._id_to_position[product_id] for product_id, _ in candidates]
            query_vector = self._vectorizer.transform([query])
            lexical_scores = (query_vector @ self._lexical_matrix[positions].T).toarray().ravel()
        else:
            lexical_scores = np.zeros(len(candidates), dtype=np.float32)
        scores = self.lambda_emb * embedding_scores + (1.0 - self.lambda_emb) * lexical_scores
        order = np.argsort(-scores, kind="stable")[:min(max(int(top_k), 0), len(candidates))]
        return [(candidates[index][0], float(scores[index])) for index in order]
