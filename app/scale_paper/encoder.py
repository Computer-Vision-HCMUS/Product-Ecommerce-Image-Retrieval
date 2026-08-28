"""SCALE paper embedding encoder for retrieval API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCALE_ROOT = Path(__file__).resolve().parents[1] / "SCALE"
if str(SCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALE_ROOT))

from scale_runtime.modality import ProductModalities  # noqa: E402


class ScalePaperEncoder:
    """Serve tpiva embeddings using precomputed sidecars + SCALE checkpoint."""

    def __init__(
        self,
        work_dir: Path,
        checkpoint_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.work_dir / "checkpoints"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.id_label = json.loads((self.work_dir / "id_label.json").read_text(encoding="utf-8"))
        self.path_manifest = json.loads((self.work_dir / "path_manifest.json").read_text(encoding="utf-8"))
        self._feature_cache: dict[str, np.ndarray] = {}
        self._online_encoder = None
        features_root = self.work_dir / "features"
        for split in ("gallery", "test"):
            tpiva = features_root / split / "tpiva_feature_np.npy"
            ids_path = features_root / split / "id.npy"
            if tpiva.is_file() and ids_path.is_file():
                feats = np.load(tpiva)
                ids = np.load(ids_path, allow_pickle=True).tolist()
                for pid, vec in zip(ids, feats):
                    pid = str(pid)
                    vector = vec.astype(np.float32)
                    self._feature_cache[pid] = vector

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return (vector / (np.linalg.norm(vector) + 1e-9)).astype(np.float32)

    def encode_product(self, product: ProductModalities) -> tuple[np.ndarray, dict[str, bool]]:
        if self._online_encoder is None:
            from scale_paper.online_encoder import OnlineScaleEncoder
            self._online_encoder = OnlineScaleEncoder(self.work_dir, self.device)
        return self._online_encoder.encode_product(product)

    def encode_by_id(self, product_id: str) -> np.ndarray:
        return self._normalize(self._feature_cache[str(product_id)])
