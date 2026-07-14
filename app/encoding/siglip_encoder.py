"""Shared SigLIP image/text encoder for offline export and online search."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


class SiglipEncoder:
    """Encode catalog images and metadata with a pretrained SigLIP dual tower."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        image_weight: float = 0.7,
    ) -> None:
        if not 0 <= image_weight <= 1:
            raise ValueError("image_weight must be in [0, 1].")
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_weight = image_weight
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _to_numpy(self, tensor) -> np.ndarray:
        return tensor.detach().float().cpu().numpy()

    def encode_images(self, images: Sequence[object]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return l2_normalize(self._to_numpy(features))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return l2_normalize(self._to_numpy(features))

    def fuse(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        if image_embeddings.shape != text_embeddings.shape:
            raise ValueError("SigLIP image/text embeddings have incompatible dimensions.")
        fused = self.image_weight * image_embeddings + (1 - self.image_weight) * text_embeddings
        return l2_normalize(fused)
