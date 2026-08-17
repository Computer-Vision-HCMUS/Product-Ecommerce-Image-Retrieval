"""SCALE-inspired five-modality fusion encoder for retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from encoding.siglip_encoder import SiglipEncoder, l2_normalize
from scale_runtime.modality import MODALITY_NAMES, ProductModalities


def sample_video_frames(path: Path, count: int = 4) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return []
    indices = {round(i * (total - 1) / max(count - 1, 1)) for i in range(count)}
    frames: list[Image.Image] = []
    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        if index in indices:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def mfcc_embedding(path: Path, dimension: int, frames: int = 64) -> np.ndarray | None:
    try:
        import librosa
    except ImportError:
        return None
    try:
        waveform, sample_rate = librosa.load(str(path), sr=16000, mono=True)
    except (OSError, ValueError):
        return None
    if waveform.size == 0:
        return None
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)
    if mfcc.shape[1] < frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :frames]
    pooled = mfcc.mean(axis=1)
    out = np.zeros(dimension, dtype=np.float32)
    out[: min(len(pooled), dimension)] = pooled[:dimension]
    return l2_normalize(out[None, :])[0]


class ScaleFusionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        video_frames: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = SiglipEncoder(model_name, device)
        self.device = self.backbone.device
        self.dimension = 768
        self.video_frames = video_frames
        self.modality_weights = nn.Parameter(torch.ones(len(MODALITY_NAMES), dtype=torch.float32))
        self.audio_projection = nn.Linear(40, self.dimension, bias=False)
        self.to(self.device)
        self.eval()

    def _encode_text_batch(self, texts: list[str]) -> np.ndarray:
        valid = [text if text.strip() else " " for text in texts]
        return self.backbone.encode_texts(valid)

    def encode_modality_matrix(self, product: ProductModalities) -> tuple[np.ndarray, np.ndarray]:
        presence = product.presence()
        vectors = np.zeros((len(MODALITY_NAMES), self.dimension), dtype=np.float32)
        if presence["image"]:
            with Image.open(product.image_path) as image:
                vectors[0] = self.backbone.encode_images([image.convert("RGB")])[0]
        if presence["text"]:
            vectors[1] = self._encode_text_batch([product.text_blob()])[0]
        if presence["table"]:
            vectors[2] = self._encode_text_batch([product.table_blob()])[0]
        if presence["video"]:
            frames = sample_video_frames(Path(product.video_path), self.video_frames)
            if frames:
                frame_vectors = self.backbone.encode_images(frames)
                vectors[3] = l2_normalize(frame_vectors.mean(axis=0, keepdims=True))[0]
                for frame in frames:
                    frame.close()
        if presence["audio"]:
            audio_vec = mfcc_embedding(Path(product.audio_path), self.dimension)
            if audio_vec is not None:
                with torch.inference_mode():
                    projected = self.audio_projection(
                        torch.from_numpy(audio_vec[:40]).float().to(self.device)
                    )
                vectors[4] = l2_normalize(projected.detach().cpu().numpy()[None, :])[0]
        mask = np.array([presence[name] for name in MODALITY_NAMES], dtype=np.float32)
        return vectors, mask

    def encode_product(self, product: ProductModalities) -> tuple[np.ndarray, dict[str, bool]]:
        vectors, mask = self.encode_modality_matrix(product)
        fused = self._fuse_numpy(vectors, mask)
        presence = {name: bool(mask[index]) for index, name in enumerate(MODALITY_NAMES)}
        return fused, presence

    def fuse_modality_matrix_torch(self, vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = F.softmax(self.modality_weights, dim=0)
        weighted = weight.unsqueeze(0) * mask
        denominator = weighted.sum(dim=1, keepdim=True).clamp(min=1e-12)
        fused = (vectors * weighted.unsqueeze(-1)).sum(dim=1) / denominator
        return F.normalize(fused, dim=1)

    def encode_batch(self, products: list[ProductModalities]) -> tuple[np.ndarray, np.ndarray]:
        all_vectors = []
        all_masks = []
        for product in products:
            vector, presence = self.encode_product(product)
            all_vectors.append(vector)
            all_masks.append([presence[name] for name in MODALITY_NAMES])
        return np.vstack(all_vectors), np.asarray(all_masks, dtype=np.float32)

    def _fuse_numpy(self, vectors: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            weight = F.softmax(self.modality_weights, dim=0).cpu().numpy()
        weighted = weight * mask
        denominator = max(float(weighted.sum()), 1e-12)
        fused = (vectors * weighted[:, None]).sum(axis=0) / denominator
        return l2_normalize(fused[None, :])[0]

    def save_weights(self, path: Path) -> None:
        payload = {
            "modality_weights": self.modality_weights.detach().cpu().tolist(),
            "audio_projection": self.audio_projection.weight.detach().cpu().tolist(),
            "model_name": self.backbone.model_name,
            "dimension": self.dimension,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_weights(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.modality_weights.data = torch.tensor(payload["modality_weights"], dtype=torch.float32)
        self.audio_projection.weight.data = torch.tensor(payload["audio_projection"], dtype=torch.float32)
