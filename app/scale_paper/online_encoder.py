"""Online TPIVA encoder using the same SCALE checkpoint and sidecar rules.

The offline evaluator operates on pre-extracted region, video and mel features.
This module applies those extractors to one uploaded query, then calls the
checkpoint directly.  It deliberately has no catalog-title or label fallback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from scale_runtime.modality import ProductModalities

SCALE_ROOT = Path(__file__).resolve().parents[1] / "SCALE"
if str(SCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALE_ROOT))

from model.SCALE import BertConfig, BertForMultiModalPreTraining  # noqa: E402
from pytorch_pretrained_bert.tokenization import BertTokenizer  # noqa: E402


class OnlineScaleEncoder:
    """Encode a single raw query into the 768-d TPIVA space."""

    text_len = 36
    pv_len = 64
    region_len = 36
    video_len = 12
    audio_bins = 80

    def __init__(self, work_dir: Path, device: torch.device) -> None:
        self.work_dir = Path(work_dir)
        self.device = device
        repo_root = Path(__file__).resolve().parents[2]
        checkpoint = self._latest_checkpoint()
        config = BertConfig.from_json_file(str(SCALE_ROOT / "config" / "bert_base_6layer_6conect_capture_itp3va.json"))
        config.fast_mode = True
        config.v_target_size = 2048
        config.predict_feature = True
        self.model = BertForMultiModalPreTraining.from_pretrained(str(checkpoint), config).eval().to(device)
        self.tokenizer = BertTokenizer.from_pretrained(str(repo_root / "artifacts" / "pretrained" / "bert-base-chinese"), do_lower_case=True)
        self._region_backend: str | None = None
        self._region_model = None
        self._video_model = None

    def _latest_checkpoint(self) -> Path:
        candidates = list((self.work_dir / "checkpoints" / "scale_paper_simcl").glob("pytorch_model_*.bin"))
        if not candidates:
            raise FileNotFoundError("No SCALE checkpoint found under checkpoints/scale_paper_simcl.")
        return max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        return (vector / max(float(np.linalg.norm(vector)), 1e-12)).astype(np.float32)

    def _tokens(self, text: str, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = self.tokenizer.tokenize(text)[: limit - 2]
        ids = self.tokenizer.convert_tokens_to_ids(["[CLS]", *tokens, "[SEP]"])
        mask = [1] * len(ids)
        ids += [0] * (limit - len(ids))
        mask += [0] * (limit - len(mask))
        return np.asarray(ids), np.asarray(mask), np.zeros(limit, dtype=np.int64)

    @staticmethod
    def _pv_text(value: str) -> str:
        words: list[str] = []
        for pair in value.split("#;#"):
            if "#:#" not in pair:
                continue
            key, val = pair.split("#:#", 1)
            if key.strip() and val.strip():
                words.extend((key.strip(), val.strip()))
        return " ".join(words)

    def _regions(self, image_path: str | None) -> tuple[np.ndarray, np.ndarray, int]:
        features = np.zeros((self.region_len, 2048), dtype=np.float32)
        locations = np.zeros((self.region_len, 5), dtype=np.float32)
        if not image_path or not Path(image_path).is_file():
            return features, locations, 0
        from tools.bp_feature.extract.extract_regions_windows import (
            build_detectron2_predictor, build_torchvision_detector, extract_detectron2,
            extract_torchvision, load_image_bgr,
        )
        if self._region_backend is None:
            try:
                self._region_model = build_detectron2_predictor(SCALE_ROOT / "tools" / "bp_feature" / "extract" / "models")
                self._region_backend = "detectron2"
            except Exception:
                self._region_model = build_torchvision_detector(self.device)
                self._region_backend = "torchvision"
        path = Path(image_path)
        if self._region_backend == "detectron2":
            boxes, feats = extract_detectron2(self._region_model, path)
        else:
            boxes, feats = extract_torchvision(self._region_model, path, self.device)
        image = load_image_bgr(path)
        height, width = image.shape[:2]
        count = min(len(boxes), self.region_len)
        features[:count, : min(feats.shape[1], 2048)] = feats[:count, :2048]
        locations[:count, :4] = boxes[:count]
        locations[:count, 4] = ((locations[:count, 3] - locations[:count, 1]) * (locations[:count, 2] - locations[:count, 0])) / max(height * width, 1)
        locations[:count, [0, 2]] /= max(width, 1)
        locations[:count, [1, 3]] /= max(height, 1)
        return features, locations, count

    def _video(self, video_path: str | None) -> tuple[np.ndarray, int]:
        result = np.zeros((self.video_len, 1024), dtype=np.float32)
        if not video_path or not Path(video_path).is_file():
            return result, 0
        from preprocess.extract_video_audio import build_resnet, extract_video_resnet
        import tempfile
        if self._video_model is None:
            self._video_model = build_resnet(self.device)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            if not extract_video_resnet(Path(video_path), temp_path, self._video_model, self.device, self.video_len):
                return result, 0
            result = np.load(temp_path).astype(np.float32)
            return result, int(np.count_nonzero(np.any(result, axis=1)))
        finally:
            temp_path.unlink(missing_ok=True)

    def _audio(self, audio_path: str | None) -> tuple[np.ndarray, int]:
        result = np.zeros((self.audio_bins, 751), dtype=np.float32)
        if not audio_path or not Path(audio_path).is_file():
            return result, 0
        import librosa
        waveform, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        target = self.video_len * 16000
        waveform = np.pad(waveform[:target], (0, max(target - len(waveform), 0)))
        mel = librosa.feature.melspectrogram(y=waveform, sr=16000, n_fft=1024, hop_length=256, n_mels=self.audio_bins)
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-9)
        result[:, : min(mel.shape[1], 751)] = mel[:, :751]
        return result, self.audio_bins

    def encode_product(self, product: ProductModalities) -> tuple[np.ndarray, dict[str, bool]]:
        presence = product.presence()
        text_ids, text_mask, text_segments = self._tokens(product.text_blob(), self.text_len)
        pv_ids, pv_mask, pv_segments = self._tokens(self._pv_text(product.table_blob()), self.pv_len)
        image, image_loc, image_count = self._regions(product.image_path)
        video, video_count = self._video(product.video_path)
        audio, audio_count = self._audio(product.audio_path)
        image_target, video_target, audio_target = image.copy(), video.copy(), audio.copy()

        image_mask = np.r_[1 if image_count else 0, [1] * image_count, [0] * (self.region_len - image_count)]
        video_mask = np.r_[1 if video_count else 0, [1] * video_count, [0] * (self.video_len - video_count)]
        audio_mask = np.r_[1 if audio_count else 0, [1] * audio_count, [0] * (self.audio_bins - audio_count)]
        image = np.vstack((image.sum(axis=0, keepdims=True) / max(image_count, 1), image))
        image_loc = np.vstack((np.array([[0, 0, 1, 1, 1]], dtype=np.float32), image_loc))
        video = np.vstack((video.sum(axis=0, keepdims=True) / max(video_count, 1), video))
        audio = np.vstack((audio.mean(axis=0, keepdims=True) if audio_count else np.zeros((1, 751), dtype=np.float32), audio))

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(
                tensor(text_ids), tensor(pv_ids), tensor(image), tensor(video), tensor(audio), tensor(image_loc),
                tensor(text_segments), tensor(pv_segments), tensor(text_mask), tensor(pv_mask), tensor(image_mask),
                tensor(video_mask), tensor(audio_mask), tensor(np.full(self.text_len, -1)), tensor(np.full(self.pv_len, -1)),
                tensor(np.full(self.region_len, -1)), tensor(image_target), tensor(np.full(self.video_len, -1)), tensor(video_target),
                tensor(np.full(self.audio_bins, -1)), tensor(audio_target), tensor(np.array(0)), return_features=True,
            )
        vector = sum(output[index] for index in range(6, 11)).squeeze(0).cpu().numpy()
        return self._normalise(vector), presence
