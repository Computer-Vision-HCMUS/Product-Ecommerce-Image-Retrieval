"""Extract mel-spectrogram audio features (80 x 751) for SCALE pretraining."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

AUDIO_LEN = 12
MEL_BINS = 80
MEL_FRAMES = 751


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_audio_feature(item_id: str, audio_dir: Path, audio_feature_dir: Path) -> bool:
    mp3_path = audio_dir / f"{item_id}.mp3"
    out_path = audio_feature_dir / f"{item_id}.npy"
    if not mp3_path.is_file():
        return False
    if out_path.is_file():
        return True
    try:
        waveform, sample_rate = librosa.load(str(mp3_path), sr=16000, mono=True)
        target_len = AUDIO_LEN * 16000
        if waveform.shape[0] < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[: waveform.shape[0]] = waveform
            waveform = padded
        else:
            waveform = waveform[:target_len]
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=16000, n_fft=1024, hop_length=256, n_mels=MEL_BINS
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = mel.astype(np.float32)
        cur_mean, cur_std = mel.mean(axis=1, keepdims=True), mel.std(axis=1, keepdims=True)
        mel = (mel - cur_mean) / (cur_std + 1e-9)
        if mel.shape[1] < MEL_FRAMES:
            padded = np.zeros((MEL_BINS, MEL_FRAMES), dtype=np.float32)
            padded[:, : mel.shape[1]] = mel
            mel = padded
        else:
            mel = mel[:, :MEL_FRAMES]
        np.save(out_path, mel)
        return True
    except Exception as exc:
        print(f"audio error {item_id}: {exc}")
        return False


def write_zero_audio(item_id: str, audio_feature_dir: Path) -> None:
    out_path = audio_feature_dir / f"{item_id}.npy"
    if not out_path.is_file():
        np.save(out_path, np.zeros((MEL_BINS, MEL_FRAMES), dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-label", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zero-fill-missing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = list(read_json(args.id_label).keys())

    for item_id in tqdm(ids, desc="audio features"):
        ok = extract_audio_feature(item_id, args.audio_dir, args.output_dir)
        if not ok and args.zero_fill_missing:
            write_zero_audio(item_id, args.output_dir)


if __name__ == "__main__":
    main()
