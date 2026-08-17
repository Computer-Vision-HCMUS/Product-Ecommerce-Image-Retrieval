"""Extract video (1024-d) and audio sidecars — only zero-fill when video truly missing."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

VIDEO_LEN = 12
VIDEO_DIM = 1024


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    repo_path = Path(__file__).resolve().parents[3] / raw_path
    if repo_path.is_file():
        return repo_path
    return None


def is_zero_feature(path: Path) -> bool:
    if not path.is_file():
        return True
    return bool(np.allclose(np.load(path), 0))


def zero_video(out_path: Path) -> None:
    np.save(out_path, np.zeros((VIDEO_LEN, VIDEO_DIM), dtype=np.float32))


def extract_video_frames(video_path: Path, max_frames: int = VIDEO_LEN) -> list:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        # Downscale early to reduce RAM (Unicode paths OK via cv2 on Windows when str path works)
        h, w = frame.shape[:2]
        if max(h, w) > 480:
            scale = 480 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frames.append(frame)
    cap.release()
    return frames


def build_resnet(device):
    import torch
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)
    return model


def extract_video_resnet(
    video_path: Path, out_path: Path, model, device,
) -> bool:
    import torch
    from torchvision.transforms.functional import to_tensor, resize

    frames = extract_video_frames(video_path)
    if not frames:
        return False
    feats = []
    with torch.no_grad():
        for frame in frames[:VIDEO_LEN]:
            rgb = frame[:, :, ::-1].copy()
            tensor = resize(to_tensor(rgb), [224, 224]).to(device)
            feats.append(model(tensor.unsqueeze(0)))
    stacked = torch.cat(feats, dim=0)
    if stacked.shape[0] < VIDEO_LEN:
        pad = torch.zeros(VIDEO_LEN - stacked.shape[0], stacked.shape[1], device=device)
        stacked = torch.cat([stacked, pad], dim=0)
    arr = stacked[:VIDEO_LEN].cpu().numpy().astype(np.float32)
    if arr.shape[1] != VIDEO_DIM:
        proj = np.zeros((VIDEO_LEN, VIDEO_DIM), dtype=np.float32)
        dim = min(arr.shape[1], VIDEO_DIM)
        proj[:, :dim] = arr[:, :dim]
        arr = proj
    np.save(out_path, arr)
    return True


def extract_mp3(video_path: Path, mp3_path: Path) -> bool:
    if mp3_path.is_file():
        return True
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame", "-q:a", "4", str(mp3_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return mp3_path.is_file()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-manifest", type=Path, required=True)
    parser.add_argument("--video-output-dir", type=Path, required=True)
    parser.add_argument("--audio-output-dir", type=Path, default=None)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--zero-fill-missing", action="store_true",
                        help="Zero-fill only when video_path is absent")
    parser.add_argument("--reextract-zero", action="store_true",
                        help="Re-extract when .npy exists but is all zeros and video_path is present")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_manifest = read_json(args.path_manifest)
    if args.ids_file and args.ids_file.is_file():
        ids = read_json(args.ids_file)
        if isinstance(ids, dict):
            ids = ids.get("ids", list(ids.keys()))
    else:
        ids = list(path_manifest.keys())
    if args.limit > 0:
        ids = ids[: args.limit]

    args.video_output_dir.mkdir(parents=True, exist_ok=True)
    if args.audio_output_dir:
        args.audio_output_dir.mkdir(parents=True, exist_ok=True)

    model = build_resnet(device)
    extracted = skipped = zeroed = 0

    for product_id in tqdm(ids, desc="video/audio"):
        paths = path_manifest.get(product_id, {})
        video_path = resolve_path(paths.get("video_path"))
        out_video = args.video_output_dir / f"{product_id}.npy"

        if out_video.is_file() and not (args.reextract_zero and is_zero_feature(out_video)):
            skipped += 1
            continue
        if out_video.is_file() and args.reextract_zero and is_zero_feature(out_video):
            out_video.unlink(missing_ok=True)

        ok = False
        if video_path is not None:
            try:
                ok = extract_video_resnet(video_path, out_video, model, device)
                if ok and args.audio_output_dir:
                    extract_mp3(video_path, args.audio_output_dir / f"{product_id}.mp3")
            except Exception as exc:
                print(f"video error {product_id}: {exc}", file=sys.stderr)
                gc.collect()

        if ok:
            extracted += 1
        elif args.zero_fill_missing and video_path is None:
            zero_video(out_video)
            zeroed += 1

    del model
    gc.collect()
    print(f"extracted={extracted} skipped={skipped} zero-filled(missing video)={zeroed}")


if __name__ == "__main__":
    main()
