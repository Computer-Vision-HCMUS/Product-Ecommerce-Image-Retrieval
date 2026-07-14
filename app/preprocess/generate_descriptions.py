"""Generate offline image/video descriptions for SigLIP text fusion.

BLIP is used only to enrich catalog metadata. SigLIP's text tower later
encodes the product title, PV attributes, and the generated description.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm


def sample_video_frames(path: Path, count: int) -> list[Image.Image]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto when omitted")
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Delayed imports keep data preparation usable without captioning packages.
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    records = json.loads(args.records.read_text(encoding="utf-8"))

    def caption(image: Image.Image) -> str:
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=32)
        return processor.decode(output[0], skip_special_tokens=True).strip()

    for row in tqdm(records.values(), desc="Captioning"):
        if row.get("description") and not args.overwrite:
            continue
        image_description = caption(Image.open(row["image_path"]).convert("RGB"))
        video_descriptions: list[str] = []
        if row.get("has_video"):
            for frame in sample_video_frames(Path(row["video_path"]), args.video_frames):
                video_descriptions.append(caption(frame))
        row["image_description"] = image_description
        row["video_description"] = " ".join(dict.fromkeys(video_descriptions))
        row["description"] = " ".join(
            value for value in (image_description, row["video_description"]) if value
        )
        row["text_input"] = " ".join(
            value for value in (row.get("title", ""), row["description"]) if value
        )

    args.records.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote captions to {args.records}")


if __name__ == "__main__":
    main()
