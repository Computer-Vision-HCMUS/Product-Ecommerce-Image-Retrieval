"""Export fused SigLIP embeddings for downloaded_2k splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from siglip_encoder import SiglipEncoder
except ImportError:
    from encoding.siglip_encoder import SiglipEncoder


def product_text(record: dict[str, object]) -> str:
    """Keep product title, structured attributes, and optional BLIP caption."""
    return " ".join(
        str(value).strip()
        for value in (
            record.get("title", ""),
            record.get("pv", ""),
            record.get("description", ""),
        )
        if str(value).strip()
    )


def flush_batch(
    encoder: SiglipEncoder,
    images: list[Image.Image],
    texts: list[str],
    ids: list[str],
    output: list[np.ndarray],
    output_ids: list[str],
) -> None:
    if not images:
        return
    output.append(encoder.fuse(encoder.encode_images(images), encoder.encode_texts(texts)))
    output_ids.extend(ids)
    for image in images:
        image.close()
    images.clear()
    texts.clear()
    ids.clear()


def export_split(
    encoder: SiglipEncoder,
    records: dict[str, dict[str, object]],
    split_file: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    split = json.loads(split_file.read_text(encoding="utf-8"))
    embeddings: list[np.ndarray] = []
    exported_ids: list[str] = []
    images: list[Image.Image] = []
    texts: list[str] = []
    ids: list[str] = []
    for product_id in tqdm(split["ids"], desc=f"Embedding {split['split']}"):
        record = records[product_id]
        try:
            images.append(Image.open(str(record["image_path"])).convert("RGB"))
        except (OSError, ValueError) as exc:
            print(f"Skipping unreadable image {product_id}: {exc}")
            continue
        texts.append(product_text(record))
        ids.append(product_id)
        if len(images) >= batch_size:
            flush_batch(encoder, images, texts, ids, embeddings, exported_ids)
    flush_batch(encoder, images, texts, ids, embeddings, exported_ids)
    if not embeddings:
        raise RuntimeError(f"No embeddings exported for {split_file}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embedding.npy", np.vstack(embeddings))
    np.save(output_dir / "id.npy", np.asarray(exported_ids, dtype=str))
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "split": split["split"],
                "count": len(exported_ids),
                "model": encoder.model_name,
                "image_weight": encoder.image_weight,
                "text_weight": 1 - encoder.image_weight,
                "text_fields": ["title", "pv", "description"],
                "positive_definition": split["positive_definition"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("gallery", "query"))
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--image-weight", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=4, help="Use 1-4 for a 4 GB GPU.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    encoder = SiglipEncoder(args.model, args.device, args.image_weight)
    for split in args.splits:
        export_split(
            encoder,
            records,
            args.splits_dir / f"{split}.json",
            args.output_dir / split,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
