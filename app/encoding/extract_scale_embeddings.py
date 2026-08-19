"""Export SCALE five-modality fused embeddings for catalog splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scale_runtime.fusion_encoder import ScaleFusionEncoder
from scale_runtime.modality import ProductModalities


def export_split(
    encoder: ScaleFusionEncoder,
    records: dict[str, dict[str, object]],
    split_file: Path,
    output_dir: Path,
) -> None:
    split = json.loads(split_file.read_text(encoding="utf-8"))
    embeddings: list[np.ndarray] = []
    exported_ids: list[str] = []
    modality_stats = {name: 0 for name in ("image", "text", "table", "video", "audio")}
    for product_id in tqdm(split["ids"], desc=f"SCALE embed {split['split']}"):
        record = records.get(product_id)
        if not record:
            continue
        product = ProductModalities.from_record(record)
        try:
            vector, presence = encoder.encode_product(product)
        except (OSError, ValueError) as exc:
            print(f"Skipping {product_id}: {exc}")
            continue
        for name, present in presence.items():
            if present:
                modality_stats[name] += 1
        embeddings.append(vector)
        exported_ids.append(product_id)
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
                "encoder": "scale_fusion",
                "modalities": list(modality_stats.keys()),
                "modality_counts": modality_stats,
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
    parser.add_argument("--splits", nargs="+", default=("gallery", "test"))
    parser.add_argument("--fusion-weights", type=Path, default=None)
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    encoder = ScaleFusionEncoder(args.model, args.device)
    if args.fusion_weights and args.fusion_weights.is_file():
        encoder.load_weights(args.fusion_weights)
    for split in args.splits:
        export_split(encoder, records, args.splits_dir / f"{split}.json", args.output_dir / split)


if __name__ == "__main__":
    main()
