"""Contrastive fine-tuning of SCALE fusion weights on the train split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scale_runtime.fusion_encoder import ScaleFusionEncoder
from scale_runtime.modality import ProductModalities


def supervised_contrastive_loss(
    vectors: torch.Tensor, labels: list[str], temperature: float = 0.07
) -> torch.Tensor:
    vectors = F.normalize(vectors, dim=1)
    logits = vectors @ vectors.T / temperature
    label_to_id = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    label_tensor = torch.tensor(
        [label_to_id[label] for label in labels], device=vectors.device, dtype=torch.long
    )
    positives = label_tensor[:, None] == label_tensor[None, :]
    positives.fill_diagonal_(False)
    self_mask = torch.eye(len(labels), device=vectors.device, dtype=torch.bool)
    negatives = ~positives & ~self_mask
    losses = []
    for index in range(len(labels)):
        pos = logits[index][positives[index]]
        neg = logits[index][negatives[index]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        denom = torch.logsumexp(torch.cat([pos, neg]), dim=0)
        losses.append(-(torch.logsumexp(pos, dim=0) - denom))
    if not losses:
        return torch.tensor(0.0, device=vectors.device, requires_grad=True)
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    ids = [product_id for product_id in split["ids"] if product_id in records]
    if len(ids) < 2:
        raise SystemExit("Need at least two train samples to fine-tune fusion weights.")

    encoder = ScaleFusionEncoder(device=args.device)
    encoder.train()
    optimizer = torch.optim.Adam([encoder.modality_weights], lr=args.lr)

    rng = random.Random(42)
    for epoch in range(args.epochs):
        rng.shuffle(ids)
        losses: list[float] = []
        for start in tqdm(range(0, len(ids), args.batch_size), desc=f"Epoch {epoch + 1}"):
            batch_ids = ids[start : start + args.batch_size]
            modality_rows = []
            masks = []
            labels = []
            for product_id in batch_ids:
                matrix, mask = encoder.encode_modality_matrix(
                    ProductModalities.from_record(records[product_id])
                )
                modality_rows.append(matrix)
                masks.append(mask)
                labels.append(records[product_id]["label"])
            vectors = torch.from_numpy(np.stack(modality_rows)).float().to(encoder.device)
            mask_tensor = torch.from_numpy(np.stack(masks)).float().to(encoder.device)
            fused = encoder.fuse_modality_matrix_torch(vectors, mask_tensor)
            loss = supervised_contrastive_loss(fused, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.4f}")

    encoder.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(args.output)
    print(f"Saved fusion weights to {args.output}")


if __name__ == "__main__":
    main()
