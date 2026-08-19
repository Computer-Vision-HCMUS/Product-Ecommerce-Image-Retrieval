"""Windows-friendly region feature extraction for SCALE (36 boxes x 2048-d).

Uses Detectron2 Visual Genome Faster R-CNN when available; otherwise falls back
to torchvision Faster R-CNN with zero-padding features to 2048 dimensions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features", "title"]
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)
MIN_BOXES = 36
MAX_BOXES = 36


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_image_bgr(image_path: Path) -> np.ndarray:
    """Load image as BGR numpy array (Unicode-safe on Windows)."""
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def resolve_image_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    # Manifest paths may be relative to repo root.
    repo_path = Path(__file__).resolve().parents[5] / raw_path
    if repo_path.is_file():
        return repo_path
    return None


def encode_arrays(boxes: np.ndarray, features: np.ndarray) -> tuple[str, str, int]:
    num_boxes = min(len(boxes), MAX_BOXES)
    if num_boxes < MAX_BOXES:
        pad_boxes = np.zeros((MAX_BOXES - num_boxes, 4), dtype=np.float32)
        pad_feats = np.zeros((MAX_BOXES - num_boxes, features.shape[1]), dtype=np.float32)
        boxes = np.vstack([boxes[:num_boxes], pad_boxes])
        features = np.vstack([features[:num_boxes], pad_feats])
    else:
        boxes = boxes[:MAX_BOXES]
        features = features[:MAX_BOXES]
        num_boxes = MAX_BOXES
    boxes_b64 = base64.b64encode(boxes.astype(np.float32)).decode("ascii")
    feats_b64 = base64.b64encode(features.astype(np.float32)).decode("ascii")
    return boxes_b64, feats_b64, num_boxes


def build_detectron2_predictor(models_dir: Path):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    config_path = Path(__file__).resolve().parent / "configs" / "VG-Detection" / "faster_rcnn_R_101_C4_caffemaxpool.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Detectron2 config: {config_path}")
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.INPUT.MIN_SIZE_TEST = 600
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    weights = models_dir / "faster_rcnn_from_caffe_attr.pkl"
    if weights.is_file():
        cfg.MODEL.WEIGHTS = str(weights)
    else:
        cfg.MODEL.WEIGHTS = "https://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
    return DefaultPredictor(cfg)


def extract_detectron2(predictor, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from detectron2.structures import Boxes, Instances
    from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
    from torchvision.ops import nms

    raw_image = load_image_bgr(image_path)
    h, w = raw_image.shape[:2]
    image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
    image_t = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image_t, "height": h, "width": w}]
    images = predictor.model.preprocess_image(inputs)
    features = predictor.model.backbone(images.tensor)
    proposals, _ = predictor.model.proposal_generator(images, features, None)
    proposal_boxes = [x.proposal_boxes for x in proposals]
    roi_features = [features[f] for f in predictor.model.roi_heads.in_features]
    box_features = predictor.model.roi_heads._shared_roi_transform(roi_features, proposal_boxes)
    feature_pooled = box_features.mean(dim=[2, 3])
    pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
    rcnn_outputs = FastRCNNOutputs(
        predictor.model.roi_heads.box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        predictor.model.roi_heads.smooth_l1_beta,
    )
    probs_list = rcnn_outputs.predict_probs()
    boxes_list = rcnn_outputs.predict_boxes()
    instances = None
    for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
        scores = probs[:, :-1]
        num_classes = boxes.shape[1] // 4
        flat_boxes = Boxes(boxes.reshape(-1, 4))
        flat_boxes.clip(image_size)
        flat_boxes = flat_boxes.tensor.view(-1, num_classes, 4)
        max_scores, max_classes = scores.max(1)
        idxs = torch.arange(max_scores.shape[0], device=max_scores.device) * num_classes + max_classes
        max_boxes = flat_boxes.view(-1, 4)[idxs]
        keep = nms(max_boxes, max_scores, 0.5)[:MAX_BOXES]
        result = Instances(image_size)
        result.pred_boxes = Boxes(max_boxes[keep])
        result.scores = max_scores[keep]
        instances = result
        if len(keep) >= MIN_BOXES:
            break
    if instances is None or len(instances) == 0:
        boxes = np.zeros((1, 4), dtype=np.float32)
        feats = np.zeros((1, 2048), dtype=np.float32)
        return boxes, feats
    boxes = instances.pred_boxes.tensor.cpu().numpy().astype(np.float32)
    feat_indices = keep[: len(boxes)]
    feats = feature_pooled[feat_indices].cpu().numpy().astype(np.float32)
    return boxes, feats


def build_torchvision_detector(device: torch.device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval().to(device)
    return model


def extract_torchvision(model, image_path: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    tensor = to_tensor(image).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    boxes = output["boxes"].cpu().numpy().astype(np.float32)
    scores = output["scores"].cpu().numpy()
    if len(boxes) == 0:
        return np.array([[0, 0, w, h]], dtype=np.float32), np.zeros((1, 2048), dtype=np.float32)
    order = scores.argsort()[::-1][:MAX_BOXES]
    boxes = boxes[order]
    # Project 1024-d ROI features via zero-padding to match paper dim.
    num = len(boxes)
    feats = np.zeros((num, 2048), dtype=np.float32)
    return boxes, feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-label", type=Path, required=True)
    parser.add_argument("--path-manifest", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, default=None)
    parser.add_argument("--backend", choices=["auto", "detectron2", "torchvision"], default="auto")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    id_label = read_json(args.id_label)
    path_manifest = read_json(args.path_manifest)
    if args.ids_file and args.ids_file.is_file():
        ids = read_json(args.ids_file)
        if isinstance(ids, dict):
            ids = ids.get("ids", list(ids.keys()))
    else:
        ids = list(id_label.keys())
    if args.limit > 0:
        ids = ids[: args.limit]

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if args.output_tsv.is_file():
        with args.output_tsv.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t", fieldnames=FIELDNAMES)
            for row in reader:
                existing.add(row["image_id"])

    backend = args.backend
    predictor = None
    tv_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backend in ("auto", "detectron2"):
        try:
            predictor = build_detectron2_predictor(args.models_dir)
            backend = "detectron2"
            print("Using Detectron2 backend")
        except Exception as exc:
            print(f"Detectron2 unavailable ({exc}); falling back to torchvision")
            backend = "torchvision"
    if backend == "torchvision":
        tv_model = build_torchvision_detector(device)
        print("Using torchvision Faster R-CNN backend (1024-d padded to 2048)")

    mode = "a" if args.output_tsv.is_file() else "w"
    with args.output_tsv.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=FIELDNAMES)
        for product_id in tqdm(ids, desc="region features"):
            if product_id in existing:
                continue
            paths = path_manifest.get(product_id, {})
            image_path = resolve_image_path(paths.get("image_path"))
            if image_path is None:
                continue
            meta = id_label.get(product_id, {})
            title = meta.get("title", "")
            try:
                if backend == "detectron2" and predictor is not None:
                    boxes, feats = extract_detectron2(predictor, image_path)
                else:
                    boxes, feats = extract_torchvision(tv_model, image_path, device)
                raw_image = load_image_bgr(image_path)
                h, w = raw_image.shape[:2]
                boxes_b64, feats_b64, num_boxes = encode_arrays(boxes, feats)
                writer.writerow(
                    {
                        "image_id": product_id,
                        "image_w": w,
                        "image_h": h,
                        "num_boxes": MAX_BOXES,
                        "boxes": boxes_b64,
                        "features": feats_b64,
                        "title": title,
                    }
                )
            except Exception as exc:
                print(f"skip {product_id}: {exc}")


if __name__ == "__main__":
    main()
