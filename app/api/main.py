from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import HealthResponse, SearchResponse, SearchResultItem, DEFAULT_FUSION_WEIGHTS, DEFAULT_WORK_DIR
from indexing.catalog_hybrid import MetadataReranker
from indexing.improved_search import build_search_backend
from indexing.pipeline_config import PipelineMode
from indexing.rerank import RerankWeights
from scale_runtime.fusion_encoder import ScaleFusionEncoder
from scale_runtime.modality import ProductModalities

BACKEND = os.getenv("SCALE_BACKEND", "siglip").lower()
PIPELINE_MODE = PipelineMode.from_value(os.getenv("SCALE_PIPELINE", "baseline"))
RERANK_CANDIDATES = int(os.getenv("SCALE_RERANK_CANDIDATES", "100"))
RERANK_LAMBDA = float(os.getenv("SCALE_RERANK_LAMBDA", "0.7"))

app = FastAPI(title="SCALE Product Retrieval API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path(os.getenv("SCALE_WORK_DIR", str(DEFAULT_WORK_DIR)))
RECORDS_PATH = Path(os.getenv("SCALE_RECORDS", str(WORK_DIR / "records.json")))
INDEX_DIR = Path(os.getenv("SCALE_INDEX_DIR", str(WORK_DIR / "index_hnsw")))
FUSION_WEIGHTS = Path(os.getenv("SCALE_FUSION_WEIGHTS", str(DEFAULT_FUSION_WEIGHTS)))


@lru_cache(maxsize=1)
def get_encoder():
    if BACKEND == "paper":
        from scale_paper.encoder import ScalePaperEncoder
        return ScalePaperEncoder(WORK_DIR)
    encoder = ScaleFusionEncoder(device=os.getenv("SCALE_DEVICE"))
    if FUSION_WEIGHTS.is_file():
        encoder.load_weights(FUSION_WEIGHTS)
    return encoder


@lru_cache(maxsize=1)
def get_search_backend():
    weights = RerankWeights(lambda_emb=RERANK_LAMBDA)
    return build_search_backend(
        WORK_DIR,
        PIPELINE_MODE,
        candidate_n=RERANK_CANDIDATES,
        weights=weights,
    )


@lru_cache(maxsize=1)
def get_metadata_reranker() -> MetadataReranker:
    return MetadataReranker(WORK_DIR, lambda_emb=RERANK_LAMBDA)


def rank_query(embedding, *, title: str, pv: str, top_k: int) -> list[tuple[str, float]]:
    """Run the single serving protocol used by catalog and raw queries."""
    candidates = get_search_backend().search(embedding, max(top_k, RERANK_CANDIDATES))
    return get_metadata_reranker().rerank(candidates, title=title, pv=pv, top_k=top_k)


@lru_cache(maxsize=1)
def get_records() -> dict[str, dict]:
    if BACKEND == "paper":
        id_label_path = WORK_DIR / "id_label.json"
        path_manifest_path = WORK_DIR / "path_manifest.json"
        if not id_label_path.is_file():
            raise FileNotFoundError(f"Missing id_label file: {id_label_path}")
        id_label = json.loads(id_label_path.read_text(encoding="utf-8"))
        paths = {}
        if path_manifest_path.is_file():
            paths = json.loads(path_manifest_path.read_text(encoding="utf-8"))
        records = {}
        for pid, meta in id_label.items():
            p = paths.get(pid, {})
            records[pid] = {
                **meta,
                "image_path": p.get("image_path") or "",
                "video_path": p.get("video_path"),
                "has_video": bool(p.get("video_path")),
            }
        return records
    if not RECORDS_PATH.is_file():
        raise FileNotFoundError(f"Missing records file: {RECORDS_PATH}")
    return json.loads(RECORDS_PATH.read_text(encoding="utf-8"))


async def _save_upload(upload: UploadFile | None, suffix: str) -> str | None:
    if upload is None or not upload.filename:
        return None
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(await upload.read())
    temp.close()
    return temp.name


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    records = get_records()
    backend = get_search_backend()
    if hasattr(backend, "_ids"):
        index_count = len(backend._ids)
    elif hasattr(backend, "_index") and hasattr(backend._index, "_ids"):
        index_count = len(backend._index._ids)
    else:
        raise RuntimeError("Search backend does not expose its indexed IDs.")
    return HealthResponse(
        status="ok",
        index_count=index_count,
        records_count=len(records),
        fusion_weights_loaded=BACKEND != "paper" and FUSION_WEIGHTS.is_file(),
    )


@app.get("/file")
def serve_file(path: str = Query(...)) -> FileResponse:
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)


@app.get("/products/{product_id}")
def product_detail(product_id: str) -> dict:
    """Return the five-modality metadata required by the product detail view."""
    item = get_records().get(product_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Product not found.")
    audio_path = WORK_DIR / "audios" / f"{product_id}.mp3"
    return {
        "id": product_id,
        "title": item.get("title", ""),
        "label": item.get("label", ""),
        "super_category": item.get("super_category", ""),
        "pv": item.get("pv", ""),
        "masked_modalities": item.get("masked_modalities", []),
        "modalities": {
            "image": item.get("image_path", ""),
            "text": bool(str(item.get("title", "")).strip()),
            "pv": bool(str(item.get("pv", "")).strip()),
            "video": item.get("video_path") or "",
            "audio": str(audio_path) if audio_path.is_file() else "",
        },
    }


@app.post("/search/by-id", response_model=SearchResponse)
async def search_by_id(
    product_id: Annotated[str, Form()],
    top_k: Annotated[int, Form()] = 10,
) -> SearchResponse:
    """Run the same I/T/Tab -> HNSW -> reranking protocol as ``/search``."""
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive.")
    product_id = product_id.strip()
    item = get_records().get(product_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Product ID is not in the serving catalog.")
    if BACKEND != "paper":
        raise HTTPException(status_code=501, detail="Catalog-ID retrieval requires the paper backend.")
    encoder = get_encoder()
    product = ProductModalities(title=str(item.get("title", "")).strip(), pv=str(item.get("pv", "")).strip(), image_path=item.get("image_path"))
    embedding, presence = encoder.encode_product(product)
    raw_hits = rank_query(embedding, title=product.title, pv=product.pv, top_k=top_k)
    results = []
    records = get_records()
    for hit_id, score in raw_hits:
        hit = records.get(hit_id, {})
        results.append(SearchResultItem(
            id=hit_id,
            score=score,
            title=hit.get("title", ""),
            label=hit.get("label", ""),
            image_path=hit.get("image_path", ""),
            modality_present={
                "image": bool(hit.get("image_path")),
                "text": bool(str(hit.get("title", "")).strip()),
                "table": bool(str(hit.get("pv", "")).strip()),
                "video": bool(hit.get("has_video")),
                "audio": bool(hit.get("audio_available")),
            },
        ))
    return SearchResponse(top_k=results, query_modalities=presence)


@app.post("/search", response_model=SearchResponse)
async def search(
    top_k: Annotated[int, Form()] = 10,
    title: Annotated[str, Form()] = "",
    caption: Annotated[str, Form()] = "",
    pv: Annotated[str, Form()] = "",
    table_serialized: Annotated[str, Form()] = "",
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
) -> SearchResponse:
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive.")
    image_path = await _save_upload(image, ".jpg")
    video_path = await _save_upload(video, ".mp4")
    audio_path = await _save_upload(audio, ".wav")

    product = ProductModalities(
        title=title.strip(),
        caption=caption.strip(),
        pv=pv.strip(),
        table_serialized=table_serialized.strip(),
        image_path=image_path,
        video_path=video_path,
        audio_path=audio_path,
    )
    if not any(product.presence().values()):
        raise HTTPException(status_code=400, detail="Provide at least one modality.")

    try:
        encoder = get_encoder()
        embedding, presence = encoder.encode_product(product)
        records = get_records()
        raw_hits = rank_query(embedding, title=product.title, pv=product.table_blob(), top_k=top_k)

        results = []
        for product_id, score in raw_hits:
            item = records.get(product_id, {})
            results.append(
                SearchResultItem(
                    id=product_id,
                    score=score,
                    title=item.get("title", ""),
                    label=item.get("label", ""),
                    image_path=item.get("image_path", ""),
                    modality_present={
                        "image": bool(item.get("image_path")),
                        "text": bool(str(item.get("title", "")).strip() or str(item.get("description", "")).strip()),
                        "table": bool(str(item.get("pv", "")).strip()),
                        "video": bool(item.get("has_video")),
                        "audio": bool(item.get("audio_path")),
                    },
                )
            )
    finally:
        for path in (image_path, video_path, audio_path):
            if path and Path(path).exists():
                Path(path).unlink(missing_ok=True)

    return SearchResponse(top_k=results, query_modalities=presence)
