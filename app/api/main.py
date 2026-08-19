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
from indexing.search import ProductIndex
from scale_runtime.fusion_encoder import ScaleFusionEncoder
from scale_runtime.modality import ProductModalities

BACKEND = os.getenv("SCALE_BACKEND", "siglip").lower()

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
def get_index() -> ProductIndex:
    return ProductIndex(INDEX_DIR)


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
    index = get_index()
    return HealthResponse(
        status="ok",
        index_count=len(index._ids),
        records_count=len(records),
        fusion_weights_loaded=BACKEND != "paper" and FUSION_WEIGHTS.is_file(),
    )


@app.get("/file")
def serve_file(path: str = Query(...)) -> FileResponse:
    file_path = Path(path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)


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
        embedding, presence = get_encoder().encode_product(product)
        records = get_records()
        results = []
        for product_id, score in get_index().search(embedding, top_k):
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
