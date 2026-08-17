from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_WORK_DIR = Path(os.getenv("SCALE_WORK_DIR", "artifacts/downloaded_2k"))
DEFAULT_FUSION_WEIGHTS = Path(os.getenv("SCALE_FUSION_WEIGHTS", ""))


class SearchResultItem(BaseModel):
    id: str
    score: float
    title: str = ""
    label: str = ""
    image_path: str = ""
    modality_present: dict[str, bool] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    top_k: list[SearchResultItem]
    query_modalities: dict[str, bool]


class HealthResponse(BaseModel):
    status: str
    index_count: int
    records_count: int
    fusion_weights_loaded: bool


@lru_cache(maxsize=1)
def load_records(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))
