"""Shared paths and modes for the local retrieval service."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class PipelineMode(str, Enum):
    BASELINE = "baseline"
    IMPROVED = "improved"

    @classmethod
    def from_value(cls, value: str | "PipelineMode") -> "PipelineMode":
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass(frozen=True)
class PipelinePaths:
    work_dir: Path
    mode: PipelineMode

    def __init__(self, work_dir: Path | str, mode: PipelineMode | str) -> None:
        object.__setattr__(self, "work_dir", Path(work_dir))
        object.__setattr__(self, "mode", PipelineMode.from_value(mode))

    @property
    def suffix(self) -> str:
        return "" if self.mode is PipelineMode.BASELINE else "_improved"

    @property
    def retrieval_results(self) -> Path:
        return self.work_dir / f"retrieval_results{self.suffix}"

    @property
    def retrieval_metric(self) -> Path:
        return self.work_dir / f"retrieval_metric{self.suffix}"

    @property
    def evaluation_benchmark(self) -> Path:
        return self.work_dir / f"evaluation_benchmark{self.suffix}.json"

    @property
    def index_dir(self) -> Path:
        return self.work_dir / f"index_hnsw{self.suffix}"
