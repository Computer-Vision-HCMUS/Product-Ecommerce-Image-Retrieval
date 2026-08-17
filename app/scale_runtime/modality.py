"""Five-modality product representation aligned with M5Product/SCALE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODALITY_NAMES = ("image", "text", "table", "video", "audio")


@dataclass
class ProductModalities:
    """Raw product inputs; any modality may be absent."""

    product_id: str = ""
    title: str = ""
    caption: str = ""
    pv: str = ""
    table_serialized: str = ""
    image_path: str | None = None
    video_path: str | None = None
    audio_path: str | None = None
    label: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ProductModalities:
        return cls(
            product_id=str(record.get("id", "")),
            title=str(record.get("title", "")).strip(),
            caption=str(record.get("description", "")).strip(),
            pv=str(record.get("pv", "")).strip(),
            table_serialized=str(record.get("table_serialized", "")).strip(),
            image_path=record.get("image_path"),
            video_path=record.get("video_path"),
            audio_path=record.get("audio_path"),
            label=str(record.get("label", "")).strip(),
        )

    def text_blob(self) -> str:
        return " ".join(part for part in (self.title, self.caption) if part)

    def table_blob(self) -> str:
        return self.table_serialized or self.pv

    def presence(self) -> dict[str, bool]:
        return {
            "image": bool(self.image_path and Path(str(self.image_path)).is_file()),
            "text": bool(self.text_blob()),
            "table": bool(self.table_blob()),
            "video": bool(self.video_path and Path(str(self.video_path)).is_file()),
            "audio": bool(self.audio_path and Path(str(self.audio_path)).is_file()),
        }
