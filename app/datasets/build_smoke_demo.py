"""Build a tiny on-disk dataset for pipeline/API smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "smoke_demo"
IMAGES = ROOT / "images"


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, str]] = {}
    manifest_lines: list[str] = []
    for index in range(40):
        product_id = f"demo_{index:03d}"
        label = "泊泉雅" if index < 20 else "假睫毛"
        image_path = IMAGES / f"{product_id}.jpg"
        color = (40 + index * 3, 80 + index * 2, 120 + index)
        Image.new("RGB", (224, 224), color).save(image_path, format="JPEG")
        metadata[product_id] = {
            "title": f"Demo product {index}",
            "label": label,
            "pv": "Brand#:#Demo#;#Color#:#Blue#;#Merchant#:#ShopA",
            "url": str(image_path),
            "video": "",
        }
        manifest_lines.append(
            json.dumps(
                {"id": product_id, "image_path": str(image_path.resolve()), "video_path": None},
                ensure_ascii=False,
            )
        )
    (ROOT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Wrote smoke demo dataset to {ROOT}")


if __name__ == "__main__":
    main()
