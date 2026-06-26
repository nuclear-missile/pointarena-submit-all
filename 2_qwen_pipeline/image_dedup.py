"""Helpers for stable image-level dedup keys across local cleaning pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def normalized_image_key_from_record(record: dict[str, Any]) -> str:
    raw_meta = record.get("raw_meta", {}) if isinstance(record, dict) else {}
    candidates = [
        raw_meta.get("image_sha256"),
        raw_meta.get("image_id"),
        record.get("image_sha256") if isinstance(record, dict) else None,
        record.get("image_id") if isinstance(record, dict) else None,
    ]
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text

    image_path = str(record.get("image_path") or "").strip() if isinstance(record, dict) else ""
    if image_path:
        path = Path(image_path)
        return path.stem or path.name

    return str(record.get("id") or "").strip() if isinstance(record, dict) else ""


def normalized_image_key_from_sample(sample: dict[str, Any]) -> str:
    record = sample.get("record", {}) if isinstance(sample, dict) else {}
    return normalized_image_key_from_record(record)


def normalized_image_key_from_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.suffix:
        return path.stem or path.name
    return text
