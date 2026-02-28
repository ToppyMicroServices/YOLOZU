"""Segmentation prediction I/O.

Normalises, validates, and loads segmentation prediction JSON in
multiple supported formats (list, wrapped, id-map).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "ValidationResult",
    "normalize_segmentation_predictions_json",
    "normalize_segmentation_predictions_payload",
    "validate_segmentation_predictions_entries",
    "validate_segmentation_predictions_payload",
    "load_segmentation_predictions_entries",
    "build_id_to_mask",
]

from .schema_governance import validate_payload_schema_version


@dataclass(frozen=True)
class ValidationResult:
    warnings: list[str]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_segmentation_predictions_json(data: Any) -> list[dict[str, Any]]:
    """Normalize supported segmentation prediction JSON shapes into a list of entries.

    Supported:
      1) [{"id": "...", "mask": "pred.png"}, ...]
      2) {"predictions": [ ...same as 1... ], ...}
      3) {"sample_id": "pred.png", ...}  (id->mask mapping)
    """

    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if isinstance(data, list):
        out: list[dict[str, Any]] = []
        for entry in data:
            if isinstance(entry, dict):
                out.append(entry)
        return out

    if isinstance(data, dict):
        out: list[dict[str, Any]] = []
        for sample_id, value in data.items():
            if isinstance(value, str):
                out.append({"id": str(sample_id), "mask": value})
            elif isinstance(value, dict):
                mask = value.get("mask")
                if isinstance(mask, str):
                    entry = dict(value)
                    entry["id"] = str(sample_id)
                    out.append(entry)
        return out

    raise ValueError("Unsupported segmentation predictions JSON format")


def normalize_segmentation_predictions_payload(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Normalize segmentation predictions JSON while preserving optional wrapped meta.

    Returns (entries, meta).
    """

    meta: dict[str, Any] | None = None
    if isinstance(data, dict) and "predictions" in data:
        raw_meta = data.get("meta")
        if raw_meta is not None:
            if not isinstance(raw_meta, dict):
                raise ValueError("meta must be an object when present")
            meta = raw_meta
        data = data["predictions"]

    return normalize_segmentation_predictions_json(data), meta


def validate_segmentation_predictions_entries(
    entries: Iterable[dict[str, Any]],
    *,
    where: str = "predictions",
) -> ValidationResult:
    warnings: list[str] = []
    seen: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{where}[{idx}]: entry must be an object")
        sample_id = entry.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{where}[{idx}]: missing 'id'")
        mask = entry.get("mask")
        if not isinstance(mask, str) or not mask:
            raise ValueError(f"{where}[{idx}]: missing 'mask'")
        if sample_id in seen:
            warnings.append(f"{where}[{idx}]: duplicate id: {sample_id}")
        seen.add(sample_id)
    return ValidationResult(warnings=warnings)


def validate_segmentation_predictions_payload(payload: Any) -> ValidationResult:
    warnings = validate_payload_schema_version(payload, artifact="segmentation_predictions")
    entries = normalize_segmentation_predictions_json(payload)
    res = validate_segmentation_predictions_entries(entries)
    return ValidationResult(warnings=[*warnings, *res.warnings])


def load_segmentation_predictions_entries(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries, meta = normalize_segmentation_predictions_payload(data)
    validate_segmentation_predictions_entries(entries)
    return entries, meta


def build_id_to_mask(entries: Iterable[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in entries:
        sample_id = entry.get("id")
        mask = entry.get("mask")
        if isinstance(sample_id, str) and isinstance(mask, str):
            out[sample_id] = mask
    return out

