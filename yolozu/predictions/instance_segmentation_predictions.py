"""Instance-segmentation prediction I/O.

Normalises, validates, and loads instance-segmentation prediction
JSON payloads into a canonical entry format suitable for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "ValidationResult",
    "normalize_instance_segmentation_predictions_json",
    "normalize_instance_segmentation_predictions_payload",
    "validate_instance_segmentation_predictions_entries",
    "validate_instance_segmentation_predictions_payload",
    "load_instance_segmentation_predictions_entries",
    "iter_instances",
]

from .schema_governance import validate_payload_schema_version

_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)


@dataclass(frozen=True)
class ValidationResult:
    warnings: list[str]


def normalize_instance_segmentation_predictions_json(data: Any) -> list[dict[str, Any]]:
    """Normalize supported instance-segmentation prediction JSON shapes into entries.

    Supported shapes:
      1) [{"image": "...", "instances": [...]}, ...]
      2) {"predictions": [ ...same as 1... ], "meta": {...}}
      3) {"image.jpg": {"instances":[...]} , ...} (image->entry mapping)
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
        for image, value in data.items():
            if isinstance(value, dict):
                inst = value.get("instances")
                out.append({"image": str(image), "instances": inst if isinstance(inst, list) else []})
            elif isinstance(value, list):
                out.append({"image": str(image), "instances": value})
        return out

    raise ValueError("Unsupported instance-segmentation predictions JSON format")


def normalize_instance_segmentation_predictions_payload(
    data: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Normalize predictions while preserving optional wrapper meta.

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

    return normalize_instance_segmentation_predictions_json(data), meta


def validate_instance_segmentation_predictions_entries(
    entries: Iterable[dict[str, Any]],
    *,
    where: str = "predictions",
) -> ValidationResult:
    warnings: list[str] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{where}[{idx}]: entry must be an object")
        image = entry.get("image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"{where}[{idx}]: missing 'image'")
        instances = entry.get("instances")
        if instances is None:
            instances = []
        if not isinstance(instances, list):
            raise ValueError(f"{where}[{idx}].instances: must be a list")

        for j, inst in enumerate(instances):
            if not isinstance(inst, dict):
                raise ValueError(f"{where}[{idx}].instances[{j}]: must be an object")
            if "mask" not in inst:
                raise ValueError(f"{where}[{idx}].instances[{j}]: missing 'mask'")
            mask = inst.get("mask")
            if not isinstance(mask, str) or not mask:
                raise ValueError(f"{where}[{idx}].instances[{j}].mask: must be a non-empty string")
            if "class_id" not in inst:
                raise ValueError(f"{where}[{idx}].instances[{j}]: missing 'class_id'")
            try:
                int(inst.get("class_id"))
            except _NUMERIC_ERRORS:
                raise ValueError(f"{where}[{idx}].instances[{j}].class_id: must be int-like")
            if "score" in inst:
                try:
                    float(inst.get("score"))
                except _NUMERIC_ERRORS:
                    raise ValueError(f"{where}[{idx}].instances[{j}].score: must be float-like")
            else:
                warnings.append(f"{where}[{idx}].instances[{j}]: missing 'score' (defaulting to 1.0 in evaluation)")

    return ValidationResult(warnings=warnings)


def validate_instance_segmentation_predictions_payload(payload: Any) -> ValidationResult:
    warnings = validate_payload_schema_version(payload, artifact="instance_segmentation_predictions")
    entries = normalize_instance_segmentation_predictions_json(payload)
    res = validate_instance_segmentation_predictions_entries(entries)
    return ValidationResult(warnings=[*warnings, *res.warnings])


def load_instance_segmentation_predictions_entries(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries, meta = normalize_instance_segmentation_predictions_payload(data)
    validate_instance_segmentation_predictions_entries(entries)
    return entries, meta


def iter_instances(entries: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield flattened instances with an 'image' key attached."""
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not isinstance(image, str) or not image:
            continue
        insts = entry.get("instances") or []
        if not isinstance(insts, list):
            continue
        for inst in insts:
            if isinstance(inst, dict):
                out = dict(inst)
                out["image"] = image
                yield out
