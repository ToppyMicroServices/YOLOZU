"""mAP target loading and validation.

Loads per-model-bucket expected-accuracy thresholds used by the
YOLO26 benchmark suite to gate pass/fail decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_map_targets_doc", "validate_map_targets_doc", "load_targets_map"]


_YOLO26_BUCKETS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_map_targets_doc(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    doc = json.loads(path.read_text())
    validate_map_targets_doc(doc)
    return doc


def validate_map_targets_doc(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise ValueError("targets file must be an object")

    metric_key = doc.get("metric_key")
    if metric_key is not None and (not isinstance(metric_key, str) or not metric_key):
        raise ValueError("targets.metric_key must be a non-empty string or null")

    protocol_id = doc.get("protocol_id")
    if protocol_id is not None and (not isinstance(protocol_id, str) or not protocol_id):
        raise ValueError("targets.protocol_id must be a non-empty string or null")

    imgsz = doc.get("imgsz")
    if imgsz is not None and not (isinstance(imgsz, int) and imgsz > 0):
        raise ValueError("targets.imgsz must be a positive int or null")

    targets = doc.get("targets")
    if not isinstance(targets, dict):
        raise ValueError("targets.targets must be an object")

    for bucket in _YOLO26_BUCKETS:
        if bucket not in targets:
            raise ValueError(f"targets.targets missing required key: {bucket}")

    for key, value in targets.items():
        if value is None:
            continue
        if not _is_number(value):
            raise ValueError(f"targets.targets.{key} must be a number or null")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"targets.targets.{key} must be in [0, 1] (got {value})")


def load_targets_map(path: str | Path) -> dict[str, float | None]:
    doc = load_map_targets_doc(path)
    raw = doc.get("targets") or {}
    out: dict[str, float | None] = {}
    for key, value in raw.items():
        if value is None:
            out[str(key)] = None
        else:
            out[str(key)] = float(value)
    return out

