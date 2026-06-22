"""Dataset Contract v1 helpers.

Dataset records keep backend-neutral boxes at the dataset boundary.  The
preferred stored representation is absolute pixel ``xyxy_abs``; backend
adapters can derive YOLO ``cxcywh_norm`` or COCO ``xywh_abs`` views from it.
"""

from __future__ import annotations

import math
from typing import Any

DATASET_CONTRACT_VERSION = "1"
SUPPORTED_BBOX_FORMATS = ("xyxy_abs", "xywh_abs", "cxcywh_norm")


def image_wh_from_record(record: dict[str, Any]) -> tuple[float, float] | None:
    """Return ``(width, height)`` from common record size hints."""

    image_hw = record.get("image_hw") or record.get("hw")
    if isinstance(image_hw, (list, tuple)) and len(image_hw) >= 2:
        try:
            height = float(image_hw[0])
            width = float(image_hw[1])
        except (TypeError, ValueError):
            width = height = 0.0
        if width > 0.0 and height > 0.0:
            return width, height

    image_size = record.get("image_size")
    if isinstance(image_size, dict):
        try:
            width = float(image_size.get("width"))
            height = float(image_size.get("height"))
        except (TypeError, ValueError):
            width = height = 0.0
        if width > 0.0 and height > 0.0:
            return width, height
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        try:
            height = float(image_size[0])
            width = float(image_size[1])
        except (TypeError, ValueError):
            width = height = 0.0
        if width > 0.0 and height > 0.0:
            return width, height

    return None


def _finite_four(values: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out  # type: ignore[return-value]


def _dict_from_xyxy(x1: float, y1: float, x2: float, y2: float) -> dict[str, float | str]:
    return {"format": "xyxy_abs", "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def _dict_from_xywh(x: float, y: float, w: float, h: float) -> dict[str, float | str]:
    return {"format": "xywh_abs", "x": float(x), "y": float(y), "w": float(w), "h": float(h)}


def _dict_from_cxcywh(cx: float, cy: float, w: float, h: float) -> dict[str, float | str]:
    return {"format": "cxcywh_norm", "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)}


def _read_box_payload(payload: Any, *, explicit_format: str | None = None) -> tuple[str, tuple[float, float, float, float]] | None:
    if isinstance(payload, dict):
        fmt = str(payload.get("format") or explicit_format or "").strip()
        if not fmt:
            if all(k in payload for k in ("x1", "y1", "x2", "y2")):
                fmt = "xyxy_abs"
            elif all(k in payload for k in ("x", "y", "w", "h")):
                fmt = "xywh_abs"
            elif all(k in payload for k in ("cx", "cy", "w", "h")):
                fmt = "cxcywh_norm"
        if fmt == "xyxy_abs":
            values = _finite_four((payload.get("x1"), payload.get("y1"), payload.get("x2"), payload.get("y2")))  # type: ignore[arg-type]
        elif fmt == "xywh_abs":
            values = _finite_four((payload.get("x"), payload.get("y"), payload.get("w"), payload.get("h")))  # type: ignore[arg-type]
        elif fmt == "cxcywh_norm":
            values = _finite_four((payload.get("cx"), payload.get("cy"), payload.get("w"), payload.get("h")))  # type: ignore[arg-type]
        else:
            return None
        return (fmt, values) if values is not None else None

    if isinstance(payload, (list, tuple)) and len(payload) >= 4:
        fmt = str(explicit_format or "xyxy_abs").strip()
        values = _finite_four((payload[0], payload[1], payload[2], payload[3]))
        if fmt in SUPPORTED_BBOX_FORMATS and values is not None:
            return fmt, values
    return None


def _extract_source_box(label: dict[str, Any]) -> tuple[str, tuple[float, float, float, float]] | None:
    for key, fmt in (
        ("bbox_xyxy_abs", "xyxy_abs"),
        ("bbox_xywh_abs", "xywh_abs"),
        ("bbox_cxcywh_norm", "cxcywh_norm"),
    ):
        parsed = _read_box_payload(label.get(key), explicit_format=fmt)
        if parsed is not None:
            return parsed

    explicit_format = label.get("bbox_format")
    parsed = _read_box_payload(label.get("bbox"), explicit_format=str(explicit_format) if explicit_format else None)
    if parsed is not None:
        return parsed

    if all(k in label for k in ("x1", "y1", "x2", "y2")):
        return _read_box_payload(label, explicit_format="xyxy_abs")
    if all(k in label for k in ("x", "y", "w", "h")):
        return _read_box_payload(label, explicit_format="xywh_abs")
    if all(k in label for k in ("cx", "cy", "w", "h")):
        return _read_box_payload(label, explicit_format="cxcywh_norm")
    return None


def _to_xyxy_abs(fmt: str, values: tuple[float, float, float, float], image_wh: tuple[float, float] | None) -> tuple[float, float, float, float] | None:
    a, b, c, d = values
    if fmt == "xyxy_abs":
        return a, b, c, d
    if fmt == "xywh_abs":
        return a, b, a + c, b + d
    if fmt == "cxcywh_norm" and image_wh is not None:
        width, height = image_wh
        cx = a * width
        cy = b * height
        bw = c * width
        bh = d * height
        return cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
    return None


def _to_xywh_abs(fmt: str, values: tuple[float, float, float, float], image_wh: tuple[float, float] | None) -> tuple[float, float, float, float] | None:
    xyxy = _to_xyxy_abs(fmt, values, image_wh)
    if xyxy is None:
        return values if fmt == "xywh_abs" else None
    x1, y1, x2, y2 = xyxy
    return x1, y1, x2 - x1, y2 - y1


def _to_cxcywh_norm(fmt: str, values: tuple[float, float, float, float], image_wh: tuple[float, float] | None) -> tuple[float, float, float, float] | None:
    if fmt == "cxcywh_norm":
        return values
    if image_wh is None:
        return None
    width, height = image_wh
    if width <= 0.0 or height <= 0.0:
        return None
    xyxy = _to_xyxy_abs(fmt, values, image_wh)
    if xyxy is None:
        return None
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    return (x1 + bw / 2.0) / width, (y1 + bh / 2.0) / height, bw / width, bh / height


def normalize_label_bbox(
    label: dict[str, Any],
    *,
    image_wh: tuple[float, float] | None = None,
    bbox_field: str = "preserve",
) -> dict[str, Any]:
    """Return a label with Dataset Contract v1 bbox views.

    ``bbox_field`` controls the legacy ``bbox`` field:
    - ``preserve`` leaves it as provided.
    - ``xyxy_abs`` writes the preferred contract representation.
    - ``cxcywh_norm`` writes the backend-adapter view expected by YOLO-style trainers.
    """

    out = dict(label)
    parsed = _extract_source_box(out)
    if parsed is None:
        return out
    source_format, values = parsed
    out["bbox_source_format"] = str(source_format)

    xyxy = _to_xyxy_abs(source_format, values, image_wh)
    xywh = _to_xywh_abs(source_format, values, image_wh)
    cxcywh = _to_cxcywh_norm(source_format, values, image_wh)

    if xyxy is not None:
        out["bbox_xyxy_abs"] = _dict_from_xyxy(*xyxy)
        out["bbox_format"] = "xyxy_abs"
    elif source_format in SUPPORTED_BBOX_FORMATS:
        out["bbox_format"] = source_format
    if xywh is not None:
        out["bbox_xywh_abs"] = _dict_from_xywh(*xywh)
    if cxcywh is not None:
        cx, cy, w, h = cxcywh
        out["bbox_cxcywh_norm"] = _dict_from_cxcywh(cx, cy, w, h)
        out.setdefault("cx", float(cx))
        out.setdefault("cy", float(cy))
        out.setdefault("w", float(w))
        out.setdefault("h", float(h))

    if bbox_field == "xyxy_abs" and xyxy is not None:
        out["bbox"] = _dict_from_xyxy(*xyxy)
        out["bbox_format"] = "xyxy_abs"
    elif bbox_field == "cxcywh_norm" and cxcywh is not None:
        out["bbox"] = _dict_from_cxcywh(*cxcywh)
        out["bbox_format"] = "cxcywh_norm"

    return out


def normalize_record_bboxes(
    record: dict[str, Any],
    *,
    bbox_field: str = "preserve",
) -> dict[str, Any]:
    """Normalize all label boxes in a dataset record."""

    out = dict(record)
    image_wh = image_wh_from_record(out)
    labels = out.get("labels")
    if isinstance(labels, list):
        out["labels"] = [
            normalize_label_bbox(item, image_wh=image_wh, bbox_field=bbox_field)
            for item in labels
            if isinstance(item, dict)
        ]
    out["dataset_contract_version"] = DATASET_CONTRACT_VERSION
    out["bbox_storage_preference"] = "xyxy_abs"
    return out
