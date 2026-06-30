"""BBox interpretation helpers for prediction artifacts."""

from __future__ import annotations

import math
from typing import Any

from yolozu.boxes import cxcywh_norm_to_xyxy_abs

SUPPORTED_PREDICTION_BBOX_FORMATS = ("auto", "cxcywh_norm", "cxcywh_abs", "xywh_abs", "xyxy_abs")


def _finite_four(values: tuple[Any, Any, Any, Any]) -> tuple[float, float, float, float] | None:
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in out):
        return None
    return out  # type: ignore[return-value]


def infer_bbox_format(bbox: dict[str, Any], *, default_format: str = "cxcywh_norm") -> str:
    fmt = str(bbox.get("format") or "").strip()
    if fmt:
        return fmt
    if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
        return "xyxy_abs"
    if all(k in bbox for k in ("x", "y", "w", "h")):
        return "xywh_abs"
    if all(k in bbox for k in ("cx", "cy", "w", "h")):
        return str(default_format or "cxcywh_norm")
    return ""


def bbox_to_xyxy_abs(
    bbox: dict[str, Any],
    *,
    width: int | float,
    height: int | float,
    bbox_format: str = "auto",
) -> tuple[float, float, float, float] | None:
    """Convert a prediction bbox dict to absolute ``xyxy`` coordinates."""

    fmt = infer_bbox_format(bbox) if bbox_format == "auto" else str(bbox_format)
    values: tuple[float, float, float, float] | None
    if fmt == "cxcywh_norm":
        values = _finite_four((bbox.get("cx"), bbox.get("cy"), bbox.get("w"), bbox.get("h")))
        if values is None:
            return None
        return cxcywh_norm_to_xyxy_abs(values, width=int(width), height=int(height))
    if fmt == "cxcywh_abs":
        values = _finite_four((bbox.get("cx"), bbox.get("cy"), bbox.get("w"), bbox.get("h")))
        if values is None:
            return None
        cx, cy, bw, bh = values
        return cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
    if fmt == "xywh_abs":
        values = _finite_four((bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h")))
        if values is None:
            return None
        x, y, bw, bh = values
        return x, y, x + bw, y + bh
    if fmt == "xyxy_abs":
        values = _finite_four((bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")))
        if values is None:
            return None
        return values
    return None


def bbox_to_cxcywh_norm(
    bbox: dict[str, Any],
    *,
    width: int | float,
    height: int | float,
    bbox_format: str = "auto",
) -> tuple[float, float, float, float] | None:
    """Convert a prediction bbox dict to normalized ``cxcywh`` coordinates."""

    fmt = infer_bbox_format(bbox) if bbox_format == "auto" else str(bbox_format)
    if fmt == "cxcywh_norm":
        return _finite_four((bbox.get("cx"), bbox.get("cy"), bbox.get("w"), bbox.get("h")))
    xyxy = bbox_to_xyxy_abs(bbox, width=width, height=height, bbox_format=fmt)
    if xyxy is None:
        return None
    w_img = float(width)
    h_img = float(height)
    if w_img <= 0.0 or h_img <= 0.0:
        return None
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    return (x1 + bw / 2.0) / w_img, (y1 + bh / 2.0) / h_img, bw / w_img, bh / h_img
