from __future__ import annotations

from typing import Any


def cxcywh_norm_to_xyxy_abs(
    cxcywh: tuple[float, float, float, float], *, width: int | float, height: int | float
) -> tuple[float, float, float, float]:
    cx, cy, w, h = cxcywh
    cx_abs = float(cx) * float(width)
    cy_abs = float(cy) * float(height)
    w_abs = float(w) * float(width)
    h_abs = float(h) * float(height)
    x1 = cx_abs - w_abs / 2.0
    y1 = cy_abs - h_abs / 2.0
    x2 = cx_abs + w_abs / 2.0
    y2 = cy_abs + h_abs / 2.0
    return float(x1), float(y1), float(x2), float(y2)


def iou_xyxy_abs(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), float(by2))

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aw = max(0.0, float(ax2) - float(ax1))
    ah = max(0.0, float(ay2) - float(ay1))
    bw = max(0.0, float(bx2) - float(bx1))
    bh = max(0.0, float(by2) - float(by1))
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def iou_cxcywh_norm_dict(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1 = float(a["cx"]) - float(a["w"]) / 2.0
    ay1 = float(a["cy"]) - float(a["h"]) / 2.0
    ax2 = float(a["cx"]) + float(a["w"]) / 2.0
    ay2 = float(a["cy"]) + float(a["h"]) / 2.0

    bx1 = float(b["cx"]) - float(b["w"]) / 2.0
    by1 = float(b["cy"]) - float(b["h"]) / 2.0
    bx2 = float(b["cx"]) + float(b["w"]) / 2.0
    by2 = float(b["cy"]) + float(b["h"]) / 2.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def xyxy_to_cxcywh_abs(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def cxcywh_abs_to_norm(
    cxcywh: tuple[float, float, float, float], *, width: int | float, height: int | float
) -> tuple[float, float, float, float]:
    cx, cy, w, h = cxcywh
    width, height = float(width), float(height)
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")
    return cx / width, cy / height, w / width, h / height


def xyxy_abs_to_cxcywh_norm(
    xyxy: tuple[float, float, float, float], *, width: int | float, height: int | float
) -> tuple[float, float, float, float]:
    return cxcywh_abs_to_norm(xyxy_to_cxcywh_abs(xyxy), width=width, height=height)
