"""Letterbox padding / scaling utilities.

Computes the padding, scale, and offset needed to fit an image into a
fixed input size while preserving aspect ratio, and maps coordinates
between input-space and original-image-space.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Letterbox",
    "compute_letterbox",
    "input_xyxy_to_orig_xyxy",
    "orig_xyxy_to_cxcywh_norm",
]


@dataclass(frozen=True)
class Letterbox:
    input_size: int
    scale: float
    pad_x: float
    pad_y: float
    new_w: int
    new_h: int


def compute_letterbox(
    *,
    orig_w: int,
    orig_h: int,
    input_size: int | tuple[int, int] = 640,
) -> Letterbox:
    """Compute letterbox parameters.

    ``input_size`` may be a single int (square) **or** a (width, height) tuple
    for rectangular targets.
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("invalid image size")
    if isinstance(input_size, (list, tuple)):
        target_w, target_h = int(input_size[0]), int(input_size[1])
    else:
        target_w = target_h = int(input_size)
    if target_w <= 0 or target_h <= 0:
        raise ValueError("input_size must be positive")
    scale = min(float(target_w) / float(orig_w), float(target_h) / float(orig_h))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_w = float(target_w) - float(new_w)
    pad_h = float(target_h) - float(new_h)
    pad_x = pad_w / 2.0
    pad_y = pad_h / 2.0
    # Match Ultralytics LetterBox rounding for top/left padding.
    left = float(round(pad_x - 0.1))
    top = float(round(pad_y - 0.1))
    return Letterbox(
        input_size=int(target_w),
        scale=float(scale),
        pad_x=float(left),
        pad_y=float(top),
        new_w=int(new_w),
        new_h=int(new_h),
    )


def input_xyxy_to_orig_xyxy(
    xyxy: tuple[float, float, float, float], *, letterbox: Letterbox, orig_w: int, orig_h: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x1 = (float(x1) - letterbox.pad_x) / letterbox.scale
    y1 = (float(y1) - letterbox.pad_y) / letterbox.scale
    x2 = (float(x2) - letterbox.pad_x) / letterbox.scale
    y2 = (float(y2) - letterbox.pad_y) / letterbox.scale

    x1 = max(0.0, min(float(orig_w), x1))
    y1 = max(0.0, min(float(orig_h), y1))
    x2 = max(0.0, min(float(orig_w), x2))
    y2 = max(0.0, min(float(orig_h), y2))
    return x1, y1, x2, y2


def orig_xyxy_to_cxcywh_norm(
    xyxy: tuple[float, float, float, float], *, orig_w: int, orig_h: int
) -> dict[str, float]:
    x1, y1, x2, y2 = xyxy
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0
    return {
        "cx": cx / float(orig_w),
        "cy": cy / float(orig_h),
        "w": w / float(orig_w),
        "h": h / float(orig_h),
    }

