"""Keypoint normalisation and coordinate helpers.

Handles flat-list and dict keypoint formats, auto-detects
normalised-vs-pixel coordinates, and converts between them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["normalize_keypoints", "infer_keypoints_normalized", "keypoints_to_pixels"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_keypoints(value: Any, *, where: str = "keypoints") -> list[dict[str, Any]]:
    """Normalize keypoints into a list of dicts with at least {x,y}.

    Supported input shapes:
      - list[dict]: [{"x": .., "y": .., "v": ..?, "score": ..?}, ...]
      - flat list numbers:
          - [x1,y1,x2,y2,...] (len%2==0)
          - [x1,y1,v1,x2,y2,v2,...] (len%3==0)  # YOLO pose-style visibility

    Notes:
      - This function does not interpret coordinate units. Many tools assume [0,1] normalized coordinates.
      - Visibility `v` is kept as-is (usually 0/1/2 for COCO-style).
    """

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")

    if not value:
        return []

    if all(isinstance(it, dict) for it in value):
        out: list[dict[str, Any]] = []
        for idx, kp in enumerate(value):
            if not isinstance(kp, dict):  # pragma: no cover
                raise ValueError(f"{where}[{idx}] must be an object")
            if "x" not in kp or "y" not in kp:
                raise ValueError(f"{where}[{idx}] missing x/y")
            x = kp.get("x")
            y = kp.get("y")
            if not _is_number(x) or not _is_number(y):
                raise ValueError(f"{where}[{idx}].x/.y must be numbers")
            out.append(dict(kp))
        return out

    if not all(_is_number(it) for it in value):
        raise ValueError(f"{where} must be list[dict] or flat list of numbers")

    nums = [float(it) for it in value]
    if len(nums) % 3 == 0:
        out = []
        for i in range(0, len(nums), 3):
            out.append({"x": float(nums[i]), "y": float(nums[i + 1]), "v": nums[i + 2]})
        return out
    if len(nums) % 2 == 0:
        out = []
        for i in range(0, len(nums), 2):
            out.append({"x": float(nums[i]), "y": float(nums[i + 1])})
        return out

    raise ValueError(f"{where} flat list length must be divisible by 2 or 3")


def infer_keypoints_normalized(keypoints: list[dict[str, Any]]) -> bool | None:
    """Heuristic: return True if keypoints look like normalized coords, False if pixel-like, else None."""

    if not keypoints:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for kp in keypoints:
        if not isinstance(kp, dict):
            continue
        x = kp.get("x")
        y = kp.get("y")
        if _is_number(x) and _is_number(y):
            xs.append(float(x))
            ys.append(float(y))

    if not xs or not ys:
        return None

    # Pixel coords typically exceed 2.0; normalized coords usually stay in [-0.5,1.5].
    if any(abs(v) > 2.0 for v in xs + ys):
        return False
    if all(-0.5 <= v <= 1.5 for v in xs + ys):
        return True
    return None


def keypoints_to_pixels(
    keypoints: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    normalized: bool | None = None,
) -> list[tuple[float, float, Any]]:
    """Convert normalized keypoints to pixel coords.

    Returns: list of (x_px, y_px, v) where v is an optional visibility value.
    """

    if normalized is None:
        normalized = infer_keypoints_normalized(keypoints)

    out: list[tuple[float, float, Any]] = []
    for kp in keypoints:
        if not isinstance(kp, dict):
            continue
        x = kp.get("x")
        y = kp.get("y")
        if not _is_number(x) or not _is_number(y):
            continue
        v = kp.get("v")
        if normalized:
            out.append((float(x) * float(width), float(y) * float(height), v))
        else:
            out.append((float(x), float(y), v))
    return out

