"""Camera intrinsics parsing.

Converts various intrinsic representations (3×3 matrix, dict,
OpenCV FileStorage XML) into a canonical ``{fx, fy, cx, cy}`` dict.
"""

from __future__ import annotations

from typing import Any

__all__ = ["parse_intrinsics"]


def _maybe_to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (AttributeError, TypeError, ValueError):
            return value
    return value


def _parse_k_3x3_from_opencv_matrix_dict(value: dict[str, Any]) -> list[list[float]] | None:
    # OpenCV FileStorage / YAML style: {rows:3, cols:3, dt:..., data:[...]}
    try:
        rows = int(value.get("rows"))
        cols = int(value.get("cols"))
    except (TypeError, ValueError):
        return None
    if rows != 3 or cols != 3:
        return None
    data = value.get("data")
    if not isinstance(data, (list, tuple)) or len(data) < 9:
        return None
    try:
        flat = [float(data[i]) for i in range(9)]
    except (TypeError, ValueError, IndexError):
        return None
    return [
        [flat[0], flat[1], flat[2]],
        [flat[3], flat[4], flat[5]],
        [flat[6], flat[7], flat[8]],
    ]


def parse_intrinsics(value: Any) -> dict[str, float] | None:
    """Parse camera intrinsics into a canonical {fx, fy, cx, cy} dict.

    Supported forms (OpenCV-friendly):
      - {"fx":..,"fy":..,"cx":..,"cy":..}
      - [fx, fy, cx, cy]
      - 3x3 K matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
      - flat row-major 9 values [k00,k01,k02,k10,...,k22]
      - OpenCV FileStorage matrix dict {rows:3, cols:3, data:[...]} (optionally nested under camera_matrix)
      - Nested dicts containing keys like camera_matrix/K/intrinsics.
    """

    value = _maybe_to_list(value)
    if value is None:
        return None

    if isinstance(value, dict):
        if all(k in value for k in ("fx", "fy", "cx", "cy")):
            try:
                return {
                    "fx": float(value["fx"]),
                    "fy": float(value["fy"]),
                    "cx": float(value["cx"]),
                    "cy": float(value["cy"]),
                }
            except (TypeError, ValueError, KeyError):
                return None

        k_3x3 = _parse_k_3x3_from_opencv_matrix_dict(value)
        if k_3x3 is not None:
            value = k_3x3
        else:
            for key in ("camera_matrix", "cameraMatrix", "K", "K_gt", "intrinsics"):
                if key in value:
                    parsed = parse_intrinsics(value.get(key))
                    if parsed is not None:
                        return parsed
            return None

    if isinstance(value, (list, tuple)):
        # (fx, fy, cx, cy)
        if len(value) == 4 and not isinstance(value[0], (list, tuple, dict)):
            try:
                fx, fy, cx, cy = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
            except (TypeError, ValueError):
                return None
            return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

        # Flat row-major 3x3.
        if len(value) == 9 and not isinstance(value[0], (list, tuple, dict)):
            try:
                flat = [float(v) for v in value]
            except (TypeError, ValueError):
                return None
            return {"fx": flat[0], "fy": flat[4], "cx": flat[2], "cy": flat[5]}

        # Nested 3x3.
        if len(value) == 3 and isinstance(value[0], (list, tuple)):
            try:
                fx = float(value[0][0])
                fy = float(value[1][1])
                cx = float(value[0][2])
                cy = float(value[1][2])
                return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
            except (TypeError, ValueError, IndexError):
                return None

    return None
