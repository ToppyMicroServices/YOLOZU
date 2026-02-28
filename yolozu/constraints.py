"""Geometric constraints for 6-DoF pose post-processing.

Constraints include depth priors, table-plane checks, and upright / roll–pitch
range filtering.  ``apply_constraints`` is the single entry point used by
the evaluation and refinement pipelines.
"""

from __future__ import annotations

import math

__all__ = [
    "depth_prior",
    "depth_prior_penalty",
    "plane_signed_distance",
    "is_above_plane",
    "roll_pitch_yaw",
    "upright_violation_deg",
    "apply_constraints",
]


def depth_prior(
    bbox_wh: tuple[float, float],
    size_wh: tuple[float, float],
    intrinsics_fx_fy: tuple[float, float],
    eps: float = 1e-6,
) -> float:
    """Estimate object depth from bounding-box and physical size via pinhole model."""
    bbox_w, bbox_h = bbox_wh
    size_w, size_h = size_wh
    fx, fy = intrinsics_fx_fy
    z_w = fx * size_w / max(bbox_w, eps)
    z_h = fy * size_h / max(bbox_h, eps)
    return 0.5 * (z_w + z_h)


def depth_prior_penalty(z_pred: float, z_prior: float, eps: float = 1e-6) -> float:
    """Return absolute log-ratio between predicted and prior depth."""
    return abs(math.log(max(z_pred, eps)) - math.log(max(z_prior, eps)))


def plane_signed_distance(
    t_xyz: tuple[float, float, float],
    n: list[float],
    d: float,
) -> float:
    """Signed distance from point *t_xyz* to the plane defined by (n, d)."""
    x, y, z = t_xyz
    return n[0] * x + n[1] * y + n[2] * z + d


def is_above_plane(
    t_xyz: tuple[float, float, float],
    n: list[float],
    d: float,
    tol: float = 0.0,
) -> bool:
    """Return ``True`` if *t_xyz* is on or above the plane (within *tol*)."""
    return plane_signed_distance(t_xyz, n, d) >= -abs(tol)


def roll_pitch_yaw(
    r: list[list[float]],
) -> tuple[float, float, float]:
    """Extract (roll, pitch, yaw) Euler angles from a 3×3 rotation matrix."""
    pitch = math.asin(max(-1.0, min(1.0, -r[2][0])))
    roll = math.atan2(r[2][1], r[2][2])
    yaw = math.atan2(r[1][0], r[0][0])
    return roll, pitch, yaw


def upright_violation_deg(
    r: list[list[float]],
    roll_range_deg: tuple[float, float],
    pitch_range_deg: tuple[float, float],
) -> float:
    """Sum of roll and pitch limit violations in degrees."""
    roll, pitch, _yaw = roll_pitch_yaw(r)
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    roll_min, roll_max = roll_range_deg
    pitch_min, pitch_max = pitch_range_deg
    roll_violation = max(0.0, roll_min - roll_deg, roll_deg - roll_max)
    pitch_violation = max(0.0, pitch_min - pitch_deg, pitch_deg - pitch_max)
    return roll_violation + pitch_violation


def _lookup_per_class(section, class_key):
    per_class = section.get("per_class", {}) if isinstance(section, dict) else {}
    if class_key in per_class:
        return per_class[class_key]
    if isinstance(class_key, int):
        return per_class.get(str(class_key))
    if isinstance(class_key, str) and class_key.isdigit():
        return per_class.get(int(class_key))
    return None


def apply_constraints(
    cfg,
    class_key,
    bbox_wh,
    size_wh,
    intrinsics_fx_fy,
    t_xyz,
    r_mat,
    z_pred,
    *,
    eps: float = 1e-9,
):
    enabled = cfg.get("enabled", {}) if isinstance(cfg, dict) else {}
    result = {
        "depth_prior_penalty": 0.0,
        "depth_range_violation": 0.0,
        "plane_ok": True,
        "upright_violation": 0.0,
    }

    # Validate r_mat: must be a 3x3 nested list; supply identity as safe default.
    if r_mat is None:
        r_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    elif not (
        isinstance(r_mat, (list, tuple))
        and len(r_mat) == 3
        and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in r_mat)
    ):
        r_mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    if enabled.get("depth_prior", False):
        depth_cfg = cfg.get("depth_prior", {})
        override = _lookup_per_class(depth_cfg, class_key) or depth_cfg.get("default", {})
        z_prior = depth_prior(bbox_wh, size_wh, intrinsics_fx_fy)
        result["depth_prior_penalty"] = depth_prior_penalty(z_pred, z_prior)
        min_z = override.get("min_z")
        max_z = override.get("max_z")
        if min_z is not None and z_pred < min_z:
            result["depth_range_violation"] += min_z - z_pred
        if max_z is not None and z_pred > max_z:
            result["depth_range_violation"] += z_pred - max_z

    if enabled.get("table_plane", False):
        plane = cfg.get("table_plane", {})
        result["plane_ok"] = is_above_plane(
            t_xyz, plane.get("n", [0.0, 0.0, 1.0]), plane.get("d", 0.0)
        )

    if enabled.get("upright", False):
        upright_cfg = cfg.get("upright", {})
        override = _lookup_per_class(upright_cfg, class_key) or upright_cfg.get(
            "default", {}
        )
        roll_range = override.get("roll_deg", (-180.0, 180.0))
        pitch_range = override.get("pitch_deg", (-180.0, 180.0))
        result["upright_violation"] = upright_violation_deg(
            r_mat, roll_range, pitch_range
        )

    return result
