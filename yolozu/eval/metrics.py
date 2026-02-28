"""6-DoF pose-estimation metrics.

Symmetry-aware geodesic rotation distance and ADD-S (average closest
point distance under symmetry).
"""

import math

from yolozu.geometry.math3d import mat_mul, rotation_matrix_axis_angle
from yolozu.geometry.symmetry import enumerate_symmetry_rotations, min_symmetry_geodesic, normalize_symmetry

__all__ = ["symmetry_geodesic", "add_s"]


def symmetry_geodesic(r_pred, r_gt, sym_spec, sample_count=8):
    return min_symmetry_geodesic(r_pred, r_gt, sym_spec, sample_count=sample_count)


def _mat_vec_mul(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _transform_point(r, t, p):
    x, y, z = _mat_vec_mul(r, p)
    return (x + t[0], y + t[1], z + t[2])


def _mean_distance(points, r_pred, t_pred, r_gt, t_gt, s):
    total = 0.0
    for p in points:
        p_pred = _transform_point(r_pred, t_pred, p)
        p_gt = _transform_point(mat_mul(r_gt, s), t_gt, p)
        dx = p_pred[0] - p_gt[0]
        dy = p_pred[1] - p_gt[1]
        dz = p_pred[2] - p_gt[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total / max(1, len(points))


def add_s(points, r_pred, t_pred, r_gt, t_gt, sym_spec, sample_count=8):
    spec = normalize_symmetry(sym_spec)
    sym_type = spec["type"]
    best = None
    if sym_type == "Cinf":
        axis = spec["axis"]
        for k in range(sample_count):
            angle = 2.0 * math.pi * k / sample_count
            s = rotation_matrix_axis_angle(axis, angle)
            best = _update_best(points, r_pred, t_pred, r_gt, t_gt, s, best)
        return best if best is not None else 0.0
    for s in enumerate_symmetry_rotations(spec):
        best = _update_best(points, r_pred, t_pred, r_gt, t_gt, s, best)
    return best if best is not None else 0.0


def _update_best(points, r_pred, t_pred, r_gt, t_gt, s, best):
    value = _mean_distance(points, r_pred, t_pred, r_gt, t_gt, s)
    if best is None or value < best:
        return value
    return best
