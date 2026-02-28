"""Discrete and continuous rotation-symmetry groups.

Handles C2, C4, Cn, and Cinf symmetry for geodesic distance
computation and template scoring.
"""

import math

from .math3d import geodesic_distance, mat_identity, mat_mul, rotation_matrix_axis_angle

__all__ = [
    "normalize_symmetry",
    "enumerate_symmetry_rotations",
    "min_symmetry_geodesic",
    "score_template_sym",
]


_SUPPORTED = {"none", "C2", "C4", "Cn", "Cinf"}


def normalize_symmetry(spec):
    if spec is None:
        return {"type": "none"}
    sym_type = spec.get("type", "none")
    if sym_type not in _SUPPORTED:
        raise ValueError(f"unsupported symmetry type: {sym_type}")
    if sym_type in ("C2", "C4"):
        spec = dict(spec)
        spec["n"] = 2 if sym_type == "C2" else 4
        spec["type"] = "Cn"
    if sym_type == "Cn":
        n = spec.get("n")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("Cn requires positive integer n")
    axis = spec.get("axis", [0.0, 0.0, 1.0])
    if len(axis) != 3:
        raise ValueError("axis must be length 3")
    spec = dict(spec)
    spec.setdefault("axis", axis)
    return spec


def enumerate_symmetry_rotations(spec):
    spec = normalize_symmetry(spec)
    sym_type = spec["type"]
    if sym_type == "none":
        return [mat_identity()]
    if sym_type == "Cinf":
        raise ValueError("Cinf is continuous; use sampling instead")
    n = spec["n"]
    axis = spec["axis"]
    return [
        rotation_matrix_axis_angle(axis, 2.0 * math.pi * k / n)
        for k in range(n)
    ]


def min_symmetry_geodesic(r_pred, r_gt, spec, sample_count=8):
    spec = normalize_symmetry(spec)
    sym_type = spec["type"]
    if sym_type == "Cinf":
        axis = spec["axis"]
        best = None
        for k in range(sample_count):
            angle = 2.0 * math.pi * k / sample_count
            s = rotation_matrix_axis_angle(axis, angle)
            candidate = geodesic_distance(r_pred, mat_mul(r_gt, s))
            if best is None or candidate < best:
                best = candidate
        return best
    best = None
    for s in enumerate_symmetry_rotations(spec):
        candidate = geodesic_distance(r_pred, mat_mul(r_gt, s))
        if best is None or candidate < best:
            best = candidate
    return best


def score_template_sym(score_fn, r_pred, spec, sample_count=8):
    spec = normalize_symmetry(spec)
    sym_type = spec["type"]
    best = None
    if sym_type == "Cinf":
        axis = spec["axis"]
        for k in range(sample_count):
            angle = 2.0 * math.pi * k / sample_count
            s = rotation_matrix_axis_angle(axis, angle)
            candidate = score_fn(mat_mul(r_pred, s))
            if best is None or candidate > best:
                best = candidate
        return best
    for s in enumerate_symmetry_rotations(spec):
        candidate = score_fn(mat_mul(r_pred, s))
        if best is None or candidate > best:
            best = candidate
    return best
