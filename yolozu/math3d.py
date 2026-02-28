"""Pure-Python 3×3 matrix math for rotations.

Axis-angle transforms, matrix multiply, transpose, trace, and
geodesic distance on SO(3).
"""

import math

__all__ = [
    "clamp",
    "mat_identity",
    "mat_mul",
    "mat_t",
    "mat_trace",
    "normalize_axis",
    "rotation_matrix_axis_angle",
    "rotation_z",
    "geodesic_distance",
]


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def mat_identity():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def mat_mul(a, b):
    return [
        [
            a[0][0] * b[0][j] + a[0][1] * b[1][j] + a[0][2] * b[2][j]
            for j in range(3)
        ],
        [
            a[1][0] * b[0][j] + a[1][1] * b[1][j] + a[1][2] * b[2][j]
            for j in range(3)
        ],
        [
            a[2][0] * b[0][j] + a[2][1] * b[1][j] + a[2][2] * b[2][j]
            for j in range(3)
        ],
    ]


def mat_t(m):
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]


def mat_trace(m):
    return m[0][0] + m[1][1] + m[2][2]


def normalize_axis(axis):
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    return [x / norm, y / norm, z / norm]


def rotation_matrix_axis_angle(axis, angle_rad):
    x, y, z = normalize_axis(axis)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return [
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ]


def rotation_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ]


def geodesic_distance(r1, r2):
    rel = mat_mul(mat_t(r1), r2)
    cos_theta = (mat_trace(rel) - 1.0) * 0.5
    cos_theta = clamp(cos_theta, -1.0, 1.0)
    return math.acos(cos_theta)
