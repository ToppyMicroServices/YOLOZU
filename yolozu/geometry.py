"""Camera geometry helpers.

Intrinsics correction and 3-D translation recovery from 2-D detections
combined with depth estimates.
"""

__all__ = ["corrected_intrinsics", "recover_translation"]


def corrected_intrinsics(k, delta):
    """Apply multiplicative/additive intrinsics correction.

    Accepts k and delta as 4-element sequences (fx, fy, cx, cy).
    """
    if len(k) != 4 or len(delta) != 4:
        raise ValueError("k and delta must each have 4 elements (fx, fy, cx, cy)")
    fx, fy, cx, cy = (float(x) for x in k)
    dfx, dfy, dcx, dcy = (float(x) for x in delta)
    return (
        fx * (1.0 + dfx),
        fy * (1.0 + dfy),
        cx + dcx,
        cy + dcy,
    )


def recover_translation(bbox_center, offsets, z, k_prime, *, eps: float = 1e-9):
    """Recover 3-D translation from 2-D detection + depth.

    Protects against zero focal-length values via ``eps``.
    """
    u, v = bbox_center
    du, dv = offsets
    fx, fy, cx, cy = (float(x) for x in k_prime)
    if abs(fx) < eps:
        fx = eps
    if abs(fy) < eps:
        fy = eps
    u_prime = u + du
    v_prime = v + dv
    z = float(z)
    x = (u_prime - cx) / fx * z
    y = (v_prime - cy) / fy * z
    return (x, y, z)
