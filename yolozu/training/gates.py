"""Scoring gates for filtering detections.

Combines detection, template-symmetry, and uncertainty scores into a
final gate score, optionally rejecting candidates that fall below
configurable thresholds.
"""

__all__ = ["final_score", "passes_template_gate", "passes_low_fp_gate"]


def final_score(score_det, score_tmp_sym, sigma_z, sigma_rot, weights):
    """Compute final detection score from components.

    ``weights`` is a dict with optional keys ``det``, ``tmp``, ``unc``.
    Missing keys default to 1.0.
    """
    if not isinstance(weights, dict):
        weights = {}
    w_det = float(weights.get("det", 1.0))
    w_tmp = float(weights.get("tmp", 1.0))
    w_unc = float(weights.get("unc", 1.0))
    return w_det * float(score_det) + w_tmp * float(score_tmp_sym) - w_unc * (float(sigma_z) + float(sigma_rot))


def passes_template_gate(score_tmp_sym, enabled, tau):
    if not enabled:
        return True
    return float(score_tmp_sym) >= float(tau)


def passes_low_fp_gate(score_tmp_sym, enabled, tau):
    if not enabled:
        return True
    return float(score_tmp_sym) >= float(tau)
