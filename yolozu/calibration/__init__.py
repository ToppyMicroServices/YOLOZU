from __future__ import annotations

import math
from typing import Any

from .lbfgs_scale_k import CalibConfig, CalibResult, calibrate_predictions_lbfgs
from .hessian_solver import (
    HessianSolverConfig,
    RefinementResult,
    refine_detection_hessian,
    refine_predictions_hessian,
)

# ---------------------------------------------------------------------------
# Lightweight temperature-scaling helpers (formerly in standalone calibration.py)
# ---------------------------------------------------------------------------


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _logit(p: float) -> float:
    p = _clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_temperature(score: float, temperature: float) -> float:
    """Apply temperature scaling to a confidence score."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return float(_sigmoid(_logit(float(score)) / float(temperature)))


def calibrate_predictions_entries(
    entries: list[dict[str, Any]],
    *,
    temperature: float,
    min_score: float | None = None,
    max_score: float | None = None,
) -> list[dict[str, Any]]:
    """Calibrate detection scores across prediction entries.

    Preserves all original entry keys (image_size, preprocess, etc.).
    """
    out: list[dict[str, Any]] = []
    for entry in entries:
        dets = []
        for det in entry.get("detections", []) or []:
            score = apply_temperature(float(det.get("score", 0.0)), temperature)
            if min_score is not None:
                score = max(float(min_score), score)
            if max_score is not None:
                score = min(float(max_score), score)
            det_out = dict(det)
            det_out["score"] = score
            dets.append(det_out)
        # Preserve all original entry keys (image_size, preprocess, etc.).
        entry_out = dict(entry)
        entry_out["detections"] = dets
        out.append(entry_out)
    return out


__all__ = [
    "CalibConfig",
    "CalibResult",
    "calibrate_predictions_lbfgs",
    "HessianSolverConfig",
    "RefinementResult",
    "refine_detection_hessian",
    "refine_predictions_hessian",
    "apply_temperature",
    "calibrate_predictions_entries",
]
