"""High-level inference utilities.

Wraps constraint inference and template verification into a single
``apply_inference_utilities`` call that mutates prediction entries
in-place.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .inference import infer_constraints

__all__ = ["apply_inference_utilities"]
from yolozu.geometry.template_verification import apply_template_verification


def apply_inference_utilities(
    entries: Iterable[dict[str, Any]],
    *,
    constraints_cfg: dict[str, Any] | None = None,
    bbox_format: str = "cxcywh_norm",
    default_size_wh: tuple[float, float] = (1.0, 1.0),
    symmetry_map: dict[str, Any] | None = None,
    template_score_fn: Callable[[dict[str, Any], list[list[float]]], float] | None = None,
    template_top_k: int = 10,
    template_sample_count: int = 8,
) -> list[dict[str, Any]]:
    out_entries = list(entries)
    if constraints_cfg is not None:
        out_entries = infer_constraints(
            out_entries,
            constraints_cfg=constraints_cfg,
            bbox_format=bbox_format,
            default_size_wh=default_size_wh,
        )
    if symmetry_map is not None and template_score_fn is not None:
        out_entries = apply_template_verification(
            out_entries,
            symmetry_map=symmetry_map,
            score_fn=template_score_fn,
            top_k=template_top_k,
            sample_count=template_sample_count,
        )
    return out_entries
