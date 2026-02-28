"""Symmetry-aware template verification.

Applies template scoring to the top-K detections in prediction
entries and writes the resulting ``score_tmp_sym`` field.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .symmetry import score_template_sym

__all__ = ["compute_score_tmp_sym", "apply_template_verification"]


def _matrix_from_det(det: dict[str, Any]) -> list[list[float]] | None:
    for key in ("R", "r_mat", "rot_mat"):
        value = det.get(key)
        if isinstance(value, list) and len(value) == 3 and all(isinstance(row, list) and len(row) == 3 for row in value):
            try:
                return [[float(x) for x in row] for row in value]
            except Exception:
                return None
    return None


def compute_score_tmp_sym(
    det: dict[str, Any],
    *,
    spec: dict[str, Any],
    score_fn: Callable[[dict[str, Any], list[list[float]]], float],
    sample_count: int = 8,
) -> float | None:
    r_pred = _matrix_from_det(det)
    if r_pred is None:
        return None

    def _score(rot):
        return float(score_fn(det, rot))

    return float(score_template_sym(_score, r_pred, spec, sample_count=sample_count))


def apply_template_verification(
    entries: Iterable[dict[str, Any]],
    *,
    symmetry_map: dict[str, Any],
    score_fn: Callable[[dict[str, Any], list[list[float]]], float],
    top_k: int = 10,
    sample_count: int = 8,
    score_key: str = "score",
    output_key: str = "score_tmp_sym",
) -> list[dict[str, Any]]:
    out_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []
        indexed = []
        for idx, det in enumerate(dets):
            if not isinstance(det, dict):
                continue
            score = det.get(score_key)
            score_val = float(score) if isinstance(score, (int, float)) else float("-inf")
            indexed.append((idx, score_val, det))

        indexed.sort(key=lambda item: item[1], reverse=True)
        top_set = {idx for idx, _, _ in indexed[: max(0, int(top_k))]}

        new_dets = []
        for idx, _, det in indexed:
            new_det = dict(det)
            if idx in top_set:
                class_key = new_det.get("class_id", new_det.get("class_name"))
                spec = symmetry_map.get(class_key)
                if spec is not None:
                    score_tmp = compute_score_tmp_sym(
                        new_det,
                        spec=spec,
                        score_fn=score_fn,
                        sample_count=sample_count,
                    )
                    if score_tmp is not None:
                        new_det[output_key] = float(score_tmp)
            new_dets.append(new_det)

        new_entry["detections"] = new_dets
        out_entries.append(new_entry)
    return out_entries
