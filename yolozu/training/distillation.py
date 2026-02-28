"""Score-level knowledge distillation for detections.

``distill_predictions`` blends teacher scores into student predictions
and optionally injects unmatched teacher detections, following a simple
label-free distillation recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yolozu.core.image_keys import add_image_aliases, lookup_image_alias
from yolozu.eval.simple_map import _bbox_iou_cxcywh_norm

__all__ = ["DistillStats", "distill_predictions"]


@dataclass(frozen=True)
class DistillStats:
    matched: int
    added: int
    avg_score_gap: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_near_duplicate(
    existing: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    iou_threshold: float,
) -> bool:
    candidate_class = int(candidate.get("class_id", -1))
    candidate_bbox = candidate.get("bbox", {})
    for det in existing:
        if int(det.get("class_id", -2)) != candidate_class:
            continue
        iou = _bbox_iou_cxcywh_norm(det.get("bbox", {}), candidate_bbox)
        if iou >= iou_threshold:
            return True
    return False


def _index_by_image(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        image = str(entry.get("image", ""))
        if not image:
            continue
        dets = entry.get("detections", []) or []
        add_image_aliases(index, image, list(dets) if isinstance(dets, list) else [])
    return index


def distill_predictions(
    student_entries: list[dict[str, Any]],
    teacher_entries: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.7,
    alpha: float = 0.5,
    add_missing: bool = True,
    add_score_scale: float = 0.5,
    teacher_min_score: float = 0.0,
    max_added_per_image: int | None = None,
    add_duplicate_iou_threshold: float = 0.9,
) -> tuple[list[dict[str, Any]], DistillStats]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0,1]")
    if add_score_scale < 0.0:
        raise ValueError("add_score_scale must be >= 0")
    if not 0.0 <= teacher_min_score <= 1.0:
        raise ValueError("teacher_min_score must be in [0,1]")
    if max_added_per_image is not None and max_added_per_image < 0:
        raise ValueError("max_added_per_image must be >= 0 or None")
    if not 0.0 <= add_duplicate_iou_threshold <= 1.0:
        raise ValueError("add_duplicate_iou_threshold must be in [0,1]")

    teacher_index = _index_by_image(teacher_entries)

    out_entries: list[dict[str, Any]] = []
    matched = 0
    added = 0
    total_gap = 0.0

    for entry in student_entries:
        image = str(entry.get("image", ""))
        if not image:
            continue
        student_dets = [dict(d) for d in (entry.get("detections", []) or [])]
        teacher_dets = lookup_image_alias(teacher_index, image) or []
        teacher_used = [False] * len(teacher_dets)

        for det in student_dets:
            best_iou = 0.0
            best_idx = -1
            for idx, tdet in enumerate(teacher_dets):
                if int(tdet.get("class_id", -1)) != int(det.get("class_id", -2)):
                    continue
                iou = _bbox_iou_cxcywh_norm(det.get("bbox", {}), tdet.get("bbox", {}))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                teacher_used[best_idx] = True
                t_score = float(teacher_dets[best_idx].get("score", 0.0))
                s_score = float(det.get("score", 0.0))
                total_gap += abs(t_score - s_score)
                matched += 1
                det["score"] = _clamp01(max(s_score, alpha * t_score + (1.0 - alpha) * s_score))

        if add_missing:
            added_in_image = 0
            teacher_order = sorted(
                range(len(teacher_dets)),
                key=lambda idx: float(teacher_dets[idx].get("score", 0.0)),
                reverse=True,
            )
            for idx in teacher_order:
                tdet = teacher_dets[idx]
                if teacher_used[idx]:
                    continue
                if float(tdet.get("score", 0.0)) < teacher_min_score:
                    continue
                if max_added_per_image is not None and added_in_image >= max_added_per_image:
                    break
                if _is_near_duplicate(
                    student_dets,
                    tdet,
                    iou_threshold=add_duplicate_iou_threshold,
                ):
                    continue
                score = _clamp01(float(tdet.get("score", 0.0)) * float(add_score_scale))
                det_out = dict(tdet)
                det_out["score"] = score
                student_dets.append(det_out)
                added += 1
                added_in_image += 1

        # Preserve all original entry keys (image_size, preprocess, etc.).
        entry_out = dict(entry)
        entry_out["detections"] = student_dets
        out_entries.append(entry_out)

    avg_gap = total_gap / float(max(1, matched))
    return out_entries, DistillStats(matched=matched, added=added, avg_score_gap=float(avg_gap))
