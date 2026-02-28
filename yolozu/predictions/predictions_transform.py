"""Prediction transforms: class-ID normalisation, TTA, score fusion.

Maps COCO category IDs to contiguous class IDs, applies test-time
augmentation, and fuses detection scores via configurable gate
weights.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "TransformResult",
    "load_classes_json",
    "build_category_id_to_class_id_map",
    "normalize_class_ids",
    "apply_tta",
    "fuse_detection_scores",
]

from yolozu.training.gates import final_score, passes_template_gate


@dataclass(frozen=True)
class TransformResult:
    entries: list[dict[str, Any]]
    warnings: list[str]

def load_classes_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _to_int_key_map(mapping: dict[Any, Any]) -> dict[int, int]:
    out: dict[int, int] = {}
    for k, v in mapping.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def build_category_id_to_class_id_map(classes_json: dict[str, Any]) -> dict[int, int]:
    raw = classes_json.get("category_id_to_class_id")
    if not isinstance(raw, dict):
        raise ValueError("classes.json missing category_id_to_class_id")
    return _to_int_key_map(raw)


def normalize_class_ids(
    entries: Iterable[dict[str, Any]],
    *,
    classes_json: dict[str, Any] | None = None,
    assume_class_id_is_category_id: bool = False,
) -> TransformResult:
    """Normalize detections to use contiguous `class_id` (0..N-1).

    Supported normalization:
    - If a detection has `category_id` and is missing `class_id`, map it.
    - If assume_class_id_is_category_id=True, treat `class_id` as a COCO category id and map it.
    """

    warnings: list[str] = []
    cat_to_cls: dict[int, int] | None = None
    if classes_json is not None:
        cat_to_cls = build_category_id_to_class_id_map(classes_json)

    out_entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        new_dets = []
        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                continue
            new_det = dict(det)

            if assume_class_id_is_category_id and "class_id" in new_det and cat_to_cls is not None:
                try:
                    cat_id = int(new_det["class_id"])
                    if cat_id in cat_to_cls:
                        new_det["class_id"] = int(cat_to_cls[cat_id])
                    else:
                        warnings.append(f"predictions[{idx}].detections[{j}]: unknown category_id {cat_id}")
                except Exception:
                    warnings.append(f"predictions[{idx}].detections[{j}]: invalid class_id")

            if "class_id" not in new_det and "category_id" in new_det and cat_to_cls is not None:
                try:
                    cat_id = int(new_det["category_id"])
                    if cat_id in cat_to_cls:
                        new_det["class_id"] = int(cat_to_cls[cat_id])
                    else:
                        warnings.append(f"predictions[{idx}].detections[{j}]: unknown category_id {cat_id}")
                except Exception:
                    warnings.append(f"predictions[{idx}].detections[{j}]: invalid category_id")

            new_dets.append(new_det)

        new_entry["detections"] = new_dets
        out_entries.append(new_entry)

    return TransformResult(entries=out_entries, warnings=warnings)


def _entry_image_size(entry: dict[str, Any]) -> tuple[float | None, float | None]:
    value = entry.get("image_size")
    if isinstance(value, dict):
        width = value.get("width")
        height = value.get("height")
        try:
            return (float(width), float(height))
        except Exception:
            return (None, None)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return (None, None)
    return (None, None)


def apply_tta(
    entries: Iterable[dict[str, Any]],
    *,
    enabled: bool = True,
    seed: int | None = None,
    flip_prob: float = 0.5,
    norm_only: bool = False,
) -> TransformResult:
    """Apply a simple test-time augmentation transform to predictions.

    When enabled, a per-detection mask is sampled (seeded) to decide whether
    to apply a horizontal flip in normalized space (cx -> 1 - cx). If
    norm_only is False and bbox_abs is present, a best-effort absolute flip is
    applied using entry.image_size.
    """

    warnings: list[str] = []
    out_entries: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        new_dets = []
        mask: list[bool] = []

        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                continue
            new_det = dict(det)
            apply = bool(enabled) and (rng.random() < float(flip_prob))
            mask.append(apply)

            if apply:
                bbox = new_det.get("bbox")
                if isinstance(bbox, dict) and all(k in bbox for k in ("cx", "cy", "w", "h")):
                    new_bbox = dict(bbox)
                    try:
                        new_bbox["cx"] = 1.0 - float(new_bbox["cx"])
                    except Exception:
                        warnings.append(f"predictions[{idx}].detections[{j}]: invalid bbox.cx")
                    new_det["bbox"] = new_bbox
                else:
                    warnings.append(f"predictions[{idx}].detections[{j}]: missing bbox for tta")

                if not norm_only:
                    bbox_abs = new_det.get("bbox_abs")
                    if isinstance(bbox_abs, dict) and "cx" in bbox_abs:
                        width, _ = _entry_image_size(entry)
                        if width is None:
                            warnings.append(f"predictions[{idx}].detections[{j}]: missing image_size for bbox_abs")
                        else:
                            new_bbox_abs = dict(bbox_abs)
                            try:
                                new_bbox_abs["cx"] = float(width) - float(new_bbox_abs["cx"])
                            except Exception:
                                warnings.append(f"predictions[{idx}].detections[{j}]: invalid bbox_abs.cx")
                            new_det["bbox_abs"] = new_bbox_abs

            new_dets.append(new_det)

        new_entry["detections"] = new_dets
        if enabled:
            new_entry["tta_mask"] = mask
        out_entries.append(new_entry)

    return TransformResult(entries=out_entries, warnings=warnings)

def fuse_detection_scores(
    entries: Iterable[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
    det_score_key: str = "score",
    template_score_key: str = "score_tmp_sym",
    sigma_z_key: str = "sigma_z",
    sigma_rot_key: str = "sigma_rot",
    out_score_key: str = "score",
    preserve_det_score_key: str | None = "score_det",
    template_gate_enabled: bool = False,
    template_gate_tau: float = 0.0,
    min_score: float | None = None,
    topk_per_image: int | None = None,
) -> TransformResult:
    """Fuse detection scores for inference-time gating/tuning.

    The fused score is:
      w_det * score_det + w_tmp * score_tmp_sym - w_unc * (sigma_z + sigma_rot)

    This is intended for *postprocess-time* score shaping (ordering/thresholding),
    and can be tuned offline on a fixed eval subset.

    Notes:
    - Missing template/uncertainty fields default to 0.0.
    - If template_gate_enabled=True, detections with score_tmp_sym < template_gate_tau are dropped.
    - If preserve_det_score_key is set and missing, the original det score is stored there.
    """

    warnings: list[str] = []
    w = {"det": 1.0, "tmp": 1.0, "unc": 1.0}
    if weights:
        for k, v in weights.items():
            try:
                w[str(k)] = float(v)
            except Exception:
                continue

    out_entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not image:
            warnings.append(f"predictions[{idx}]: missing image")
            continue

        dets = entry.get("detections") or []
        if not isinstance(dets, list):
            warnings.append(f"predictions[{idx}]: detections must be a list")
            dets = []

        new_dets: list[dict[str, Any]] = []
        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                continue

            try:
                score_det = float(det.get(det_score_key, 0.0))
            except Exception:
                warnings.append(f"predictions[{idx}].detections[{j}]: invalid {det_score_key}")
                score_det = 0.0

            try:
                score_tmp = float(det.get(template_score_key, 0.0))
            except Exception:
                warnings.append(f"predictions[{idx}].detections[{j}]: invalid {template_score_key}")
                score_tmp = 0.0

            try:
                sigma_z = float(det.get(sigma_z_key, 0.0))
            except Exception:
                warnings.append(f"predictions[{idx}].detections[{j}]: invalid {sigma_z_key}")
                sigma_z = 0.0

            try:
                sigma_rot = float(det.get(sigma_rot_key, 0.0))
            except Exception:
                warnings.append(f"predictions[{idx}].detections[{j}]: invalid {sigma_rot_key}")
                sigma_rot = 0.0

            if template_gate_enabled and not passes_template_gate(score_tmp, enabled=True, tau=float(template_gate_tau)):
                continue

            fused = final_score(score_det, score_tmp, sigma_z, sigma_rot, w)
            if min_score is not None and fused < float(min_score):
                continue

            new_det = dict(det)
            if preserve_det_score_key and preserve_det_score_key not in new_det:
                new_det[preserve_det_score_key] = float(score_det)
            new_det[out_score_key] = float(fused)
            new_dets.append(new_det)

        if topk_per_image is not None and topk_per_image > 0 and len(new_dets) > topk_per_image:
            new_dets.sort(key=lambda d: float(d.get(out_score_key, 0.0)), reverse=True)
            new_dets = new_dets[: int(topk_per_image)]

        out_entries.append({"image": image, "detections": new_dets})

    return TransformResult(entries=out_entries, warnings=warnings)
