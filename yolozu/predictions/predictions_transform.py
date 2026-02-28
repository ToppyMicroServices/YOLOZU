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
    "apply_ttt_lite",
    "summarize_task_coverage",
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


def _flip_norm_x(value: Any) -> float:
    return 1.0 - float(value)


def _flip_abs_x(value: Any, *, width: float) -> float:
    return float(width) - float(value)


def _flip_keypoint_x(
    value: Any,
    *,
    width: float | None,
    norm_only: bool,
    warnings: list[str],
    where: str,
) -> float | None:
    try:
        x = float(value)
    except Exception:
        warnings.append(f"{where}: invalid keypoint.x")
        return None

    is_norm = 0.0 <= x <= 1.0
    if is_norm:
        return _flip_norm_x(x)
    if norm_only:
        warnings.append(f"{where}: keypoint.x appears absolute but --tta-norm-only is enabled")
        return None
    if width is None:
        warnings.append(f"{where}: missing image_size for absolute keypoint.x")
        return None
    return _flip_abs_x(x, width=width)


def _flip_keypoints_inplace(
    det: dict[str, Any],
    *,
    width: float | None,
    norm_only: bool,
    swap_pairs: list[tuple[int, int]] | None,
    warnings: list[str],
    where: str,
) -> None:
    keypoints = det.get("keypoints")
    if keypoints is None:
        return
    if not isinstance(keypoints, list):
        warnings.append(f"{where}: keypoints must be a list when present")
        return

    out: list[Any] = []
    for kp_idx, kp in enumerate(keypoints):
        kp_where = f"{where}.keypoints[{kp_idx}]"
        if isinstance(kp, dict):
            new_kp = dict(kp)
            flipped = _flip_keypoint_x(
                new_kp.get("x"),
                width=width,
                norm_only=norm_only,
                warnings=warnings,
                where=kp_where,
            )
            if flipped is not None:
                new_kp["x"] = float(flipped)
            out.append(new_kp)
            continue

        if isinstance(kp, (list, tuple)) and len(kp) >= 2:
            new_kp = list(kp)
            flipped = _flip_keypoint_x(
                new_kp[0],
                width=width,
                norm_only=norm_only,
                warnings=warnings,
                where=kp_where,
            )
            if flipped is not None:
                new_kp[0] = float(flipped)
            out.append(new_kp)
            continue

        warnings.append(f"{kp_where}: unsupported keypoint format")
        out.append(kp)

    if swap_pairs:
        for pair_idx, pair in enumerate(swap_pairs):
            try:
                a = int(pair[0])
                b = int(pair[1])
            except Exception:
                warnings.append(f"{where}: invalid keypoint swap pair at index {pair_idx}")
                continue
            if a < 0 or b < 0 or a >= len(out) or b >= len(out):
                warnings.append(
                    f"{where}: keypoint swap pair ({a},{b}) is out of range for {len(out)} keypoints"
                )
                continue
            out[a], out[b] = out[b], out[a]

    det["keypoints"] = out


def _flip_pose_offsets_inplace(det: dict[str, Any], *, warnings: list[str], where: str) -> None:
    offsets = det.get("offsets")
    if offsets is None:
        return
    if not isinstance(offsets, (list, tuple)) or len(offsets) < 1:
        warnings.append(f"{where}: offsets must be a list[>=1] when present")
        return
    out = list(offsets)
    try:
        out[0] = -float(out[0])
    except Exception:
        warnings.append(f"{where}: invalid offsets[0]")
        return
    det["offsets"] = out


def apply_tta(
    entries: Iterable[dict[str, Any]],
    *,
    enabled: bool = True,
    seed: int | None = None,
    flip_prob: float = 0.5,
    norm_only: bool = False,
    flip_keypoints: bool = True,
    flip_pose_offsets: bool = True,
    keypoint_swap_pairs: Iterable[tuple[int, int]] | None = None,
) -> TransformResult:
    """Apply a simple test-time augmentation transform to predictions.

    When enabled, a per-detection mask is sampled (seeded) to decide whether
    to apply a horizontal flip in normalized space (cx -> 1 - cx). If
    norm_only is False and bbox_abs/keypoints absolute coordinates are present,
    a best-effort absolute flip is applied using entry.image_size.

    Optional multi-task fields:
    - keypoints: flips x for list-of-dict or list-of-list forms.
    - offsets: flips x offset sign (pose-style center offsets).
    """

    warnings: list[str] = []
    out_entries: list[dict[str, Any]] = []
    rng = random.Random(seed)
    swap_pairs = (
        [(int(a), int(b)) for a, b in list(keypoint_swap_pairs)]
        if keypoint_swap_pairs is not None
        else None
    )

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        new_dets = []
        mask: list[bool] = []
        width, _ = _entry_image_size(entry)

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
                        if width is None:
                            warnings.append(f"predictions[{idx}].detections[{j}]: missing image_size for bbox_abs")
                        else:
                            new_bbox_abs = dict(bbox_abs)
                            try:
                                new_bbox_abs["cx"] = float(width) - float(new_bbox_abs["cx"])
                            except Exception:
                                warnings.append(f"predictions[{idx}].detections[{j}]: invalid bbox_abs.cx")
                            new_det["bbox_abs"] = new_bbox_abs

                where = f"predictions[{idx}].detections[{j}]"
                if bool(flip_keypoints):
                    _flip_keypoints_inplace(
                        new_det,
                        width=width,
                        norm_only=bool(norm_only),
                        swap_pairs=swap_pairs,
                        warnings=warnings,
                        where=where,
                    )
                if bool(flip_pose_offsets):
                    _flip_pose_offsets_inplace(
                        new_det,
                        warnings=warnings,
                        where=where,
                    )

            new_dets.append(new_det)

        new_entry["detections"] = new_dets
        if enabled:
            new_entry["tta_mask"] = mask
        out_entries.append(new_entry)

    return TransformResult(entries=out_entries, warnings=warnings)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_entropy_like(det: dict[str, Any], *, warnings: list[str], where: str) -> float | None:
    for key in ("entropy", "class_entropy", "normalized_entropy"):
        if key in det:
            try:
                return _clamp01(float(det.get(key)))
            except Exception:
                warnings.append(f"{where}: invalid {key}")
                return None

    probs = det.get("class_probs")
    if isinstance(probs, (list, tuple)) and probs:
        vals: list[float] = []
        for idx, value in enumerate(probs):
            try:
                vals.append(float(value))
            except Exception:
                warnings.append(f"{where}.class_probs[{idx}]: invalid value")
                return None
        total = sum(vals)
        if total <= 0.0:
            return None
        p = [max(0.0, v) / total for v in vals]
        import math

        entropy = 0.0
        for v in p:
            if v > 0.0:
                entropy -= v * math.log(v)
        norm = math.log(max(2, len(p)))
        if norm <= 0.0:
            return None
        return _clamp01(entropy / norm)
    return None


def apply_ttt_lite(
    entries: Iterable[dict[str, Any]],
    *,
    enabled: bool = True,
    temperature: float = 1.0,
    entropy_weight: float = 0.0,
    minmax_norm: bool = True,
    preserve_raw_score_key: str | None = "score_raw",
) -> TransformResult:
    """Apply lightweight non-torch TTT-style score adaptation.

    This path adjusts detection scores only (no weight updates) and is meant
    for backends that cannot run gradient-based TTT.
    """

    warnings: list[str] = []
    out_entries: list[dict[str, Any]] = []
    temp = max(1e-6, float(temperature))
    ent_w = max(0.0, float(entropy_weight))

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        dets = new_entry.get("detections") or []
        if not isinstance(dets, list):
            dets = []

        score_vals: list[float] = []
        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                continue
            try:
                score_vals.append(float(det.get("score", 0.0)))
            except Exception:
                warnings.append(f"predictions[{idx}].detections[{j}]: invalid score")
                score_vals.append(0.0)
        s_min = min(score_vals) if score_vals else 0.0
        s_max = max(score_vals) if score_vals else 1.0
        s_span = s_max - s_min

        new_dets: list[dict[str, Any]] = []
        for j, det in enumerate(dets):
            if not isinstance(det, dict):
                continue
            where = f"predictions[{idx}].detections[{j}]"
            try:
                score_raw = float(det.get("score", 0.0))
            except Exception:
                warnings.append(f"{where}: invalid score")
                score_raw = 0.0

            score = float(score_raw)
            if bool(enabled):
                if bool(minmax_norm) and s_span > 1e-12:
                    score = (score - s_min) / s_span
                score = _clamp01(score)
                entropy = _extract_entropy_like(det, warnings=warnings, where=where)
                if entropy is not None and ent_w > 0.0:
                    score = score * max(0.0, 1.0 - ent_w * float(entropy))
                if abs(temp - 1.0) > 1e-8:
                    score = pow(max(score, 0.0), 1.0 / temp)
                score = _clamp01(score)

            new_det = dict(det)
            if preserve_raw_score_key and preserve_raw_score_key not in new_det:
                new_det[preserve_raw_score_key] = float(score_raw)
            new_det["score"] = float(score)
            new_dets.append(new_det)

        new_entry["detections"] = new_dets
        out_entries.append(new_entry)

    return TransformResult(entries=out_entries, warnings=warnings)


def summarize_task_coverage(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Infer task coverage from predictions payload content."""

    counts = {
        "bbox": 0,
        "segmentation": 0,
        "keypoints": 0,
        "depth": 0,
        "pose6d": 0,
    }
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dets = entry.get("detections") or []
        if not isinstance(dets, list):
            continue
        for det in dets:
            if not isinstance(det, dict):
                continue
            if isinstance(det.get("bbox"), dict) and all(k in det["bbox"] for k in ("cx", "cy", "w", "h")):
                counts["bbox"] += 1
            if any(k in det for k in ("mask", "mask_path", "segmentation", "rle", "polygon")):
                counts["segmentation"] += 1
            if isinstance(det.get("keypoints"), list) and det.get("keypoints"):
                counts["keypoints"] += 1
            if any(k in det for k in ("depth", "depth_path", "log_z", "z", "sigma_z", "D_obj")):
                counts["depth"] += 1
            if any(k in det for k in ("pose", "rot6d", "R", "R_gt", "t", "t_gt", "offsets", "k_delta")):
                counts["pose6d"] += 1

    supported = {k: bool(v > 0) for k, v in counts.items()}
    tasks = [k for k, is_on in supported.items() if is_on]
    return {
        "counts": counts,
        "supported": supported,
        "tasks": tasks,
    }

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

        new_entry = dict(entry)
        new_entry["detections"] = new_dets
        out_entries.append(new_entry)

    return TransformResult(entries=out_entries, warnings=warnings)
