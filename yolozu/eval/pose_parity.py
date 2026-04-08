from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yolozu.boxes import cxcywh_norm_to_xyxy_abs, iou_xyxy_abs
from yolozu.core.image_keys import add_image_aliases
from yolozu.core.image_size import get_image_size
from yolozu.geometry.math3d import geodesic_distance
from yolozu.predictions import load_predictions_entries

__all__ = ["compare_pose_predictions"]


@dataclass(frozen=True)
class _Detection:
    class_id: int
    score: float
    bbox: dict[str, Any]
    raw: dict[str, Any]


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        try:
            return [float(v) for v in value]
        except Exception:
            return None
    if hasattr(value, "tolist"):
        try:
            return _as_float_list(value.tolist())
        except Exception:
            return None
    return None


def _as_matrix_3x3(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and isinstance(value[0], (list, tuple)) and len(value[0]) == 3:
            try:
                return [[float(x) for x in row] for row in value]
            except Exception:
                return None
        if len(value) == 9 and not isinstance(value[0], (list, tuple, dict)):
            try:
                flat = [float(v) for v in value]
            except Exception:
                return None
            return [
                [flat[0], flat[1], flat[2]],
                [flat[3], flat[4], flat[5]],
                [flat[6], flat[7], flat[8]],
            ]
    return None


def _rot6d_to_matrix(rot6d: Any) -> list[list[float]] | None:
    vals = _as_float_list(rot6d)
    if vals is None or len(vals) != 6:
        return None
    a1 = vals[0:3]
    a2 = vals[3:6]

    def _norm(v: list[float]) -> list[float] | None:
        n = float((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5)
        if n <= 0.0:
            return None
        return [v[0] / n, v[1] / n, v[2] / n]

    b1 = _norm(a1)
    if b1 is None:
        return None
    dot = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2]
    a2o = [a2[0] - dot * b1[0], a2[1] - dot * b1[1], a2[2] - dot * b1[2]]
    b2 = _norm(a2o)
    if b2 is None:
        return None
    b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ]
    return [b1, b2, b3]


def _extract_rotation(det: dict[str, Any]) -> list[list[float]] | None:
    r_mat = _as_matrix_3x3(det.get("R"))
    if r_mat is not None:
        return r_mat
    return _rot6d_to_matrix(det.get("rot6d"))


def _extract_translation(det: dict[str, Any]) -> list[float] | None:
    t_xyz = _as_float_list(det.get("t_xyz"))
    if t_xyz is None or len(t_xyz) != 3:
        return None
    return [float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])]


def _extract_depth(det: dict[str, Any]) -> float | None:
    t_xyz = _extract_translation(det)
    if t_xyz is not None:
        return float(t_xyz[2])
    try:
        if det.get("log_z") is not None:
            return float(math.exp(float(det["log_z"])))
    except Exception:
        return None
    try:
        if det.get("z") is not None:
            return float(det["z"])
    except Exception:
        return None
    return None


def _load_index(path: str | Path) -> tuple[list[str], dict[str, list[_Detection]]]:
    entries = load_predictions_entries(path)
    canonical_images: list[str] = []
    image_to_dets: dict[str, list[_Detection]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        image = str(entry.get("image", ""))
        if not image:
            continue
        canonical_images.append(image)
        detections: list[_Detection] = []
        raw_detections = entry.get("detections") or []
        if isinstance(raw_detections, list):
            for det in raw_detections:
                if not isinstance(det, dict):
                    continue
                if "class_id" not in det or "score" not in det or "bbox" not in det:
                    continue
                detections.append(
                    _Detection(
                        class_id=int(det["class_id"]),
                        score=float(det["score"]),
                        bbox=dict(det["bbox"]),
                        raw=det,
                    )
                )
        add_image_aliases(image_to_dets, image, detections)
    return canonical_images, image_to_dets


def _bbox_tuple_norm(bbox: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])


def _close(a: float, b: float, atol: float) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= float(atol)


def _match_image(
    *,
    image_path: str,
    reference: list[_Detection],
    candidate: list[_Detection],
    image_size: tuple[int, int] | None,
    iou_thresh: float,
    score_atol: float,
    bbox_atol: float,
    rot_deg_atol: float,
    trans_atol: float,
    depth_atol: float,
) -> dict[str, Any]:
    if image_size is None:
        width, height = get_image_size(image_path)
    else:
        width, height = image_size

    ref_xyxy = [cxcywh_norm_to_xyxy_abs(_bbox_tuple_norm(det.bbox), width=width, height=height) for det in reference]
    cand_xyxy = [cxcywh_norm_to_xyxy_abs(_bbox_tuple_norm(det.bbox), width=width, height=height) for det in candidate]

    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for ref_idx, ref_det in enumerate(reference):
        best_idx = None
        best_iou = -1.0
        for cand_idx, cand_det in enumerate(candidate):
            if cand_idx in used or cand_det.class_id != ref_det.class_id:
                continue
            iou = iou_xyxy_abs(ref_xyxy[ref_idx], cand_xyxy[cand_idx])
            if iou > best_iou:
                best_iou = float(iou)
                best_idx = cand_idx

        if best_idx is None or float(best_iou) < float(iou_thresh):
            failures.append(
                {
                    "type": "missing_match",
                    "ref_index": int(ref_idx),
                    "class_id": int(ref_det.class_id),
                    "best_iou": None if best_idx is None else float(best_iou),
                }
            )
            continue

        used.add(int(best_idx))
        cand_det = candidate[int(best_idx)]
        ref_bbox = _bbox_tuple_norm(ref_det.bbox)
        cand_bbox = _bbox_tuple_norm(cand_det.bbox)
        bbox_ok = all(_close(a, b, bbox_atol) for a, b in zip(ref_bbox, cand_bbox))
        score_ok = _close(float(ref_det.score), float(cand_det.score), score_atol)

        ref_rot = _extract_rotation(ref_det.raw)
        cand_rot = _extract_rotation(cand_det.raw)
        ref_t = _extract_translation(ref_det.raw)
        cand_t = _extract_translation(cand_det.raw)
        ref_depth = _extract_depth(ref_det.raw)
        cand_depth = _extract_depth(cand_det.raw)

        rot_deg_diff = None
        trans_l2_diff = None
        depth_abs_diff = None
        rot_ok = True
        trans_ok = True
        depth_ok = True

        if (ref_rot is None) != (cand_rot is None):
            rot_ok = False
        elif ref_rot is not None and cand_rot is not None:
            rot_deg_diff = float(geodesic_distance(ref_rot, cand_rot) * 180.0 / math.pi)
            rot_ok = rot_deg_diff <= float(rot_deg_atol)

        if (ref_t is None) != (cand_t is None):
            trans_ok = False
        elif ref_t is not None and cand_t is not None:
            dx = float(cand_t[0]) - float(ref_t[0])
            dy = float(cand_t[1]) - float(ref_t[1])
            dz = float(cand_t[2]) - float(ref_t[2])
            trans_l2_diff = float((dx * dx + dy * dy + dz * dz) ** 0.5)
            trans_ok = trans_l2_diff <= float(trans_atol)

        if (ref_depth is None) != (cand_depth is None):
            depth_ok = False
        elif ref_depth is not None and cand_depth is not None:
            depth_abs_diff = abs(float(cand_depth) - float(ref_depth))
            depth_ok = depth_abs_diff <= float(depth_atol)

        match = {
            "ref_index": int(ref_idx),
            "cand_index": int(best_idx),
            "class_id": int(ref_det.class_id),
            "iou": float(best_iou),
            "score_ref": float(ref_det.score),
            "score_cand": float(cand_det.score),
            "score_ok": bool(score_ok),
            "bbox_ok": bool(bbox_ok),
            "rot_deg_diff": rot_deg_diff,
            "rot_ok": bool(rot_ok),
            "trans_l2_diff": trans_l2_diff,
            "trans_ok": bool(trans_ok),
            "depth_abs_diff": depth_abs_diff,
            "depth_ok": bool(depth_ok),
        }
        matches.append(match)

        if not (bbox_ok and score_ok and rot_ok and trans_ok and depth_ok):
            failures.append(
                {
                    "type": "value_mismatch",
                    "ref_index": int(ref_idx),
                    "cand_index": int(best_idx),
                    "class_id": int(ref_det.class_id),
                    "iou": float(best_iou),
                    "match": match,
                }
            )

    extras = [index for index in range(len(candidate)) if index not in used]
    return {
        "image": str(image_path),
        "size": {"width": int(width), "height": int(height)},
        "counts": {
            "ref": int(len(reference)),
            "cand": int(len(candidate)),
            "matched": int(len(matches)),
            "extra_cand": int(len(extras)),
        },
        "matches": matches,
        "extras": extras,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def compare_pose_predictions(
    *,
    reference: str | Path,
    candidate: str | Path,
    image_size: tuple[int, int] | None = None,
    max_images: int | None = None,
    iou_thresh: float = 0.99,
    score_atol: float = 1e-4,
    bbox_atol: float = 1e-4,
    rot_deg_atol: float = 1e-3,
    trans_atol: float = 1e-4,
    depth_atol: float = 1e-4,
) -> dict[str, Any]:
    reference_path = Path(reference).expanduser()
    if not reference_path.is_absolute():
        reference_path = Path.cwd() / reference_path
    candidate_path = Path(candidate).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path

    ref_images, ref_index = _load_index(reference_path)
    _, cand_index = _load_index(candidate_path)

    seen: set[str] = set()
    images: list[str] = []
    for image_key in ref_images:
        if image_key in seen:
            continue
        seen.add(image_key)
        images.append(image_key)
    if max_images is not None:
        images = images[: max(0, int(max_images))]
    if not images:
        raise ValueError("no comparable images found in reference predictions")

    per_image: list[dict[str, Any]] = []
    ok = True
    for image_key in images:
        matched = _match_image(
            image_path=image_key,
            reference=ref_index.get(image_key, []),
            candidate=cand_index.get(image_key, []),
            image_size=image_size,
            iou_thresh=float(iou_thresh),
            score_atol=float(score_atol),
            bbox_atol=float(bbox_atol),
            rot_deg_atol=float(rot_deg_atol),
            trans_atol=float(trans_atol),
            depth_atol=float(depth_atol),
        )
        per_image.append(matched)
        ok = ok and bool(matched["ok"])

    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "bbox_format": "cxcywh_norm",
        "iou_thresh": float(iou_thresh),
        "score_atol": float(score_atol),
        "bbox_atol": float(bbox_atol),
        "rot_deg_atol": float(rot_deg_atol),
        "trans_atol": float(trans_atol),
        "depth_atol": float(depth_atol),
        "images": int(len(per_image)),
        "ok": bool(ok),
        "results": per_image,
    }
