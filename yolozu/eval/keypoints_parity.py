from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yolozu.boxes import iou_cxcywh_norm_dict
from yolozu.core.image_keys import add_image_aliases
from yolozu.keypoints import normalize_keypoints
from yolozu.predictions import load_predictions_entries

__all__ = ["compare_keypoints_predictions"]


@dataclass(frozen=True)
class _Detection:
    class_id: int
    score: float
    bbox: dict[str, Any]
    keypoints: list[dict[str, Any]]


def _close(a: float, b: float, atol: float) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= float(atol)


def _bbox_tuple_norm(bbox: dict[str, Any]) -> tuple[float, float, float, float]:
    return float(bbox["cx"]), float(bbox["cy"]), float(bbox["w"]), float(bbox["h"])


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
                if "class_id" not in det or "score" not in det or "bbox" not in det or "keypoints" not in det:
                    continue
                try:
                    keypoints = normalize_keypoints(det.get("keypoints"), where="det.keypoints")
                except Exception:
                    continue
                detections.append(
                    _Detection(
                        class_id=int(det["class_id"]),
                        score=float(det["score"]),
                        bbox=dict(det["bbox"]),
                        keypoints=keypoints,
                    )
                )
        add_image_aliases(image_to_dets, image, detections)
    return canonical_images, image_to_dets


def _match_image(
    *,
    image_key: str,
    reference: list[_Detection],
    candidate: list[_Detection],
    iou_thresh: float,
    score_atol: float,
    bbox_atol: float,
    kp_atol: float,
) -> dict[str, Any]:
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for ref_idx, ref_det in enumerate(reference):
        best_idx = None
        best_iou = -1.0
        for cand_idx, cand_det in enumerate(candidate):
            if cand_idx in used or cand_det.class_id != ref_det.class_id:
                continue
            try:
                iou = iou_cxcywh_norm_dict(ref_det.bbox, cand_det.bbox)
            except Exception:
                continue
            if iou > best_iou:
                best_iou = float(iou)
                best_idx = cand_idx

        if best_idx is None or float(best_iou) < float(iou_thresh):
            failures.append(
                {
                    "type": "missing_match",
                    "ref_index": int(ref_idx),
                    "class_id": int(ref_det.class_id),
                    "ref_score": float(ref_det.score),
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

        kp_ok = True
        kp_max = 0.0
        visibility_mismatch = False
        if len(ref_det.keypoints) != len(cand_det.keypoints):
            kp_ok = False
        else:
            for ref_kp, cand_kp in zip(ref_det.keypoints, cand_det.keypoints):
                dx = abs(float(ref_kp["x"]) - float(cand_kp["x"]))
                dy = abs(float(ref_kp["y"]) - float(cand_kp["y"]))
                kp_max = max(kp_max, float(dx), float(dy))
                ref_v = ref_kp.get("v")
                cand_v = cand_kp.get("v")
                if ref_v is not None or cand_v is not None:
                    try:
                        if ref_v is None or cand_v is None or float(ref_v) != float(cand_v):
                            visibility_mismatch = True
                    except Exception:
                        visibility_mismatch = True
                if dx > float(kp_atol) or dy > float(kp_atol):
                    kp_ok = False
        if visibility_mismatch:
            kp_ok = False

        match = {
            "ref_index": int(ref_idx),
            "cand_index": int(best_idx),
            "class_id": int(ref_det.class_id),
            "iou": float(best_iou),
            "score_ref": float(ref_det.score),
            "score_cand": float(cand_det.score),
            "score_ok": bool(score_ok),
            "bbox_ok": bool(bbox_ok),
            "keypoints_ok": bool(kp_ok),
            "visibility_ok": not visibility_mismatch,
            "keypoints_max_abs_diff": float(kp_max),
        }
        matches.append(match)

        if not (bbox_ok and score_ok and kp_ok):
            failures.append(
                {
                    "type": "value_mismatch",
                    "ref_index": int(ref_idx),
                    "cand_index": int(best_idx),
                    "class_id": int(ref_det.class_id),
                    "iou": float(best_iou),
                    "ref": {
                        "score": float(ref_det.score),
                        "bbox": ref_det.bbox,
                        "keypoints": ref_det.keypoints,
                    },
                    "cand": {
                        "score": float(cand_det.score),
                        "bbox": cand_det.bbox,
                        "keypoints": cand_det.keypoints,
                    },
                }
            )

    extras = [cand_idx for cand_idx in range(len(candidate)) if cand_idx not in used]
    return {
        "image": image_key,
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


def compare_keypoints_predictions(
    *,
    reference: str | Path,
    candidate: str | Path,
    iou_thresh: float = 0.99,
    score_atol: float = 1e-4,
    bbox_atol: float = 1e-4,
    kp_atol: float = 1e-4,
    max_images: int | None = None,
) -> dict[str, Any]:
    images, ref_index = _load_index(reference)
    _, cand_index = _load_index(candidate)
    if max_images is not None:
        images = images[: int(max_images)]
    if not images:
        raise ValueError("no images found in reference predictions")

    results: list[dict[str, Any]] = []
    ok = True
    for image in images:
        result = _match_image(
            image_key=image,
            reference=ref_index.get(image, []),
            candidate=cand_index.get(image, []),
            iou_thresh=iou_thresh,
            score_atol=score_atol,
            bbox_atol=bbox_atol,
            kp_atol=kp_atol,
        )
        results.append(result)
        ok = ok and bool(result["ok"])

    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "iou_thresh": float(iou_thresh),
        "score_atol": float(score_atol),
        "bbox_atol": float(bbox_atol),
        "kp_atol": float(kp_atol),
        "images": int(len(results)),
        "ok": bool(ok),
        "results": results,
    }
