"""Prediction parity comparison.

Compares two prediction files for numerical agreement by matching
detections via IoU, score tolerance, and bbox tolerance.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path

from yolozu.boxes import iou_xyxy_abs

__all__ = ["compare_predictions"]
from yolozu.image_keys import add_image_aliases
from yolozu.image_size import get_image_size
from yolozu.predictions import normalize_predictions_payload
from yolozu.predictions.bbox_formats import bbox_to_cxcywh_norm, bbox_to_xyxy_abs


@dataclass(frozen=True)
class _Detection:
    class_id: int
    score: float
    bbox: dict


def _load_index(path: str | Path) -> tuple[list[str], dict[str, list[_Detection]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries, _meta = normalize_predictions_payload(data)
    canonical_images: list[str] = []
    image_to_dets: dict[str, list[_Detection]] = {}

    for entry in entries:
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
                bbox = det.get("bbox")
                if not isinstance(bbox, dict):
                    continue
                try:
                    detections.append(_Detection(class_id=int(det["class_id"]), score=float(det["score"]), bbox=bbox))
                except (TypeError, ValueError):
                    continue

        add_image_aliases(image_to_dets, image, detections)

    return canonical_images, image_to_dets


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
    bbox_format: str,
) -> dict:
    if image_size is None:
        width, height = get_image_size(image_path)
    else:
        width, height = image_size

    ref_xyxy = [bbox_to_xyxy_abs(det.bbox, width=width, height=height, bbox_format=bbox_format) for det in reference]
    cand_xyxy = [bbox_to_xyxy_abs(det.bbox, width=width, height=height, bbox_format=bbox_format) for det in candidate]
    ref_norm = [bbox_to_cxcywh_norm(det.bbox, width=width, height=height, bbox_format=bbox_format) for det in reference]
    cand_norm = [bbox_to_cxcywh_norm(det.bbox, width=width, height=height, bbox_format=bbox_format) for det in candidate]

    used: set[int] = set()
    matches: list[dict] = []
    failures: list[dict] = []

    for ref_idx, ref_det in enumerate(reference):
        if ref_xyxy[ref_idx] is None or ref_norm[ref_idx] is None:
            failures.append(
                {
                    "type": "invalid_reference_bbox",
                    "ref_index": int(ref_idx),
                    "class_id": int(ref_det.class_id),
                    "bbox": ref_det.bbox,
                }
            )
            continue
        best_idx = None
        best_iou = -1.0
        for cand_idx, cand_det in enumerate(candidate):
            if cand_idx in used or cand_det.class_id != ref_det.class_id:
                continue
            if cand_xyxy[cand_idx] is None:
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
                    "ref_score": float(ref_det.score),
                    "best_iou": None if best_idx is None else float(best_iou),
                }
            )
            continue

        used.add(int(best_idx))
        cand_det = candidate[best_idx]
        ref_bbox = ref_norm[ref_idx]
        cand_bbox = cand_norm[best_idx]
        bbox_ok = bool(
            ref_bbox is not None
            and cand_bbox is not None
            and all(_close(a, b, bbox_atol) for a, b in zip(ref_bbox, cand_bbox))
        )
        score_ok = _close(float(ref_det.score), float(cand_det.score), score_atol)

        matches.append(
            {
                "ref_index": int(ref_idx),
                "cand_index": int(best_idx),
                "class_id": int(ref_det.class_id),
                "iou": float(best_iou),
                "score_ref": float(ref_det.score),
                "score_cand": float(cand_det.score),
                "score_ok": bool(score_ok),
                "bbox_ok": bool(bbox_ok),
            }
        )

        if not (bbox_ok and score_ok):
            failures.append(
                {
                    "type": "value_mismatch",
                    "ref_index": int(ref_idx),
                    "cand_index": int(best_idx),
                    "class_id": int(ref_det.class_id),
                    "iou": float(best_iou),
                    "ref": {"score": float(ref_det.score), "bbox": ref_det.bbox},
                    "cand": {"score": float(cand_det.score), "bbox": cand_det.bbox},
                }
            )

    extras = [index for index in range(len(candidate)) if index not in used]
    invalid_extras = [index for index in extras if cand_xyxy[index] is None or cand_norm[index] is None]
    for cand_idx in invalid_extras:
        failures.append(
            {
                "type": "invalid_candidate_bbox",
                "cand_index": int(cand_idx),
                "class_id": int(candidate[cand_idx].class_id),
                "bbox": candidate[cand_idx].bbox,
            }
        )
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
        "invalid_extras": invalid_extras,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def compare_predictions(
    *,
    reference: str | Path,
    candidate: str | Path,
    image_size: tuple[int, int] | None = None,
    max_images: int | None = None,
    iou_thresh: float = 0.99,
    score_atol: float = 1e-4,
    bbox_atol: float = 1e-4,
    bbox_format: str = "auto",
) -> dict:
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

    per_image: list[dict] = []
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
            bbox_format=str(bbox_format),
        )
        per_image.append(matched)
        ok = ok and bool(matched["ok"])

    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "bbox_format": str(bbox_format),
        "iou_thresh": float(iou_thresh),
        "score_atol": float(score_atol),
        "bbox_atol": float(bbox_atol),
        "images": int(len(per_image)),
        "ok": bool(ok),
        "results": per_image,
    }
