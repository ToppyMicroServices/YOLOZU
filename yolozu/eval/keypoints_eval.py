"""Keypoint detection evaluation (PCK metric).

Matches predictions to ground-truth via bbox IoU, then measures
PCK (Percentage of Correct Keypoints) at configurable thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["KeypointsMatch", "match_keypoints_detections", "evaluate_keypoints_pck"]

from yolozu.core.boxes import iou_cxcywh_norm_dict
from yolozu.core.image_keys import add_image_aliases, lookup_image_alias
from yolozu.core.keypoints import normalize_keypoints


def _bbox_iou_cxcywh_norm(a: dict[str, Any], b: dict[str, Any]) -> float:
    return iou_cxcywh_norm_dict(a, b)


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _kp_labeled(gt_kp: dict[str, Any]) -> bool:
    v = gt_kp.get("v")
    if v is None:
        return True
    try:
        return float(v) > 0.0
    except Exception:
        return True


@dataclass(frozen=True)
class KeypointsMatch:
    gt: dict[str, Any]
    pred: dict[str, Any]
    iou: float


def match_keypoints_detections(
    *,
    gt_labels: list[dict[str, Any]],
    pred_detections: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    min_score: float = 0.0,
) -> list[KeypointsMatch]:
    """Greedy match predictions to GT instances using bbox IoU (cxcywh_norm)."""

    gt = [lab for lab in gt_labels if isinstance(lab, dict) and lab.get("keypoints") is not None]
    preds = [det for det in pred_detections if isinstance(det, dict) and det.get("keypoints") is not None]

    filtered = []
    for det in preds:
        try:
            score = float(det.get("score", 0.0))
        except Exception:
            score = 0.0
        if score >= float(min_score):
            filtered.append(det)
    filtered.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)

    used_gt: set[int] = set()
    matches: list[KeypointsMatch] = []

    for det in filtered:
        bbox = det.get("bbox")
        if not isinstance(bbox, dict):
            continue
        det_cls = _as_int_or_none(det.get("class_id"))

        best_iou = 0.0
        best_j = -1
        for j, lab in enumerate(gt):
            if j in used_gt:
                continue
            lab_cls = _as_int_or_none(lab.get("class_id"))
            if det_cls is not None and lab_cls is not None and int(det_cls) != int(lab_cls):
                continue
            lab_bbox = {"cx": lab.get("cx"), "cy": lab.get("cy"), "w": lab.get("w"), "h": lab.get("h")}
            try:
                iou = _bbox_iou_cxcywh_norm(bbox, lab_bbox)
            except Exception:
                continue
            if iou > best_iou:
                best_iou = float(iou)
                best_j = int(j)

        if best_j >= 0 and best_iou >= float(iou_threshold):
            used_gt.add(int(best_j))
            matches.append(KeypointsMatch(gt=gt[best_j], pred=det, iou=float(best_iou)))

    return matches


def evaluate_keypoints_pck(
    *,
    records: list[dict[str, Any]],
    predictions_index: dict[str, list[Any]],
    iou_threshold: float = 0.5,
    pck_threshold: float = 0.1,
    min_score: float = 0.0,
    per_image_limit: int = 100,
) -> dict[str, Any]:
    warnings: list[str] = []
    per_image: list[dict[str, Any]] = []

    pred_index: dict[str, Any] = dict(predictions_index or {})
    for key, value in list(pred_index.items()):
        add_image_aliases(pred_index, key, value, overwrite_primary=False)

    total_gt_instances = 0
    total_pred_instances = 0
    total_matched_instances = 0

    labeled_total = 0
    correct_total = 0
    dist_sum = 0.0

    per_keypoint: dict[int, dict[str, Any]] = {}
    per_class: dict[int, dict[str, Any]] = {}

    def bump_keypoint(idx: int):
        entry = per_keypoint.get(idx)
        if entry is None:
            entry = {"labeled": 0, "correct": 0, "dist_sum": 0.0}
            per_keypoint[idx] = entry
        return entry

    def bump_class(cid: int):
        entry = per_class.get(cid)
        if entry is None:
            entry = {"instances_gt": 0, "instances_matched": 0, "labeled": 0, "correct": 0, "dist_sum": 0.0}
            per_class[cid] = entry
        return entry

    for record in records:
        if not isinstance(record, dict):
            continue
        image = str(record.get("image", ""))
        if not image:
            continue
        labels = record.get("labels") or []
        if not isinstance(labels, list):
            labels = []

        gt_labels = [lab for lab in labels if isinstance(lab, dict) and lab.get("keypoints") is not None]
        total_gt_instances += int(len(gt_labels))

        pred_detections = lookup_image_alias(pred_index, image)
        pred_detections = pred_detections or []
        if not isinstance(pred_detections, list):
            pred_detections = [pred_detections]
        pred_detections = [d for d in pred_detections if isinstance(d, dict) and d.get("keypoints") is not None]
        total_pred_instances += int(len(pred_detections))

        matches = match_keypoints_detections(
            gt_labels=gt_labels,
            pred_detections=pred_detections,
            iou_threshold=float(iou_threshold),
            min_score=float(min_score),
        )
        total_matched_instances += int(len(matches))

        img_labeled = 0
        img_correct = 0

        for m in matches:
            gt_kps = normalize_keypoints(m.gt.get("keypoints"), where="gt.keypoints")
            pred_kps = normalize_keypoints(m.pred.get("keypoints"), where="pred.keypoints")

            gt_bbox_w = float(m.gt.get("w", 0.0) or 0.0)
            gt_bbox_h = float(m.gt.get("h", 0.0) or 0.0)
            scale = max(gt_bbox_w, gt_bbox_h)
            if scale <= 0.0:
                warnings.append("non_positive_bbox_scale")
                continue

            class_id = _as_int_or_none(m.gt.get("class_id"))
            class_id = int(class_id) if class_id is not None else 0
            cstats = bump_class(int(class_id))
            cstats["instances_matched"] += 1

            for i, gt_kp in enumerate(gt_kps):
                if not _kp_labeled(gt_kp):
                    continue
                img_labeled += 1
                labeled_total += 1
                cstats["labeled"] += 1
                kstats = bump_keypoint(int(i))
                kstats["labeled"] += 1

                pred_kp = pred_kps[i] if i < len(pred_kps) else None
                if not isinstance(pred_kp, dict) or "x" not in pred_kp or "y" not in pred_kp:
                    continue

                dx = float(pred_kp["x"]) - float(gt_kp["x"])
                dy = float(pred_kp["y"]) - float(gt_kp["y"])
                dist = math.hypot(dx, dy)
                norm = float(dist) / float(scale)
                dist_sum += float(norm)
                cstats["dist_sum"] += float(norm)
                kstats["dist_sum"] += float(norm)

                if norm <= float(pck_threshold):
                    img_correct += 1
                    correct_total += 1
                    cstats["correct"] += 1
                    kstats["correct"] += 1

        # Count GT instances by class.
        for lab in gt_labels:
            class_id = _as_int_or_none(lab.get("class_id"))
            class_id = int(class_id) if class_id is not None else 0
            bump_class(int(class_id))["instances_gt"] += 1

        if len(per_image) < int(per_image_limit):
            per_image.append(
                {
                    "image": image,
                    "gt_instances": int(len(gt_labels)),
                    "pred_instances": int(len(pred_detections)),
                    "matched_instances": int(len(matches)),
                    "keypoints_labeled": int(img_labeled),
                    "keypoints_correct": int(img_correct),
                    "pck": (float(img_correct) / float(img_labeled) if img_labeled else None),
                }
            )

    # Summaries.
    pck = float(correct_total) / float(labeled_total) if labeled_total else None
    mean_norm_l2 = float(dist_sum) / float(labeled_total) if labeled_total else None

    per_keypoint_out: dict[str, Any] = {}
    for idx, st in sorted(per_keypoint.items(), key=lambda kv: kv[0]):
        labeled = int(st.get("labeled", 0))
        correct = int(st.get("correct", 0))
        per_keypoint_out[str(idx)] = {
            "labeled": labeled,
            "correct": correct,
            "pck": (float(correct) / float(labeled) if labeled else None),
            "mean_norm_l2": (float(st.get("dist_sum", 0.0)) / float(labeled) if labeled else None),
        }

    per_class_out: dict[str, Any] = {}
    for cid, st in sorted(per_class.items(), key=lambda kv: kv[0]):
        labeled = int(st.get("labeled", 0))
        correct = int(st.get("correct", 0))
        matched = int(st.get("instances_matched", 0))
        per_class_out[str(cid)] = {
            "instances_gt": int(st.get("instances_gt", 0)),
            "instances_matched": matched,
            "keypoints_labeled": labeled,
            "keypoints_correct": correct,
            "pck": (float(correct) / float(labeled) if labeled else None),
            "mean_norm_l2": (float(st.get("dist_sum", 0.0)) / float(labeled) if labeled else None),
        }

    return {
        "metrics": {
            "pck": pck,
            "pck_threshold": float(pck_threshold),
            "iou_threshold": float(iou_threshold),
            "min_score": float(min_score),
            "instances_gt": int(total_gt_instances),
            "instances_pred": int(total_pred_instances),
            "instances_matched": int(total_matched_instances),
            "keypoints_labeled": int(labeled_total),
            "keypoints_correct": int(correct_total),
            "mean_norm_l2": mean_norm_l2,
        },
        "per_keypoint": per_keypoint_out,
        "per_class": per_class_out,
        "per_image": per_image,
        "warnings": sorted(set(warnings)),
    }
