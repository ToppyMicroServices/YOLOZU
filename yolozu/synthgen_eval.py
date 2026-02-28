"""Synthetic-generation prediction evaluation.

Evaluates depth, keypoints, instance-segmentation, and
semantic-segmentation predictions produced by synthetic data
generation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["SynthGenEvalState", "evaluate_synthgen_predictions"]

from .segmentation_eval import compute_confusion_matrix, compute_iou_metrics


@dataclass
class SynthGenEvalState:
    samples_total: int = 0
    samples_with_predictions: int = 0
    missing_predictions: int = 0
    depth_pixels: int = 0
    depth_abs_sum: float = 0.0
    depth_sq_sum: float = 0.0
    keypoints_visible: int = 0
    keypoints_err_sum: float = 0.0
    inst_pixels: int = 0
    inst_equal_pixels: int = 0


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    raise ValueError("expected ndarray/list")


def _resolve_prediction_for_sample(sample: dict[str, Any], predictions_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    sample_id = str(sample.get("sample_id", ""))
    if sample_id and sample_id in predictions_index:
        return predictions_index[sample_id]
    image_id = str(sample.get("image_id", ""))
    if image_id and image_id in predictions_index:
        return predictions_index[image_id]
    image_path = sample.get("image_path")
    if isinstance(image_path, str) and image_path in predictions_index:
        return predictions_index[image_path]
    return None


def _accumulate_depth(gt: np.ndarray, pred: np.ndarray, state: SynthGenEvalState) -> None:
    if gt.shape != pred.shape:
        return
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    state.depth_pixels += int(diff.size)
    state.depth_abs_sum += float(np.abs(diff).sum())
    state.depth_sq_sum += float((diff * diff).sum())


def _accumulate_instances(gt: np.ndarray, pred: np.ndarray, state: SynthGenEvalState) -> None:
    if gt.shape != pred.shape:
        return
    eq = gt.astype(np.uint32, copy=False) == pred.astype(np.uint32, copy=False)
    state.inst_pixels += int(eq.size)
    state.inst_equal_pixels += int(eq.sum())


def _accumulate_keypoints(gt: np.ndarray, pred: np.ndarray, state: SynthGenEvalState) -> None:
    if gt.ndim != 3 or pred.ndim != 3:
        return
    n = min(int(gt.shape[0]), int(pred.shape[0]))
    k = min(int(gt.shape[1]), int(pred.shape[1]))
    if n <= 0 or k <= 0:
        return
    gt_slice = gt[:n, :k]
    pred_slice = pred[:n, :k]
    vis = np.rint(gt_slice[..., 2]).astype(np.int64) == 2
    if not np.any(vis):
        return
    dx = pred_slice[..., 0] - gt_slice[..., 0]
    dy = pred_slice[..., 1] - gt_slice[..., 1]
    dist = np.sqrt(dx * dx + dy * dy)
    state.keypoints_visible += int(vis.sum())
    state.keypoints_err_sum += float(dist[vis].sum())


def evaluate_synthgen_predictions(
    *,
    samples: list[dict[str, Any]],
    predictions_index: dict[str, dict[str, Any]],
    num_classes: int | None = None,
) -> dict[str, Any]:
    state = SynthGenEvalState()
    conf_total = None
    conf_classes = int(num_classes) if num_classes is not None else None

    for sample in samples:
        state.samples_total += 1
        pred = _resolve_prediction_for_sample(sample, predictions_index)
        if not isinstance(pred, dict):
            state.missing_predictions += 1
            continue
        state.samples_with_predictions += 1

        gt_depth = sample.get("depth_ndc")
        pred_depth = pred.get("depth_ndc")
        if gt_depth is not None and pred_depth is not None:
            _accumulate_depth(_as_array(gt_depth), _as_array(pred_depth), state)

        gt_inst = sample.get("inst_id")
        pred_inst = pred.get("inst_id")
        if gt_inst is not None and pred_inst is not None:
            _accumulate_instances(_as_array(gt_inst), _as_array(pred_inst), state)

        gt_kpts = sample.get("kpts2d")
        pred_kpts = pred.get("kpts2d")
        if gt_kpts is not None and pred_kpts is not None:
            _accumulate_keypoints(_as_array(gt_kpts), _as_array(pred_kpts), state)

        gt_sem = sample.get("sem_id")
        pred_sem = pred.get("sem_id")
        if gt_sem is not None and pred_sem is not None:
            gt_arr = _as_array(gt_sem)
            pred_arr = _as_array(pred_sem)
            if gt_arr.shape == pred_arr.shape:
                if conf_classes is None:
                    gt_max = int(np.max(gt_arr)) if gt_arr.size else 0
                    pred_max = int(np.max(pred_arr)) if pred_arr.size else 0
                    conf_classes = int(max(gt_max, pred_max) + 1)
                conf, _ = compute_confusion_matrix(
                    gt_arr,
                    pred_arr,
                    num_classes=max(int(conf_classes), 1),
                    ignore_index=65535,
                    allow_gt_out_of_range=True,
                )
                conf_total = conf if conf_total is None else conf_total + conf

    keypoints_mean_px = None
    if state.keypoints_visible > 0:
        keypoints_mean_px = float(state.keypoints_err_sum / float(state.keypoints_visible))

    depth_mae = None
    depth_mse = None
    if state.depth_pixels > 0:
        depth_mae = float(state.depth_abs_sum / float(state.depth_pixels))
        depth_mse = float(state.depth_sq_sum / float(state.depth_pixels))

    instance_pixel_accuracy = None
    if state.inst_pixels > 0:
        instance_pixel_accuracy = float(state.inst_equal_pixels / float(state.inst_pixels))

    segmentation = None
    if conf_total is not None:
        segmentation = compute_iou_metrics(conf_total, class_names=None, miou_ignore_background=False)

    return {
        "task": "synthgen_intake_eval",
        "summary": {
            "samples_total": state.samples_total,
            "samples_with_predictions": state.samples_with_predictions,
            "missing_predictions": state.missing_predictions,
        },
        "metrics": {
            "keypoints_mean_pixel_error": keypoints_mean_px,
            "keypoints_visible_count": state.keypoints_visible,
            "depth_mae": depth_mae,
            "depth_mse": depth_mse,
            "depth_pixel_count": state.depth_pixels,
            "instance_pixel_accuracy": instance_pixel_accuracy,
            "instance_pixel_count": state.inst_pixels,
            "segmentation": segmentation,
        },
    }
