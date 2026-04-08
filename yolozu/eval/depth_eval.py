from __future__ import annotations

from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _require_numpy_and_pillow():
    import numpy as np
    from PIL import Image

    return np, Image


def load_depth_array(path: str | Path):
    np, Image = _require_numpy_and_pillow()
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif suffix == ".npz":
        data = np.load(p, allow_pickle=False)
        keys = list(data.files)
        if not keys:
            raise ValueError(f"npz has no arrays: {p}")
        arr = data[keys[0]]
    elif suffix in _IMAGE_SUFFIXES:
        with Image.open(p) as img:
            arr = np.asarray(img)
    else:
        raise ValueError(f"unsupported depth file suffix: {p.suffix}")

    arr = np.asarray(arr)
    if arr.ndim == 3:
        if 1 in arr.shape:
            arr = np.squeeze(arr)
        else:
            raise ValueError(f"expected single-channel depth array, got shape {tuple(arr.shape)} from {p}")
    if arr.ndim != 2:
        raise ValueError(f"expected 2D depth array, got shape {tuple(arr.shape)} from {p}")
    return arr.astype(np.float32, copy=False)


def load_mask_array(path: str | Path):
    np, _ = _require_numpy_and_pillow()
    arr = load_depth_array(path)
    return np.asarray(arr > 0, dtype=bool)


def evaluate_depth_arrays(
    *,
    pred,
    gt,
    mask=None,
    align: str = "median_scale",
    pred_scale: float = 1.0,
    gt_scale: float = 1.0,
    min_depth: float | None = 1e-6,
    max_depth: float | None = None,
) -> dict[str, Any]:
    np, _ = _require_numpy_and_pillow()
    pred_arr = np.asarray(pred, dtype=np.float32) * float(pred_scale)
    gt_arr = np.asarray(gt, dtype=np.float32) * float(gt_scale)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred_arr.shape} vs {gt_arr.shape}")

    valid = np.isfinite(pred_arr) & np.isfinite(gt_arr)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != pred_arr.shape:
            raise ValueError(f"mask shape mismatch: {mask_arr.shape} vs {pred_arr.shape}")
        valid &= mask_arr
    if min_depth is not None:
        valid &= gt_arr >= float(min_depth)
        valid &= pred_arr > 0.0
    if max_depth is not None:
        valid &= gt_arr <= float(max_depth)

    valid_count = int(valid.sum())
    if valid_count <= 0:
        raise ValueError("no valid depth pixels after masking / finite filtering")

    pred_eval = pred_arr.copy()
    scale_factor = 1.0
    if str(align).strip().lower() == "median_scale":
        pred_med = float(np.median(pred_eval[valid]))
        gt_med = float(np.median(gt_arr[valid]))
        if abs(pred_med) < 1e-12:
            raise ValueError("median_scale alignment failed because prediction median is zero")
        scale_factor = gt_med / pred_med
        pred_eval *= scale_factor
    elif str(align).strip().lower() != "none":
        raise ValueError("align must be one of: none, median_scale")

    pred_v = pred_eval[valid]
    gt_v = gt_arr[valid]
    diff = pred_v - gt_v
    abs_diff = np.abs(diff)
    gt_safe = np.maximum(gt_v, 1e-6)
    pred_safe = np.maximum(pred_v, 1e-6)
    ratio = np.maximum(pred_safe / gt_safe, gt_safe / pred_safe)

    metrics = {
        "mae": float(np.mean(abs_diff)),
        "median_abs": float(np.median(abs_diff)),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "rmse_log": float(np.sqrt(np.mean(np.square(np.log(pred_safe) - np.log(gt_safe))))),
        "abs_rel": float(np.mean(abs_diff / gt_safe)),
        "sq_rel": float(np.mean(np.square(diff) / gt_safe)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < (1.25**2))),
        "delta3": float(np.mean(ratio < (1.25**3))),
        "pred_min": float(np.min(pred_v)),
        "pred_max": float(np.max(pred_v)),
        "gt_min": float(np.min(gt_v)),
        "gt_max": float(np.max(gt_v)),
    }

    return {
        "kind": "yolozu_depth_eval_report",
        "schema_version": 1,
        "alignment": str(align).strip().lower(),
        "scale_factor": float(scale_factor),
        "counts": {
            "total_pixels": int(pred_arr.size),
            "valid_pixels": valid_count,
            "masked_out_pixels": int(pred_arr.size - valid_count),
        },
        "metrics": metrics,
    }
