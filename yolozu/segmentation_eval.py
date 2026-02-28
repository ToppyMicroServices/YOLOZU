"""Semantic-segmentation IoU / mIoU evaluation.

Computes per-class IoU metrics from ground-truth and predicted
masks via confusion matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "load_mask_array",
    "ConfusionStats",
    "compute_confusion_matrix",
    "compute_iou_metrics",
]


def _try_import_deps():  # pragma: no cover
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("segmentation evaluation requires numpy and Pillow") from exc
    return np, Image


def load_mask_array(path, *, allow_rgb: bool = False):
    np, Image = _try_import_deps()
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and allow_rgb:
        if arr.shape[2] >= 1:
            if arr.shape[2] >= 3:
                # Common case: grayscale stored as RGB.
                if (arr[..., 0] == arr[..., 1]).all() and (arr[..., 0] == arr[..., 2]).all():
                    return arr[..., 0]
            return arr[..., 0]
    raise ValueError(f"mask must be 2D class-id image; got shape={getattr(arr, 'shape', None)} path={path}")


@dataclass(frozen=True)
class ConfusionStats:
    pixels_total: int
    pixels_ignored: int
    pixels_gt_out_of_range: int
    pixels_pred_out_of_range: int
    pixels_valid: int
    pixels_mismatched: int


def compute_confusion_matrix(
    gt,
    pred,
    *,
    num_classes: int,
    ignore_index: int = 255,
    allow_gt_out_of_range: bool = False,
) -> tuple[Any, ConfusionStats]:
    """Compute a confusion matrix for a (gt, pred) pair.

    - Ground-truth pixels equal to ignore_index are excluded.
    - GT labels must be in [0, num_classes-1] (unless allow_gt_out_of_range=True; then OOR GT pixels are ignored).
    - Pred labels outside [0, num_classes-1] are mapped to an extra 'unknown' column (index num_classes).

    Returns:
      conf: shape (num_classes, num_classes+1) where last column is 'unknown pred'.
    """

    np, _ = _try_import_deps()
    gt_arr = np.asarray(gt)
    pred_arr = np.asarray(pred)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"gt/pred shape mismatch: gt={gt_arr.shape} pred={pred_arr.shape}")
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be > 0")

    gt_i = gt_arr.astype("int64", copy=False)
    pred_i = pred_arr.astype("int64", copy=False)

    pixels_total = int(gt_i.size)
    ignore_index = int(ignore_index)

    ignored = (gt_i == ignore_index)
    pixels_ignored = int(ignored.sum())

    valid = ~ignored
    gt_in_range = (gt_i >= 0) & (gt_i < int(num_classes))
    gt_out = valid & ~gt_in_range
    pixels_gt_out = int(gt_out.sum())
    if pixels_gt_out and not bool(allow_gt_out_of_range):
        raise ValueError(
            "ground-truth contains labels outside [0,num_classes-1] (and not ignore_index): "
            f"count={pixels_gt_out} num_classes={num_classes} ignore_index={ignore_index}"
        )
    valid = valid & gt_in_range

    valid_gt = gt_i[valid]
    valid_pred = pred_i[valid]

    pred_in_range = (valid_pred >= 0) & (valid_pred < int(num_classes))
    pixels_pred_out = int((~pred_in_range).sum())
    unknown = int(num_classes)
    valid_pred_mapped = np.where(pred_in_range, valid_pred, unknown).astype("int64", copy=False)

    mismatched = int((valid_gt != valid_pred_mapped).sum())
    pixels_valid = int(valid_gt.size)

    k = int(num_classes) + 1
    # Confusion: rows=gt (0..num_classes-1), cols=pred (0..num_classes plus unknown).
    idx = valid_gt * k + valid_pred_mapped
    conf = np.bincount(idx, minlength=int(num_classes) * k).reshape(int(num_classes), k)

    stats = ConfusionStats(
        pixels_total=pixels_total,
        pixels_ignored=pixels_ignored,
        pixels_gt_out_of_range=pixels_gt_out,
        pixels_pred_out_of_range=pixels_pred_out,
        pixels_valid=pixels_valid,
        pixels_mismatched=mismatched,
    )
    return conf, stats


def compute_iou_metrics(
    conf,
    *,
    class_names: list[str] | None,
    miou_ignore_background: bool = False,
) -> dict[str, Any]:
    """Compute (mIoU, per-class IoU, pixel accuracy) from confusion matrix."""

    np, _ = _try_import_deps()
    conf_arr = np.asarray(conf)
    if conf_arr.ndim != 2:
        raise ValueError("confusion matrix must be 2D")
    num_classes = int(conf_arr.shape[0])
    if conf_arr.shape[1] not in (num_classes, num_classes + 1):
        raise ValueError(f"unexpected confusion matrix shape: {conf_arr.shape}")

    k = int(conf_arr.shape[1])
    unknown_col = num_classes if k == num_classes + 1 else None

    tp = np.array([conf_arr[i, i] for i in range(num_classes)], dtype="int64")
    gt_pixels = conf_arr.sum(axis=1).astype("int64")
    pred_pixels = conf_arr[:, :num_classes].sum(axis=0).astype("int64")
    fp = pred_pixels - tp
    fn = gt_pixels - tp
    union = tp + fp + fn

    iou: list[float | None] = []
    for i in range(num_classes):
        denom = int(union[i])
        if denom <= 0:
            iou.append(None)
        else:
            iou.append(float(tp[i]) / float(denom))

    # Decide which classes to include in mIoU.
    include = [i for i in range(num_classes) if union[i] > 0]
    if (
        bool(miou_ignore_background)
        and class_names
        and len(class_names) >= 1
        and str(class_names[0]).lower() == "background"
        and 0 in include
    ):
        include = [i for i in include if i != 0]

    miou: float | None
    if not include:
        miou = None
    else:
        vals = [v for i, v in enumerate(iou) if i in include and v is not None]
        miou = float(sum(vals) / float(len(vals))) if vals else None

    total_valid = int(conf_arr.sum())
    pixel_acc = float(tp.sum()) / float(total_valid) if total_valid > 0 else None

    classes_out: list[dict[str, Any]] = []
    for i in range(num_classes):
        name = str(i) if not class_names or i >= len(class_names) else str(class_names[i])
        classes_out.append(
            {
                "id": int(i),
                "name": name,
                "iou": iou[i],
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "gt_pixels": int(gt_pixels[i]),
                "pred_pixels": int(pred_pixels[i]),
                "union": int(union[i]),
                "included_in_miou": bool(i in include and iou[i] is not None),
            }
        )

    pred_unknown_pixels = None
    if unknown_col is not None:
        pred_unknown_pixels = int(conf_arr[:, unknown_col].sum())

    return {
        "miou": miou,
        "pixel_accuracy": pixel_acc,
        "num_classes": int(num_classes),
        "classes": classes_out,
        "pred_unknown_pixels": pred_unknown_pixels,
        "pixels_total": int(total_valid),
    }

