"""Instance segmentation evaluation.

Mask loading, IoU computation, and GT-vs-prediction instance-map
matching with configurable IoU thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "load_mask_bool",
    "load_mask_int",
    "mask_iou",
    "InstanceMapResult",
    "extract_gt_instances_from_record",
    "evaluate_instance_map",
]

from yolozu.core.image_keys import image_basename, image_key_aliases, lookup_image_alias

_NUMERIC_ERRORS = (TypeError, ValueError, OverflowError)
_MASK_LOAD_ERRORS = (OSError, ValueError, TypeError, RuntimeError)


def _try_import_deps():  # pragma: no cover
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("instance-segmentation evaluation requires numpy and Pillow") from exc
    return np, Image


def load_mask_bool(value: Any, *, allow_rgb: bool = False):
    """Load a binary mask as a numpy bool array.

    value can be:
      - Path/str: image path (png recommended)
      - 2D list/tuple
      - numpy array
    """

    np, Image = _try_import_deps()

    if isinstance(value, (str, Path)):
        img = Image.open(Path(value))
        arr = np.array(img)
    else:
        arr = np.asarray(value)

    if arr.ndim == 2:
        return (arr.astype("int64", copy=False) != 0)
    if arr.ndim == 3 and allow_rgb:
        # Treat any non-zero in channel 0 as foreground. Common for grayscale stored as RGB.
        return (arr[..., 0].astype("int64", copy=False) != 0)
    raise ValueError(f"mask must be 2D binary image; got shape={getattr(arr, 'shape', None)}")


def load_mask_int(value: Any, *, allow_rgb: bool = False):
    """Load an integer mask as numpy int64 array (for instance-id or class-id masks)."""
    np, Image = _try_import_deps()
    if isinstance(value, (str, Path)):
        img = Image.open(Path(value))
        arr = np.array(img)
    else:
        arr = np.asarray(value)
    if arr.ndim == 2:
        return arr.astype("int64", copy=False)
    if arr.ndim == 3 and allow_rgb:
        return arr[..., 0].astype("int64", copy=False)
    raise ValueError(f"mask must be 2D integer image; got shape={getattr(arr, 'shape', None)}")


def mask_iou(a_bool, b_bool) -> float:
    np, _ = _try_import_deps()
    a = np.asarray(a_bool, dtype=bool)
    b = np.asarray(b_bool, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: a={a.shape} b={b.shape}")
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _connected_components(mask_bool) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes of connected components in a bool mask.

    This is intentionally simple (4-connectivity) and used only for small masks.
    """

    np, _ = _try_import_deps()
    m = np.asarray(mask_bool, dtype=bool)
    if m.ndim != 2:
        raise ValueError("connected_components expects 2D mask")
    h, w = m.shape
    visited = np.zeros((h, w), dtype=bool)
    bboxes: list[tuple[int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not bool(m[y, x]) or bool(visited[y, x]):
                continue
            stack = [(y, x)]
            visited[y, x] = True
            y_min = y_max = y
            x_min = x_max = x
            while stack:
                cy, cx = stack.pop()
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)
                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and bool(m[ny, nx]) and not bool(visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes


@dataclass(frozen=True)
class InstanceMapResult:
    map50: float
    map50_95: float
    per_class: dict[int, dict[str, float]]
    counts: dict[str, int]
    warnings: list[str]
    per_image: list[dict[str, Any]] = field(default_factory=list)


def _compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls or not precisions:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return float(ap)


def extract_gt_instances_from_record(
    record: dict[str, Any],
    *,
    allow_rgb_masks: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract GT instance masks from a YOLOZU dataset record.

    Supported GT formats:
      A) Per-instance PNG list:
         - mask_path: ["/abs/or/rel.png", ...]
         - mask_classes: [0, 1, ...] (same length)
      B) Single instance-id PNG:
         - mask_path: "/abs/or/rel.png"
         - mask_format: "instance"
         - optional mask_class_map: {"1": 3, "2": 7, ...}
         - optional mask_class_id: 3  (fallback class for all instances)
    """

    warnings: list[str] = []
    mask_value = record.get("mask_path")
    if mask_value is None:
        mask_value = record.get("mask")
    if mask_value is None:
        mask_value = record.get("M")

    if mask_value is None:
        return [], warnings

    mask_format = record.get("mask_format")
    mask_instances = bool(record.get("mask_instances", False))
    mask_classes = record.get("mask_classes")
    mask_class_id = record.get("mask_class_id")
    mask_class_map = record.get("mask_class_map")

    instances: list[dict[str, Any]] = []

    if isinstance(mask_value, (list, tuple)):
        if not mask_value:
            return [], warnings
        if not isinstance(mask_classes, (list, tuple)) or len(mask_classes) != len(mask_value):
            if mask_class_id is None:
                warnings.append("mask_path is a list but mask_classes is missing or wrong length; skipping")
                return [], warnings
            mask_classes = [mask_class_id for _ in mask_value]
        for item, cid in zip(mask_value, mask_classes):
            try:
                class_id = int(cid)
            except _NUMERIC_ERRORS:
                warnings.append("invalid mask_classes entry; skipping an instance")
                continue
            try:
                m = load_mask_bool(item, allow_rgb=allow_rgb_masks)
            except _MASK_LOAD_ERRORS as exc:
                warnings.append(f"failed to load gt mask: {exc}")
                continue
            instances.append({"class_id": class_id, "mask": m})
        return instances, warnings

    # Single mask.
    if isinstance(mask_format, str) and mask_format.lower() == "instance":
        try:
            arr = load_mask_int(mask_value, allow_rgb=allow_rgb_masks)
        except _MASK_LOAD_ERRORS as exc:
            warnings.append(f"failed to load instance-id gt mask: {exc}")
            return [], warnings
        unique = [int(v) for v in set(int(x) for x in arr.reshape(-1).tolist()) if int(v) != 0]
        unique.sort()

        class_map: dict[int, int] = {}
        if isinstance(mask_class_map, dict):
            for k, v in mask_class_map.items():
                try:
                    class_map[int(k)] = int(v)
                except _NUMERIC_ERRORS:
                    continue
        try:
            fallback_class = int(mask_class_id) if mask_class_id is not None else 0
        except _NUMERIC_ERRORS:
            fallback_class = 0
            warnings.append("invalid mask_class_id fallback; using class_id=0")

        for inst_id in unique:
            m = (arr == int(inst_id))
            class_id = int(class_map.get(int(inst_id), fallback_class))
            instances.append({"class_id": class_id, "mask": m})
        return instances, warnings

    # Optional: interpret a class-id mask and split connected components into instances.
    if mask_instances:
        try:
            arr = load_mask_int(mask_value, allow_rgb=allow_rgb_masks)
        except _MASK_LOAD_ERRORS as exc:
            warnings.append(f"failed to load gt mask: {exc}")
            return [], warnings
        class_map: dict[int, int] = {}
        if isinstance(mask_class_map, dict):
            for k, v in mask_class_map.items():
                try:
                    class_map[int(k)] = int(v)
                except _NUMERIC_ERRORS:
                    continue
        for class_val in sorted(int(v) for v in set(int(x) for x in arr.reshape(-1).tolist()) if int(v) != 0):
            class_id = int(class_map.get(int(class_val), int(class_val)))
            m_class = (arr == int(class_val))
            for bbox in _connected_components(m_class):
                x0, y0, x1, y1 = bbox
                m = m_class.copy()
                m[:y0, :] = False
                m[y1 + 1 :, :] = False
                m[:, :x0] = False
                m[:, x1 + 1 :] = False
                instances.append({"class_id": class_id, "mask": m})
        return instances, warnings

    return [], warnings


def _group_instances(
    *,
    items: Iterable[dict[str, Any]],
    image_key: str,
    class_key: str = "class_id",
    allow_rgb_masks: bool = False,
    mask_key: str = "mask",
    score_key: str = "score",
) -> tuple[list[dict[str, Any]], set[int], list[str]]:
    warnings: list[str] = []
    out: list[dict[str, Any]] = []
    classes: set[int] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        image = it.get(image_key)
        if not isinstance(image, str) or not image:
            continue
        try:
            class_id = int(it.get(class_key))
        except _NUMERIC_ERRORS:
            continue
        score = it.get(score_key, 1.0)
        try:
            score_f = float(score)
        except _NUMERIC_ERRORS:
            score_f = 1.0

        mask_val = it.get(mask_key)
        if mask_val is None:
            continue

        out.append(
            {
                "image": image,
                "class_id": int(class_id),
                "score": float(score_f),
                "mask": mask_val,
            }
        )
        classes.add(int(class_id))
    return out, classes, warnings


def evaluate_instance_map(
    *,
    records: list[dict[str, Any]],
    predictions_entries: Iterable[dict[str, Any]],
    iou_thresholds: Iterable[float] = tuple(0.5 + 0.05 * i for i in range(10)),
    min_score: float = 0.0,
    pred_root: Path | None = None,
    allow_rgb_masks: bool = False,
    return_per_image: bool = False,
    diagnostics_iou: float = 0.5,
) -> InstanceMapResult:
    """Evaluate instance segmentation with a COCO-like AP computation over binary PNG masks.

    This implementation is CPU-friendly and does not require pycocotools.
    """

    np, _ = _try_import_deps()

    if pred_root is not None:
        pred_root = Path(pred_root).resolve()

    thresholds = [float(x) for x in iou_thresholds]
    if not thresholds:
        thresholds = [0.5]

    # Cache loaded masks by absolute path string.
    mask_cache: dict[str, Any] = {}

    def _resolve_mask_path(p: Any) -> Any:
        if p is None:
            return None
        if isinstance(p, (str, Path)):
            path = Path(p)
            if not path.is_absolute():
                if pred_root is not None:
                    path = pred_root / path
            return str(path)
        return p

    def _load_pred_mask(mask_val: Any):
        key = _resolve_mask_path(mask_val)
        if isinstance(key, str):
            if key in mask_cache:
                return mask_cache[key]
            m = load_mask_bool(key, allow_rgb=allow_rgb_masks)
            mask_cache[key] = m
            return m
        # Inline array
        return load_mask_bool(key, allow_rgb=allow_rgb_masks)

    # Build GT map: image -> class -> list[mask_bool]
    gt_by_image: dict[str, dict[int, list[Any]]] = {}
    gt_classes: set[int] = set()
    gt_instances_total = 0
    warnings: list[str] = []

    for record in records:
        image = str(record.get("image", ""))
        if not image:
            continue
        gt_insts, w = extract_gt_instances_from_record(record, allow_rgb_masks=allow_rgb_masks)
        warnings.extend(w)
        per_image: dict[int, list[Any]] = {}
        for inst in gt_insts:
            class_id = int(inst.get("class_id", 0))
            mask = inst.get("mask")
            if mask is None:
                continue
            try:
                m = np.asarray(mask, dtype=bool)
            except (TypeError, ValueError):
                continue
            per_image.setdefault(class_id, []).append(m)
            gt_classes.add(class_id)
            gt_instances_total += 1

        entry = gt_by_image.setdefault(image, {})
        for cid, masks in per_image.items():
            entry.setdefault(int(cid), []).extend(list(masks))

        for alias in image_key_aliases(image):
            gt_by_image.setdefault(alias, entry)

    # Flatten predictions instances.
    from yolozu.instance_segmentation_predictions import iter_instances, validate_instance_segmentation_predictions_entries

    pred_entries = list(predictions_entries)
    validation = validate_instance_segmentation_predictions_entries(pred_entries, where="predictions")
    warnings.extend(list(validation.warnings))

    pred_flat, pred_classes, w = _group_instances(items=iter_instances(pred_entries), image_key="image", allow_rgb_masks=allow_rgb_masks)
    warnings.extend(w)
    # Resolve mask paths now so caching works.
    for p in pred_flat:
        p["mask"] = _resolve_mask_path(p.get("mask"))

    pred_flat = [p for p in pred_flat if float(p.get("score", 0.0)) >= float(min_score)]

    classes = sorted(gt_classes.union(pred_classes))

    def _ap_for_class(*, class_id: int, thresh: float) -> float:
        gt_used: dict[str, list[bool]] = {}
        gt_count = 0
        for record in records:
            image_key = str(record.get("image", ""))
            if not image_key:
                continue
            image_gt = lookup_image_alias(gt_by_image, image_key) or {}
            masks = image_gt.get(int(class_id), [])
            used = [False] * len(masks)
            for alias in image_key_aliases(image_key):
                gt_used.setdefault(alias, used)
            gt_count += len(masks)
        if gt_count == 0:
            return 0.0

        class_preds = [p for p in pred_flat if int(p["class_id"]) == int(class_id)]
        class_preds.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)

        tp: list[int] = []
        fp: list[int] = []

        for pred in class_preds:
            image_key = str(pred["image"])
            image_gt = lookup_image_alias(gt_by_image, image_key) or {}
            gt_masks = image_gt.get(int(class_id), [])
            used = lookup_image_alias(gt_used, image_key) or []

            try:
                pm = _load_pred_mask(pred["mask"])
            except _MASK_LOAD_ERRORS:
                tp.append(0)
                fp.append(1)
                continue

            best_iou = 0.0
            best_idx = -1
            for idx, gm in enumerate(gt_masks):
                if idx < len(used) and used[idx]:
                    continue
                try:
                    iou = mask_iou(pm, gm)
                except (TypeError, ValueError):
                    continue
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= float(thresh) and best_idx >= 0 and best_idx < len(used) and not used[best_idx]:
                used[best_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        cum_tp = 0
        cum_fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for i in range(len(tp)):
            cum_tp += tp[i]
            cum_fp += fp[i]
            recall = float(cum_tp) / float(max(1, gt_count))
            precision = float(cum_tp) / float(max(1, cum_tp + cum_fp))
            recalls.append(recall)
            precisions.append(precision)
        return _compute_ap(recalls, precisions)

    per_class: dict[int, dict[str, float]] = {cid: {} for cid in classes}
    for thresh in thresholds:
        for cid in classes:
            ap = _ap_for_class(class_id=int(cid), thresh=float(thresh))
            per_class[int(cid)][f"ap@{float(thresh):.2f}"] = float(ap)

    map50 = 0.0
    map50_95 = 0.0
    if classes:
        map50 = sum(per_class[cid].get("ap@0.50", 0.0) for cid in classes) / float(len(classes))
        map50_95 = sum(
            sum(per_class[cid].values()) / float(len(thresholds)) for cid in classes
        ) / float(len(classes))

    per_image: list[dict[str, Any]] = []
    if return_per_image:
        pred_by_image: dict[str, list[dict[str, Any]]] = {}
        for p in pred_flat:
            key = str(p.get("image", ""))
            if not key:
                continue
            aliases = image_key_aliases(key)
            if not aliases:
                continue
            bucket = pred_by_image.setdefault(aliases[0], [])
            bucket.append(p)
            for alias in aliases[1:]:
                pred_by_image.setdefault(alias, bucket)

        diag_thresh = float(diagnostics_iou)
        for record in records:
            image_path = str(record.get("image", ""))
            if not image_path:
                continue
            base = image_basename(image_path) or image_path

            # Collect predictions for either full path key or basename key.
            preds: list[dict[str, Any]] = []
            seen: set[tuple[str, int, float]] = set()
            for key in image_key_aliases(image_path):
                for it in pred_by_image.get(str(key), []) or []:
                    try:
                        sig = (str(it.get("mask", "")), int(it.get("class_id", 0)), float(it.get("score", 0.0)))
                    except _NUMERIC_ERRORS:
                        sig = (str(it.get("mask", "")), 0, 0.0)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    preds.append(it)

            # GT map for this record (already loaded).
            gt_map = lookup_image_alias(gt_by_image, image_path) or {}

            gt_total = sum(len(v) for v in gt_map.values())
            pred_total = int(len(preds))

            # Match per-class independently (greedy by score, like AP@IoU).
            tp = 0
            fp = 0
            ious: list[float] = []

            class_ids = sorted(set(int(k) for k in gt_map.keys()).union(int(p.get("class_id", 0)) for p in preds))
            for cid in class_ids:
                gt_masks = list(gt_map.get(int(cid), []) or [])
                used = [False] * len(gt_masks)
                cls_preds = [p for p in preds if int(p.get("class_id", 0)) == int(cid)]
                cls_preds.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)
                for pred in cls_preds:
                    try:
                        pm = _load_pred_mask(pred.get("mask"))
                    except _MASK_LOAD_ERRORS:
                        fp += 1
                        continue
                    best_iou = 0.0
                    best_idx = -1
                    for idx, gm in enumerate(gt_masks):
                        if idx < len(used) and used[idx]:
                            continue
                        try:
                            iou = mask_iou(pm, gm)
                        except (TypeError, ValueError):
                            continue
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    if best_iou >= diag_thresh and best_idx >= 0 and best_idx < len(used) and not used[best_idx]:
                        used[best_idx] = True
                        tp += 1
                        ious.append(float(best_iou))
                    else:
                        fp += 1

            fn = int(max(0, int(gt_total) - int(tp)))

            denom = float(gt_total) if int(gt_total) > 0 else float(max(1, pred_total))
            mean_iou = (sum(ious) / float(len(ious))) if ious else None
            precision = float(tp) / float(max(1, tp + fp)) if (tp + fp) > 0 else None
            recall = float(tp) / float(max(1, tp + fn)) if (tp + fn) > 0 else None
            badness = int(fp + fn)
            error_rate = float(badness) / float(denom)

            per_image.append(
                {
                    "image": base,
                    "image_path": image_path,
                    "gt_instances": int(gt_total),
                    "pred_instances": int(pred_total),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "precision": float(precision) if precision is not None else None,
                    "recall": float(recall) if recall is not None else None,
                    "mean_iou": float(mean_iou) if mean_iou is not None else None,
                    "badness": int(badness),
                    "error_rate": float(error_rate),
                    "iou_thresh": float(diag_thresh),
                }
            )

    return InstanceMapResult(
        map50=float(map50),
        map50_95=float(map50_95),
        per_class=per_class,
        per_image=per_image,
        counts={
            "images": int(len(records)),
            "gt_instances": int(gt_instances_total),
            "pred_instances": int(len(pred_flat)),
            "classes": int(len(classes)),
        },
        warnings=warnings,
    )
